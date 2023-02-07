import json
import numpy as np
import random 
import os
import torch
import math
import itertools
import logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
)
logger = logging.getLogger(__name__)

def set_seeding(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def write_json(file_path, data):
    with open(file_path, 'w') as fp:
        json.dump(data, fp)
    
def load_json(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    return data

def save_checkpoint_model(model, output_dir, model_name, epoch, alpha, tokenizer=None):
    # Saving best-practices: if you use save_pretrained for the model and tokenizer, 
    # you can reload them using from_pretrained()
    output_path = os.path.join(output_dir, 'checkpoint-%s' % str(epoch), 'model_%s' % model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_to_save = (model.module if hasattr(model, "module") else model)# Take care of distributed/parallel training
    
    if model_name == 'main' or model_name == 'aux': 
        model_to_save.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        #torch.save({'model_state_dict': model_to_save.state_dict()}, model_path)
        logger.info('Save model to %s' % (output_path))
        
    elif model_name == 'net1' or model_name == 'net2':
        output_path = output_path + '/' + model_name
        torch.save({'model_state_dict': model_to_save.state_dict()}, output_path)
        logger.info('Save model to %s' % (output_path))

 
        
def load_checkpoint_model_gpt2(model_name, load_dir, epoch, alpha, do_lower_case, 
                          device, model_net=None):
    """
    load checkpoint model from GPT2
    """
    load_path = os.path.join(load_dir, 'checkpoint-%s' % str(epoch), 'model_%s' % model_name)
    if not os.path.exists(load_path):
        print("The path does not exist!")
        
    if model_name == 'main' or model_name == 'aux': 
        # Select model class
        MODEL_CLASSES = {"gpt2-main": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                         "gpt2-aux": (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
                         "bert": (BertConfig, BertForSequenceClassification, BertTokenizer)}
        config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2-" + model_name]
        
        # use the model class to load the model architecture
        model = model_class.from_pretrained(load_path)
        tokenizer = tokenizer_class.from_pretrained(load_path, do_lower_case=do_lower_case)
        
        return model.to(device), tokenizer
    
    elif model_name == 'net1' or model_name == 'net2':
        # load model
        model_path = load_path+'/' + model_name
        checkpoint = torch.load(model_path)
        
        # load model achitecture
        model_net= (model_net.module if hasattr(model_net, "module") else model_net)  # Take care of distributed/parallel training
        model = model_net
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)      


# 1 - 100: Dis, [1- 100, 201 - 300,]
# 101- 200: gen, [101- 200]
def NetSwitchSchedule(dataloader, switch_freq=100):
    '''
    The discriminator and generator training are switched every 20 batches.
    discriminator_train: list
    generator_train: list
    '''
    discriminator_train = range(1, math.ceil(len(dataloader)/switch_freq)+2, 2)
    generator_train = range(2, math.ceil(len(dataloader)/switch_freq)+2, 2)

    discriminator_train = [list(range((i-1)*switch_freq, i*switch_freq)) for i in list(discriminator_train)]
    discriminator_train = list(itertools.chain(*discriminator_train))
    generator_train = [range((i-1)*switch_freq, i*switch_freq) for i in list(generator_train)]
    generator_train = list(itertools.chain(*generator_train))
    
    return discriminator_train, generator_train

def NetSwitchScheduleNew(dataloader, switch_freq_dis, switch_freq_gen):
    discriminator_train = []
    generator_train = []

    count_dis = 0
    count_gen = 0

    i = 0
    while i < len(dataloader):
        # start from discriminator:
        if count_dis != switch_freq_dis:
            discriminator_train.append(i)
            count_dis +=1
            i += 1
        else:
            count_dis = 0
            # switch to generator:
            while count_gen < switch_freq_gen:
                generator_train.append(i)
                i += 1
                count_gen +=1            
            count_gen = 0     
    return discriminator_train, generator_train

# calculate cross entropy loss
def classification_loss(mlp_prediction, labels):
    """
    input:
        labels: fair_label
        mlp_prediction: batch_size * categories
        
    output:
        loss_cls: classification_loss
    """
    cross_loss_function = torch.nn.CrossEntropyLoss()
    loss_cls = cross_loss_function(mlp_prediction, labels)
    return loss_cls


def CalculateAttention(lm_emb, cls_emb, net1, net2, device, block_size):
    '''
    inputs:
        lm_emb: batch_size * embedding_size * block_size
        cls_emb: batch_size * embedding_size
    
    outputs:
        attention: batch_size * embedding_size/2
    '''
    
    # define cosine similarity function
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    batch_size = lm_emb.size(0)
    embedding_size = lm_emb.size(1)

    # transform the orginal embedding size to half of the original one:  
    #lm_emb = torch.transpose(net1(torch.transpose(lm_emb,1,2)), 1,2) # 256 * 768 -> 256 * 384
    #cls_emb = net2(cls_emb) # 1 * 768 -> 1 * 384
    lm_key = torch.transpose(net1(torch.transpose(lm_emb,1,2), 'key'), 1,2) # 256 * 768 -> 256 * 768
    lm_value = torch.transpose(net1(torch.transpose(lm_emb,1,2), 'value'), 1,2) # 256 * 768 -> 256 * 768
    cls_query = net2(cls_emb, 'query') # 1 * 768 -> 1 * 768

    sim_matrix = torch.zeros((batch_size, block_size)).to(device)  # 4 * 256
    weight_const = torch.zeros((batch_size, 1)).to(device) # 4 * 1
    cross_attention = torch.zeros((batch_size, embedding_size)).to(device) # 4 * 384

    for i in range(batch_size):
        for j in range(block_size):
            sim_matrix[i][j] = cos(lm_key[i][:,j], cls_query[i])

    # weight constant
    for i in range(batch_size):
        for j in range(block_size):
            if sim_matrix[i][j] != 0:
                sim = torch.exp(sim_matrix[i][j])
                weight_const[i] += sim
                sim_matrix[i][j] = sim

    # alpha weight
    weights = sim_matrix / weight_const

    for i in range(batch_size):
        cross_attention[i] = torch.matmul(weights.unsqueeze(1)[i], torch.transpose(lm_value[i], 0,1))
    return cross_attention # 4 * 768



def load_checkpoint_model_bert(model_name, load_dir, epoch, alpha, do_lower_case, 
                          device, model_net=None):
    """
    load checkpoint model from Roberta
    """
    load_path = os.path.join(load_dir, 'checkpoint-%s' % str(epoch), 'model_%s' % model_name)

    if not os.path.exists(load_path):
        print("The path does not exist!")
        
    if model_name == 'main':
        # Select model class
        MODEL_CLASSES = {"gpt2-main": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}
        config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2-" + model_name]
        
        # use the model class to load the model architecture
        model = model_class.from_pretrained(load_path)
        tokenizer = tokenizer_class.from_pretrained(load_path, do_lower_case=do_lower_case)
        
        return model.to(device), tokenizer
    
    elif model_name == 'aux': 
        # Select model class
        MODEL_CLASSES = {"bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
                         "roberta-base-aux": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta-base-" + model_name]
        
        # use the model class to load the model architecture
        model = model_class.from_pretrained(load_path)
        tokenizer = tokenizer_class.from_pretrained(load_path, do_lower_case=do_lower_case)
        
        return model.to(device), tokenizer
    
    elif model_name == 'net1' or model_name == 'net2':
        # load model
        model_path = load_path+'/' + model_name
        checkpoint = torch.load(model_path)
        
        # load model achitecture
        model_net= (model_net.module if hasattr(model_net, "module") else model_net)  # Take care of distributed/parallel training
        model = model_net
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)      



def save_loss_plot(args, epoch, total_cls_loss, Generator_lm_loss, Generator_classification_loss, total_adv_loss):
    
    # output_total_cls_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_switch_%d_Dis_CLS_loss.json' % \
    #             (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq)
    
    # output_Generator_lm_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_switch_%d_Gen_lm_loss.json' % \
    #                 (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq)

    # output_Generator_classification_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_switch_%d_Gen_CLS_loss.json' % \
    #                 (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq)

    # output_total_adv_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_switch_%d_Total_adv_loss.json' % \
    #                 (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq)
    
    output_total_cls_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_Dis_CLS_loss_D_%s_G_%s.json' % \
                    (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq_dis, args.switch_freq_gen)
    
    output_Generator_lm_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_Gen_lm_loss_D_%s_G_%s.json' % \
                    (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq_dis, args.switch_freq_gen)

    output_Generator_classification_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_Gen_CLS_loss_D_%s_G_%s.json' % \
                    (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq_dis, args.switch_freq_gen)

    output_total_adv_loss_path = '../loss_plot/sample_%d_model_%s_load_%s_alpha_%s_epoch_%d_Total_adv_loss_D_%s_G_%s.json' % \
                    (args.sub_sample_size, args.model_name_or_path_aux, args.load_pretrained_cls, str(args.alpha), epoch, args.switch_freq_dis, args.switch_freq_gen)


    write_json(output_total_cls_loss_path, total_cls_loss)
    write_json(output_Generator_lm_loss_path, Generator_lm_loss)
    write_json(output_Generator_classification_loss_path, Generator_classification_loss)
    write_json(output_total_adv_loss_path, total_adv_loss)

    