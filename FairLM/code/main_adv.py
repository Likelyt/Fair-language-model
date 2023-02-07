from __future__ import print_function, division
from sys import int_info
#%matplotlib inline
from matplotlib import pyplot as plt
import argparse
import logging
import json
import numpy as np
import random
import scipy.stats
import logging
import os
import itertools
import math

import datasets
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
#from tqdm import tqdm, trange
from tqdm.notebook import tqdm, trange
from datasets import list_datasets, load_dataset, list_metrics, load_metric
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split

from utils.modules import NetLM, NetAdv, hyper_parameters, hyper_parameters_net
from utils.util import set_seeding, load_json, NetSwitchSchedule, classification_loss, save_checkpoint_model, load_checkpoint_model_gpt2, load_checkpoint_model_bert
from utils.data_processing import OWTCDatasetMain, Gpt2ClassificationCollatorMain, DataCollator, Batches_gpt2, Batches_Bert, label_id_mapping_gpt2, label_id_mapping_bert, CLSDataset, GAN_Dataset
from utils.train_and_val import model_train_gpt2, model_validation_gpt2, model_train_bert, model_validation_bert



from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
)

logger = logging.getLogger(__name__)

def load_pretrained_model_init(model_name_or_path, block_size, do_lower_case = True, n_labels = 17):
    # load all mask tokens
    descriptors_path = "../data/mask_token/mask_all.json"
    word_label_path = "../data/mask_token/word_label.json"

    descriptors = load_json(descriptors_path) # [daughter: female]
    descriptors = [desc.lower() for desc in descriptors]

    word_label = load_json(word_label_path) # {'daughter': female}
    word_label = {key.lower(): val.lower() for key,val in word_label.items()}

    # Step 2.1: Select model class
    MODEL_CLASSES = {"gpt2-main": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                    "gpt2-aux": (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
                    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
                    "roberta-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
    if model_name_or_path == 'gpt2-main':
        
        config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        model_config = config_class.from_pretrained(pretrained_model_name_or_path='gpt2', output_hidden_states=True)
        
        tokenizer = tokenizer_class.from_pretrained('gpt2', do_lower_case=do_lower_case, add_prefix_space=False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token # 50256
        num_added_tokens = tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        model = model_class.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
        embedding_layer = model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id


    elif model_name_or_path == 'gpt2-aux':
        config_class, model_class, tokenizer_class = GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer
        model_config = config_class.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=n_labels, output_hidden_states=True)

        tokenizer = tokenizer_class.from_pretrained('gpt2', do_lower_case=do_lower_case, add_prefix_space=False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token # 50256
        num_added_tokens = tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        model = model_class.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
        embedding_layer = model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id

    elif model_name_or_path == 'roberta-base-aux':
        config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
        model_config = config_class.from_pretrained(pretrained_model_name_or_path='roberta-base', num_labels=n_labels, output_hidden_states=True)

        #(pad_token_id = 1, bos_token_id = 0, eos_token_id = 2)
        tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path='roberta-base', do_lower_case=do_lower_case)
        tokenizer.padding_side = "right"
        model = model_class.from_pretrained(pretrained_model_name_or_path='roberta-base', config=model_config)
        

    # update tokenizer
    for word in descriptors:
        if len(tokenizer.tokenize(word)) > 1:
            tokenizer.add_tokens([word])
    model.resize_token_embeddings(len(tokenizer)) 
    return model, tokenizer, descriptors, word_label


def load_pretrained_model_cls(load_aux_dir, cls_type,       
                            model_aux_name,                        
                            epoch, block_size, do_lower_case = True, n_labels = 17):
    """
    load_aux_dir: path to the directory of the auxiliary model
    """
    # load all mask tokens
    descriptors_path = "../data/mask_token/mask_all.json"
    word_label_path = "../data/mask_token/word_label.json"

    descriptors = load_json(descriptors_path) # [daughter: female]
    descriptors = [desc.lower() for desc in descriptors]

    word_label = load_json(word_label_path) # {'daughter': female}
    word_label = {key.lower(): val.lower() for key,val in word_label.items()}

    load_aux_path = os.path.join(load_aux_dir)
    print("Load AUX PATH: ", load_aux_path)

    if not os.path.exists(load_aux_path):
        print("The Classification model path does not exist!")
    
    MODEL_CLASSES = {"gpt2-aux": (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
                    "bert-aux": (BertConfig, BertForSequenceClassification, BertTokenizer),
                    "roberta-base-aux": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
    ########################################################################################################
    #############                           STEP 2:  load NET2 Model                           #############
    ########################################################################################################
    
    
    if cls_type == 'gpt2-aux': 
        # Select model class
        config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2-aux"]
        # use the model class to load the model architecture
        model = model_class.from_pretrained(load_aux_path)
        tokenizer = tokenizer_class.from_pretrained(load_aux_path, do_lower_case=do_lower_case, num_labels=n_labels)
    
    elif cls_type == 'roberta-base-aux': 
        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta-base-aux"]
        
        # use the model class to load the model architecture
        model = model_class.from_pretrained(load_aux_path)
        tokenizer = tokenizer_class.from_pretrained(load_aux_path, do_lower_case=do_lower_case, num_labels=n_labels)
        
    
    ########################################################################################################
    #############                           STEP 2:  load NET2 Model                           #############
    ########################################################################################################
    #model_net2 = net2
    #net2_path = load_net2_path +'/' + model_net2_name
    #checkpoint = torch.load(net2_path)
    
    # load model achitecture
    #model_net2= (model_net2.module if hasattr(model_net2, "module") else model_net2)  # Take care of distributed/parallel training
    #model_net2.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer, descriptors, word_label




def load_data_gpt2(train_data_path, val_data_path, sub_sample_size, tokenizer, labels_ids, \
            cls_id_2_desc_id, n_gpu, block_size, remove_zero, train_batch_size, val_batch_size):    
    """
    load data for gpt2 discriminator
    """
    # create objective function
    gpt2_classificaiton_collator_main = Gpt2ClassificationCollatorMain(use_tokenizer=tokenizer, labels_encoder=labels_ids, 
                                                                    cls_id_2_desc_id = cls_id_2_desc_id, max_sequence_len=block_size)
    
    train_aim = 'train_data'
    val_aim = 'val_data'

    print('Training Data Label Distribution...')
    inputs_train = gpt2_classificaiton_collator_main(train_data_path, train_aim, sub_sample_size, remove_zero)
    print('Validation Data Label Distribution...')
    inputs_val = gpt2_classificaiton_collator_main(val_data_path, val_aim, sub_sample_size, remove_zero)

    # train, validation dataset split
    #train, val = random_split_train_val_main(inputs, train_ratio=0.9)

    print('Dealing with Train...')
    # Create pytorch dataset.
    train_data = OWTCDatasetMain(inputs_train)

    print('Created `train_dataset` with %d examples!'%len(train_data))
    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers = 4 * n_gpu)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    print('Dealing with Validation...')
    # Create pytorch dataset.
    valid_data =  OWTCDatasetMain(inputs_val)
    print('Created `valid_dataset` with %d examples!'%len(valid_data))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_data, batch_size=val_batch_size, shuffle=False, num_workers = 4 * n_gpu)
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))        

    return train_dataloader, valid_dataloader


def load_data_bert(train_data_path, val_data_path, sub_sample_size, \
                use_tokenizer_gpt2, use_tokenizer_bert, \
                method_name_gpt2, method_name_bert,\
                labels_word, descriptor, n_gpu, block_size, remove_zero,\
                train_batch_size, val_batch_size):    
    """
    load data for bert discriminator
    """
    # create objective function

    data_collator = DataCollator(use_tokenizer_gpt2, use_tokenizer_bert, \
                                method_name_gpt2, method_name_bert, \
                                labels_word, descriptor, max_sequence_len=block_size)
    
    train_aim = 'train_data'
    val_aim = 'val_data'
    
    print('Dealing with Train...')
    # Create pytorch dataset.
    inputs_gpt2_train, inputs_bert_train = data_collator(train_data_path, train_aim, sub_sample_size, remove_zero)
    print('Dealing with Val...')
    # Create pytorch dataset.
    inputs_gpt2_val, inputs_bert_val = data_collator(val_data_path, val_aim, sub_sample_size, remove_zero)
    
    # train, validation dataset split
    #train, val = random_split_train_val_main(inputs, train_ratio=0.9)


    train_data = GAN_Dataset(inputs_gpt2_train, inputs_bert_train)
    print('Created `train_dataset` with %d examples!'%len(train_data))
    
    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers = 4 * n_gpu)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    print('Dealing with Validation...')
    # Create pytorch dataset.
    valid_data =  GAN_Dataset(inputs_gpt2_val, inputs_bert_val)
    print('Created `valid_dataset` with %d examples!'%len(valid_data))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_data, batch_size=val_batch_size, shuffle=False, num_workers = 4 * n_gpu)
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))        

    return train_dataloader, valid_dataloader

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default="../output/",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_train_epochs", default=4, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the val set.")
    parser.add_argument(
        "--opt_method", default='method_single_opt', type=str, help="Optimization method with or without updating net1 and net2."
    )
    parser.add_argument("--cls_selection_method", default='[CLS]_TOKEN', type=str, help="The CLS model embedding extraction method")
    parser.add_argument("--model_type", default='base', type=str, help="GPT2 Model type (base/large).")
    parser.add_argument(
        "--model_name_or_path_main",
        default="gpt2-main",
        type=str,
        help="The model main checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--model_name_or_path_aux",
        default="roberta-base-aux",
        type=str,
        help="The model aux checkpoint for weights initialization, or gpt2-aux",
    )
    parser.add_argument(
        "--block_size",
        default=256,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--n_labels", default=17, type=int, help="Number of categories for fairness.")
    parser.add_argument("--embedding_size", default=768, type=int, help="Embedding size.")
    parser.add_argument("--remove_zero", default=False, type=bool, help="Whether to remove zero lable from the dataset.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--alpha", default=0.1, type=float, help="Weight for aux loss.")
    parser.add_argument("--sub_sample_size", default=1000, type=int, help="Training sample size")
    parser.add_argument("--load_pretrained_cls", default='pretrain', type=str, help="Load pretrained model roberta/gpt2")
    #parser.add_argument("--local_rank", default=-1)

    # other parameters
    parser.add_argument("--seed", type=int, default=90025, help="random seed for initialization")
    parser.add_argument("--switch_freq", type=int, default=20, help="switch frequency for adversarial training.")
    parser.add_argument("--switch_freq_dis", type=int, default=80, help="frequency for adversarial training for dis.")
    parser.add_argument("--switch_freq_gen", type=int, default=20, help="frequency for adversarial training for gen.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    # Setting Seeds
    seed = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    set_seeding(seed, n_gpu)    


    # Setting CPU Accelarate
    cpu_num = cpu_count()
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    # Langauage Model Data Path
    train_data_path = args.train_data_file
    val_data_path = args.eval_data_file

    sub_sample_size = args.sub_sample_size

    # Global Training Parameters
    if train_data_path == 'stas/openwebtext-10k':
        sample_size = '10K-' + str(sub_sample_size)
    elif train_data_path == 'openwebtext':
        sample_size = '8M-' + str(sub_sample_size)

    epochs = args.num_train_epochs
    train_batch_size = args.per_gpu_train_batch_size
    val_batch_size = args.per_gpu_eval_batch_size
    do_train = args.do_train
    do_eval = args.do_eval
    opt_method = args.opt_method  # method_single_opt

    # Main and Axuiliary Network Name Selection and Model Parameters setting
    model_name_or_path_main = args.model_name_or_path_main
    model_name_or_path_aux = args.model_name_or_path_aux
    method_name_gpt2 = model_name_or_path_main
    method_name_bert = model_name_or_path_aux
    model_type_gen = args.model_type
    joint_model_type = "%s-%s" % (model_name_or_path_main, model_name_or_path_aux)
    cls_selection_method = args.cls_selection_method #'[CLS]_TOKEN'
    
    load_pretrained_cls = args.load_pretrained_cls
    if load_pretrained_cls == 'pretrain':
        pretrained = 'pretrained'
    elif load_pretrained_cls == 'init':
        pretrained = 'intialized'
    print(pretrained)

    if model_name_or_path_aux == 'gpt2-aux':
        classifier_type = 'gpt2-base'
    elif model_name_or_path_aux == 'roberta-base-aux':
        classifier_type = 'roberta-base'


    block_size = args.block_size
    n_labels = args.n_labels
    embedding_size = args.embedding_size
    do_lower_case = args.do_lower_case
    remove_zero = args.remove_zero # Fairness Data Setting

    # Adversarial Training Setting 
    switch_freq = args.switch_freq
    alpha = args.alpha

    # Model Save Path (4 models): 
    # 2 MAIN MODELS: model_main, model_aux
    # 2 Layers: net1, net2
    output_dir = args.output_dir
    output_main_dir = os.path.join(output_dir, 'main/gpt2-%s' % (model_type_gen), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 
    output_net1_dir = os.path.join(output_dir, 'main/gpt2-%s' % (model_type_gen), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 

    if model_name_or_path_aux == 'gpt2-aux':
        output_aux_dir = os.path.join(output_dir, 'aux/%s-%s' % (model_name_or_path_aux, model_type_gen), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 
        output_net2_dir = os.path.join(output_dir, 'aux/%s-%s' % (model_name_or_path_aux, model_type_gen), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 
        # Evaluation Outputs Dir
        eval_output_dir = os.path.join(output_dir, 'main/%s' % (joint_model_type), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 
        pretrained_aux_dir = os.path.join(output_dir, 'pretrained/cls/gpt2-small/block-%d/checkpoint-%d/model_gpt2/' % (block_size, epochs-1))


    elif model_name_or_path_aux == 'roberta-base-aux':
        output_aux_dir = os.path.join(output_dir, 'aux/%s-%s' % (model_name_or_path_aux, model_type_gen), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 
        output_net2_dir = os.path.join(output_dir, 'aux/%s-%s' % (model_name_or_path_aux, model_type_gen), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 
        # Evaluation Outputs Dir
        eval_output_dir = os.path.join(output_dir, 'main/%s' % (joint_model_type), sample_size, pretrained, classifier_type, opt_method, str(block_size), str(train_batch_size), 'alpha_%s' % (alpha)) 
        pretrained_aux_dir = os.path.join(output_dir, 'pretrained/cls/roberta-base-small/block-%d/checkpoint-%d/model_roberta-base/' % (block_size, epochs-1))


    # Load 4 Models 
    if load_pretrained_cls == 'init':
        print(load_pretrained_cls)
        net1 = NetLM(embedding_size, embedding_size)
        net2 = NetAdv(embedding_size, embedding_size)
        model_main, tokenizer_main, descriptors, word_label = load_pretrained_model_init(model_name_or_path_main, block_size, args.do_lower_case, n_labels)
        model_aux, tokenizer_aux, _, _ = load_pretrained_model_init(model_name_or_path_aux, block_size, args.do_lower_case, n_labels)
    elif load_pretrained_cls == 'pretrain':
        print(load_pretrained_cls)
        net1 = NetLM(embedding_size, embedding_size)
        net2 = NetAdv(embedding_size, embedding_size)
        model_main, tokenizer_main, descriptors, word_label = load_pretrained_model_init(model_name_or_path_main, block_size, args.do_lower_case, n_labels)
        model_aux, tokenizer_aux, _, _ = load_pretrained_model_cls(pretrained_aux_dir, model_name_or_path_aux, 'aux', epochs-1, block_size, do_lower_case, n_labels)
        

    # Load Model to Multiple GPU
    if n_gpu > 1:
        model_main = torch.nn.DataParallel(model_main, device_ids=list(range(n_gpu)))
        model_aux = torch.nn.DataParallel(model_aux, device_ids=list(range(n_gpu)))
        net1 = torch.nn.DataParallel(net1, device_ids=list(range(n_gpu)))
        net2 = torch.nn.DataParallel(net2, device_ids=list(range(n_gpu)))  
    model_main.to(device)
    model_aux.to(device)
    net1.to(device)
    net2.to(device)




    # Create Data Labels for language model and classification model
    if joint_model_type == 'gpt2-main-gpt2-aux':
        desc_id_2_cls_id, cls_id_2_desc_id, label_class, class2kingdom = label_id_mapping_gpt2(tokenizer_main, word_label, descriptors)
        train_dataloader, valid_dataloader = load_data_gpt2(train_data_path, val_data_path, sub_sample_size, tokenizer_main, desc_id_2_cls_id, cls_id_2_desc_id, n_gpu, block_size, remove_zero, train_batch_size, val_batch_size)
    elif joint_model_type == 'gpt2-main-roberta-base-aux':
        labels_ids, label_class, class2kingdom, labels_word = label_id_mapping_bert(tokenizer_aux, word_label, descriptors, 'roberta-base-aux')
        train_dataloader, valid_dataloader = load_data_bert(train_data_path, val_data_path, sub_sample_size, \
                                                            tokenizer_main, tokenizer_aux, \
                                                            method_name_gpt2, method_name_bert, \
                                                            labels_word, descriptors, n_gpu, block_size, remove_zero, \
                                                            train_batch_size, val_batch_size)


    # Set Hyperparameters
    optimizer_main, total_steps_main, scheduler_main = hyper_parameters(model_main, train_dataloader, epochs, lr = 2e-5, eps=1e-8)
    optimizer_aux, total_steps_aux, scheduler_aux = hyper_parameters(model_aux, train_dataloader, epochs, lr = 2e-5, eps=1e-8)
    optimizer_net, total_steps_net, scheduler_net = hyper_parameters_net(net1, net2, train_dataloader, epochs, lr = 2e-5, eps=1e-8)

    # Store the average loss
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    # Training
    for epoch in tqdm(range(epochs)):
        logger.info('Epoch %d, Train on Batches', epoch)
        # Perform one full pass over the training set.
        if joint_model_type == 'gpt2-main-gpt2-aux':
            if do_train:
                avg_cls_epoch_loss, avg_adv_epoch_loss = model_train_gpt2(train_dataloader, model_main, model_aux, net1, net2,
                                                                    tokenizer_main, tokenizer_aux,
                                                                        optimizer_main, scheduler_main, optimizer_aux, scheduler_aux, optimizer_net, scheduler_net, 
                                                                        switch_freq, alpha, block_size, n_gpu, 
                                                                        device, class2kingdom, opt_method,
                                                                        output_main_dir, output_aux_dir, output_net1_dir, output_net2_dir,
                                                                        epoch, args)
            if do_eval:
                # Get prediction form model on validation data. 
                logger.info('Epoch %d, Validation on batches', epoch)

                # load model from training then do evaluate.
                # load language model
                model_main_val = load_checkpoint_model_gpt2('main', output_main_dir, epoch, alpha, args.do_lower_case, device)
                model_aux_val = load_checkpoint_model_gpt2('aux', output_aux_dir, epoch, alpha, args.do_lower_case, device)
                # load net1 and net 2
                net1_val = load_checkpoint_model_gpt2('net1', output_net1_dir, epoch, alpha, args.do_lower_case, device, net1)
                net2_val = load_checkpoint_model_gpt2('net2', output_net2_dir, epoch, alpha, args.do_lower_case, device, net2)

                result = model_validation_gpt2(model_main_val[0], model_aux_val[0], net1_val, net2_val, 
                                        valid_dataloader, eval_output_dir, block_size, alpha,
                                        opt_method, val_batch_size, 
                                        device, epoch)


        elif joint_model_type == 'gpt2-main-roberta-base-aux':
            if do_train:
                avg_cls_epoch_loss, avg_adv_epoch_loss = model_train_bert(train_dataloader, model_main, model_aux, net1, net2,
                                                                    tokenizer_main, tokenizer_aux,
                                                                        optimizer_main, scheduler_main, optimizer_aux, scheduler_aux, optimizer_net, scheduler_net, 
                                                                        switch_freq, alpha, block_size, n_gpu, 
                                                                        device, class2kingdom, opt_method, cls_selection_method,
                                                                        output_main_dir, output_aux_dir, output_net1_dir, output_net2_dir,
                                                                        epoch, args)
            if do_eval:
                # Get prediction form model on validation data. 
                logger.info('Epoch %d, Validation on batches', epoch)

                # load model from training then do evaluate.
                # load language model
                model_main_val = load_checkpoint_model_bert('main', output_main_dir, epoch, alpha, args.do_lower_case, device)
                model_aux_val = load_checkpoint_model_bert('aux', output_aux_dir, epoch, alpha, args.do_lower_case, device)
                # load net1 and net 2
                net1_val = load_checkpoint_model_bert('net1', output_net1_dir, epoch, alpha, args.do_lower_case, device, net1)
                net2_val = load_checkpoint_model_bert('net2', output_net2_dir, epoch, alpha, args.do_lower_case, device, net2)

                result = model_validation_bert(model_main_val[0], model_aux_val[0], net1_val, net2_val, 
                                        valid_dataloader, eval_output_dir, block_size, alpha,
                                        opt_method, cls_selection_method, val_batch_size, 
                                        device, epoch)        
            
        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(avg_adv_epoch_loss)
        all_loss['val_loss'].append(result['adv_loss'])
        #all_acc_loss['train_lm_loss'].append(train_acc)
        #all_acc_loss['val_lm_loss'].append(val_acc)

    # Plot loss curves.
    # plot_dict(all_loss, use_xlabel='Epochs', magnify = 0.1, use_ylabel='Value', use_linestyles=['-', '--'])

if __name__ == "__main__":
    main()
    
