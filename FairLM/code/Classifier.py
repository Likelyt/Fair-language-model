#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import json
import numpy as np
import argparse
import random
import scipy.stats
import logging
import os
import warnings
import re
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import datasets
import torch
import pandas as pd
from tqdm import tqdm
from datasets import list_datasets, load_dataset, list_metrics, load_metric
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split
from utils.util import write_json, load_json, set_seeding, classification_loss
from utils.data_processing import random_split_train_val_cls, CLSDataset, RoBERTaClassificationCollator

from ml_things import plot_dict
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    set_seed,
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
)

logger = logging.getLogger(__name__)


'''
# write the mask_token into json
mask_gender_list = load_json("../data/mask_token/gender.json")
mask_race_list = load_json("../data/mask_token/race.json")
mask_age_list = load_json("../data/mask_token/age.json")
print("There are %d categories in total!" % (len(mask_age_list) + len(mask_gender_list) + len(mask_race_list)))

# create a (fair-words, class-label)
temp_dict = {**mask_gender_list, **mask_race_list, **mask_age_list}
word_label_dict = {}
for key, items in temp_dict.items():
    for item in items:
        word_label_dict[item] = key
write_json("../data/mask_token/word_label.json", word_label_dict)
word_label = load_json("../data/mask_token/word_label.json")
        
mask_all = []
for key, values in mask_gender_list.items():
    #print(len(mask_gender_list[key]))
    mask_all.extend(mask_gender_list[key])
    
for key, values in mask_race_list.items():
    #print(len(mask_race_list[key]))
    mask_all.extend(mask_race_list[key])

for key, values in mask_age_list.items():
    #print(len(mask_age_list[key]))
    mask_all.extend(mask_age_list[key])

#write_json("../data/mask_token/mask_all.json", mask_all)
mask_all = load_json("../data/mask_token/mask_all.json")

# create label
mask_gender_list = load_json("../data/mask_token/gender.json")
mask_race_list = load_json("../data/mask_token/race.json")
mask_age_list = load_json("../data/mask_token/age.json")
# step 1: create y label
keys = list(mask_gender_list.keys()) + list(mask_race_list.keys()) + list(mask_age_list.keys())
write_json('../data/mask_token/categories.json', keys)
'''

def load_model(model_name_or_path, block_size, do_lower_case = True, n_labels = 17):
    """
    Load model and tokenizer
    """
    # load all mask tokens
    descriptors_path = "../data/mask_token/mask_all.json"
    word_label_path = "../data/mask_token/word_label.json"

    descriptors = load_json(descriptors_path) # [daughter: female]
    descriptors = [desc.lower() for desc in descriptors]

    word_label = load_json(word_label_path) # {'daughter': female}
    word_label = {key.lower(): val.lower() for key,val in word_label.items()}

    # Step 2.1: Select model class
    MODEL_CLASSES = {"gpt2": (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
                     "roberta-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_name_or_path]
    if model_name_or_path == 'gpt2':
        model_config = config_class.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels, output_hidden_states=True)
    elif model_name_or_path == 'roberta-base':
        model_config = config_class.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels, output_hidden_states=True)
    

    # step 2.2 Tokenizer setup
    #tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
    
    if model_name_or_path == 'gpt2':
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case, add_prefix_space=False)
        tokenizer.padding_side = "left"        
        tokenizer.pad_token = tokenizer.eos_token # 50256
        num_added_tokens = tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
        embedding_layer = model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id
        
    elif model_name_or_path == 'roberta-base':
        #(pad_token_id = 1, bos_token_id = 0, eos_token_id = 2)
        tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path=model_name_or_path, do_lower_case=do_lower_case)
        tokenizer.padding_side = "right"
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

        
    if model_name_or_path == 'gpt2':
        model.config.pad_token_id = model.config.eos_token_id
        
    # update tokenizer
    for word in descriptors:
        if len(tokenizer.tokenize(word)) > 1:
            tokenizer.add_tokens([word])
    model.resize_token_embeddings(len(tokenizer)) # GPT2: 50378, RoBERTa: 50265+
    return model, tokenizer, descriptors, word_label


def label_id_mapping(user_tokenizer, word_label, descriptors, tokenizer_method):
    """
    Create label id mapping
    """
    categories_path = '../data/mask_token/categories.json'
    categories = load_json(categories_path)
    kindom2class = {'gender':[1, 2], 
              'race': [3, 4, 5, 7, 8, 9, 10,11],
              'age': [12, 13, 14, 15, 16]}
    class2kingdom = {}
    for k, v in kindom2class.items():
        for it in v:
            class2kingdom[str(it)] = k

    # 'female': class-1
    label_class = {key: i+1 for i, key in enumerate(categories)} 
    # label_id_map: {dict_word_id: label_class}
    labels_ids = {} 
    labels_word = {}
    for word in descriptors:
        # female = daughter
        class_name = word_label[word]
        # 1 = female class
        label_id = label_class[class_name]
        
        labels_word[word.lower()] = label_id
        # 24724 (word dict_id) of female
        dict_id = user_tokenizer.convert_tokens_to_ids(user_tokenizer.tokenize(word))[0]
        # 24724: 1
        labels_ids[dict_id] = label_id
    
        if tokenizer_method == 'roberta-base':
        # add space for each tokens
            space_dict_id = user_tokenizer.convert_tokens_to_ids(user_tokenizer.tokenize(' '+word))[0]
            labels_ids[space_dict_id] = label_id        
        
    #write_json("../data/mask_token/label_id_map.json", label_id_map)
    # not sure whether adjust the embedding will change the id or not
    #label_id_map = load_json("../data/mask_token/label_id_map.json")
    return labels_ids, label_class, class2kingdom, labels_word



def model_train(dataloader, model, user_tokenizer, model_type, optimizer, scheduler, n_gpu, device, class2kingdom, output_dir, epoch, block_size):
    """
    Train pytorch model on a single pass through the data loader.
    It will use the global variable `model` which is the transformer model 
    loaded on `_device` that we want to train on.
    This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.
    Arguments:
      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.
      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.
      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.
    Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss].
    Star:
    model.state_dict().keys() # all model keys
    model.score.weight.size() # the last layer 17 * 768 network.
    """

    # Use global variable for model.
    #global model

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):

        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device) for k,v in batch.items()}

        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        if model_type == 'roberta-base':
            embedding_output_raw = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], \
                                         labels = batch['labels'])
            embedding_output_transformed = torch.zeros((batch['input_ids'].size(0), embedding_output_raw.hidden_states[-1][0].size(1))).to(device)
            for i in range(len(batch['masked_pos'])):
                # check whether it has the masked id
                if torch.sum(batch['masked_pos'][i]) == 0:
                    embedding_output_transformed[i] = embedding_output_raw.hidden_states[-1][i][0]
                else:
                    embedding_sum = torch.matmul(batch['masked_pos'][i].unsqueeze(0).type(torch.float), \
                                                                   embedding_output_raw.hidden_states[-1][i])
                    embedding_output_transformed[i] = embedding_sum/torch.sum(batch['masked_pos'][i])
            logits = model.classifier(embedding_output_transformed.unsqueeze(1))
            loss = classification_loss(logits, labels = batch['labels'])

        elif model_type == 'gpt2':
            outputs =  model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], \
                                         labels = batch['labels'])
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple along with the logits. We will use logits
            # later to calculate training accuracy.
            loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)
    
    # save model
    print("Pos 2; Num %d" % model.transformer.wte.num_embeddings)
    save_model(model, output_dir, epoch, model_type, user_tokenizer)
    print("Pos 3; Num %d" % model.transformer.wte.num_embeddings)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss



def validation(dataloader, model, model_type, n_gpu, device):
    """Validation function to evaluate model performance on a 
    separate set of data.

    This function will return the true and predicted labels so we can use later
    to evaluate the model's performance.

    This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.
    """

    # Use global variable for model.
    # global model

    # Tracking variables
    predictions_labels = []
    true_labels = []
    #total loss for this epoch.
    total_loss = 0
    
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()
        # move batch to device
        batch = {k: v.type(torch.long).to(device) for k,v in batch.items()}
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            if model_type == 'roberta-base':
                embedding_output_raw = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], \
                                             labels = batch['labels'])
                embedding_output_transformed = torch.zeros((batch['input_ids'].size(0), embedding_output_raw.hidden_states[-1][0].size(1))).to(device)
                for i in range(len(batch['masked_pos'])):
                    # check whether it has the masked id
                    if torch.sum(batch['masked_pos'][i]) == 0:
                        embedding_output_transformed[i] = embedding_output_raw.hidden_states[-1][i][0]
                    else:
                        embedding_sum = torch.matmul(batch['masked_pos'][i].unsqueeze(0).type(torch.float), \
                                                                       embedding_output_raw.hidden_states[-1][i])
                        embedding_output_transformed[i] = embedding_sum/torch.sum(batch['masked_pos'][i])
                logits = model.classifier(embedding_output_transformed.unsqueeze(1))
                loss = classification_loss(logits, labels = batch['labels'])
            
            elif model_type == 'gpt2':
                outputs =  model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], \
                                labels = batch['labels'])
                # The call to `model` always returns a tuple, so we need to pull the 
                # loss value out of the tuple along with the logits. We will use logits
                # later to to calculate training accuracy.
                loss, logits = outputs[:2]
                
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                total_loss += loss.item()
            else:
                total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

        # Calculate the average loss over the training data.
        avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss



def load_data(model_name_or_path, tokenizer, data_path, sub_sample_size, \
             labels_ids, descriptors, block_size, \
             labels_word, remove_zero=True, train_batch_size = 2, val_batch_size=2):
    """
    Load data from a given path and return dataloaders for training and validation.
    """

    # load data
    dataset = load_dataset(data_path)    
    # Needs construction
    if model_name_or_path == 'gpt2':
        gpt2_classificaiton_collator = RoBERTaClassificationCollator(use_tokenizer=tokenizer, labels_word=labels_word, descriptor = descriptors, max_sequence_len=block_size)
        inputs = gpt2_classificaiton_collator(dataset, sub_sample_size, remove_zero)
    elif model_name_or_path == 'roberta-base':
        roberta_classificaiton_collator = RoBERTaClassificationCollator(use_tokenizer=tokenizer, labels_word=labels_word, descriptor = descriptors, max_sequence_len=block_size)
        inputs = roberta_classificaiton_collator(dataset, sub_sample_size, remove_zero)

    # train, validation dataset split
    train, val = random_split_train_val_cls(inputs, train_ratio=0.9)

    logger.info('Dealing with Train...')
    # Create pytorch dataset.
    train_data = CLSDataset(train)

    logger.info('Created `train_dataset` with %d examples!'%len(train_data))
    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    logger.info('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    logger.info('Dealing with Validation...')
    # Create pytorch dataset.
    valid_data =  CLSDataset(val)
    logger.info('Created `valid_dataset` with %d examples!'%len(valid_data))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_data, batch_size=val_batch_size, shuffle=False)
    logger.info('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
    
    return train_dataloader, valid_dataloader



def hyper_parameters(model, train_dataloader, epochs, lr = 2e-5, eps=1e-8):
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # default is 1e-8.
                      )
    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    return optimizer, total_steps, scheduler



def save_model(model, output_dir, epoch, model_type, tokenizer=None):
    """
    Save model to a given path.
    """
    output_path = os.path.join(output_dir, 'checkpoint-%s' % str(epoch), 'model_%s' % model_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    #torch.save({'model_state_dict': model_to_save.state_dict()}, model_path)
    model_to_save.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info('Save model to %s' % (output_path))

def load_model_para(model, output_dir, epoch):
    load_path = output_dir+'/checkpoint-' + str(epoch) + '/cls_model'
    checkpoint = torch.load(load_path)
    model = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    
    model_trained = model
    model_trained.load_state_dict(checkpoint['model_state_dict'])

    return model_trained

def inference(passages, mlp_para, desc_id_2_cls_id):
    """
    Inference on a given passage.
    """
    mask_token_ids = list(desc_id_2_cls_id.keys())
    batch_size, seq_length = passages['input_ids'].size()   
    for i in range(batch_size):
        for j in range(seq_length):
            if passages['attention_mask'][i][j].item() == 1 and passages['input_ids'][i][j].item() in mask_token_ids:
                passages['attention_mask'][i][j] -=1 
    return passages


def main():
    """
    Main function.
    """
    ###################################################################################################
    ###########                       Step 0. Hyperparameter Setting                        ###########
    ###################################################################################################

    parser = argparse.ArgumentParser()
    # parameters for data
    parser.add_argument('--data_path', type=str, default='openwebtext', help='data path')
    parser.add_argument('--sub_sample_size', type=int, default=1000000, help='sub sample size')
    parser.add_argument('--remove_zero', type=bool, default=False, help='remove zero class?')
    parser.add_argument('--n_labels', type=int, default=17, help='number of labels')
    
    # parameters for model
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='model name')
    parser.add_argument('--block_size', type=int, default=256, help='block size')
    parser.add_argument('--gpu_id', type=str, default=1, help='The id of the gpu to use')

    # parameters for training
    parser.add_argument('--epochs', type=int, default=4, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--val_batch_size', type=int, default=64, help='validation batch size')

    # other parameters
    parser.add_argument("--seed", type=int, default=90025, help="random seed for initialization")

    
    args = parser.parse_args()
    # Fix the seed for reproducibility
    seed = args.seed
    n_gpu = 1
    set_seeding(seed, n_gpu)   

    # data
    data_path = args.data_path
    sub_sample_size = args.sub_sample_size
    remove_zero = args.remove_zero
    n_labels = args.n_labels

    # model parameter
    model_name_or_path = args.model_name_or_path
    block_size = args.block_size
    do_lower_case = True

    gpu_id = args.gpu_id
    if torch.cuda.is_available():
        device = torch.device('cuda:%s' %(gpu_id))
    else:
        device = torch.device('cpu')

    # training parameters
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    

    # save model path
    output_dir = '../output/pretrained/cls/%s-small/block-%s' % (model_name_or_path, str(block_size))

    # load model, descriptors, word labels
    model, tokenizer, descriptors, word_label = load_model(model_name_or_path, block_size, do_lower_case, n_labels)
    print("Pos 1; Num %d" % model.transformer.wte.num_embeddings)

    # step 2.4 Load model to gpu
    model.to(device)
    
    # descriptor id to class id
    desc_id_2_cls_id, label_class, class2kingdom, labels_word = label_id_mapping(tokenizer, word_label, descriptors, tokenizer_method = model_name_or_path)
    
    # load data
    train_dataloader, valid_dataloader = load_data(model_name_or_path, tokenizer, data_path, sub_sample_size, desc_id_2_cls_id, descriptors, block_size, labels_word, remove_zero, train_batch_size, val_batch_size)

    # set hyper-parameters
    optimizer, total_steps, scheduler = hyper_parameters(model, train_dataloader, epochs, lr = 2e-5, eps=1e-8)
    
    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}
    
    
    # Training
    for epoch in tqdm(range(epochs)):
        logger.info('Epoch %d, Train on Batches' % (epoch))
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = model_train(train_dataloader, model, tokenizer, model_name_or_path, optimizer, scheduler, n_gpu, device, class2kingdom, output_dir, epoch, block_size)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data. 
        logger.info('Epoch %d, Validation on batches' % (epoch))
        valid_labels, valid_predict, val_loss = validation(valid_dataloader, model, model_name_or_path, n_gpu, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # Print loss and accuracy values to see how training evolves.
        logger.info(" train_loss: %.4f - val_loss: %.4f - train_acc: %.4f - valid_acc: %.4f" % (train_loss, val_loss, train_acc, val_acc))
        print(" train_loss: %.4f - val_loss: %.4f - train_acc: %.4f - valid_acc: %.4f" % (train_loss, val_loss, train_acc, val_acc))
        
        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)
    
    # Plot loss curves.
    plot_dict(all_loss, use_xlabel='Epochs', magnify = 0.1, use_ylabel='Value', use_linestyles=['-', '--'])
    # Plot accuracy curves.
    plot_dict(all_acc, use_xlabel='Epochs', magnify = 0.1, use_ylabel='Value', use_linestyles=['-', '--'])
    
    # Generate Report
    if remove_zero !=True:
        label_class['NA'] = 0
    true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader, model, model_name_or_path, n_gpu, device)
    # Create the evaluation report.
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(label_class.values()), target_names=list(label_class.keys()))
    # Show the evaluation report.
    print(evaluation_report)


if __name__ == "__main__":
    main()

