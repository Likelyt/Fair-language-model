import json
import re
import numpy as np
import random 
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split
from datasets import load_dataset
from utils.util import load_json

logger = logging.getLogger(__name__)

def random_split_train_val(inputs, train_ratio=0.9):
    """
    training data and validataion data split
    """
    n_samples = len(inputs['input_ids'])
    val_ratio = 1 - train_ratio
    train_n = int(0.9 * n_samples)
    val_n = n_samples - train_n

    train_list = random.sample(range(n_samples), train_n)
    val_list = [x for x in range(n_samples) if x not in train_list]

    train_data = {}
    train_data['input_ids'] = inputs['input_ids'][train_list,]
    train_data['attention_mask'] = inputs['attention_mask'][train_list,]
    train_data['labels'] = inputs['labels'][train_list]

    val_data = {}
    val_data['input_ids'] = inputs['input_ids'][val_list,]
    val_data['attention_mask'] = inputs['attention_mask'][val_list,]
    val_data['labels'] = inputs['labels'][val_list]
    
    return train_data, val_data    


def random_split_train_val_cls(inputs, block_size = 256, train_ratio=0.9):
    """
    training data and validataion data split for classification model
    """
    n_samples = len(inputs['input_ids'])
    val_ratio = 1 - train_ratio
    train_n = int(0.9 * n_samples)
    val_n = n_samples - train_n

    train_list = random.sample(range(n_samples), train_n)
    val_list = [x for x in range(n_samples) if x not in train_list]

    train_data = {}
    train_data['input_ids'] = inputs['input_ids'][train_list,]
    train_data['attention_mask'] = inputs['attention_mask'][train_list,]
    train_data['labels'] = inputs['labels'][train_list]
    # select the random samples
    temp_mask_pos = [inputs['masked_pos'][i] for i in train_list]
    # create a matrix for mask ids
    train_data['masked_pos'] = torch.zeros((len(temp_mask_pos), block_size))
    # transform the masked_pos to matrix
    for i in range(len(temp_mask_pos)):
        if temp_mask_pos[i] == []:
            continue
        else:
            for j in temp_mask_pos[i]: # list of mask ids for one sentence
                train_data['masked_pos'][i][j] = 1

    val_data = {}
    val_data['input_ids'] = inputs['input_ids'][val_list,]
    val_data['attention_mask'] = inputs['attention_mask'][val_list,]
    val_data['labels'] = inputs['labels'][val_list]
    temp_mask_pos = [inputs['masked_pos'][i] for i in val_list]
    # create a matrix for mask ids
    val_data['masked_pos'] = torch.zeros((len(temp_mask_pos), block_size))
    # transform the masked_pos to matrix
    for i in range(len(temp_mask_pos)):
        if temp_mask_pos[i] == []:
            continue
        else:
            for j in temp_mask_pos[i]: # list of mask ids for one sentence
                val_data['masked_pos'][i][j] = 1
    return train_data, val_data



class CLSDataset(Dataset):
    """
    PyTorch Dataset class for loading data.

    This is where the data parsing happens.

    This class is built with reusability in mind: it can be used as is as.

    Arguments:

        path (:obj:`str`):
            Path to the data partition.

    """

    def __init__(self, Dataset):
        #super(Dataset, self).__init__()

        # Number of exmaples.
        self.texts = Dataset['input_ids']
        self.att_mask = Dataset['attention_mask']
        self.labels = Dataset['labels']
        self.masked_pos = Dataset['masked_pos']

        self.n_examples = len(self.labels)
        return

    def __len__(self):
        """When used `len` return the number of examples.
        """
        return self.n_examples

    def __getitem__(self, item):
        """Given an index return an example from the position.
        Arguments:
        item (:obj:`int`):
            Index position to pick an example to return.
        Returns:
        :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
        asociated labels.

        """
        return {'input_ids':self.texts[item],
                'attention_mask':self.att_mask[item],
                'labels':self.labels[item],
                'masked_pos':self.masked_pos[item]}

class Gpt2ClassificationCollator(object):
    """
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder
        
        self.max_sequence_len = max_sequence_len

    def __call__(self, dataset, sub_sample_size, remove_zero=True):
        """
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
        # step 1.3: convert this mask tokens into ids
        # Get all texts from sequences list.
        #texts = [sequence['text'] for sequence in dataset['train']]
        #inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        inputs = self.gen_token_ids(dataset, sub_sample_size) # 10K*1024
        labels, keep_rep_i = self.gen_label(inputs['input_ids'])

        # replicate inputs
        inputs_rep = self.rep_data(dataset, keep_rep_i, sub_sample_size)
        # update x and mask
        inputs = self.update_data_and_att(inputs, inputs_rep)
        
        # add labels to data
        inputs.update({'labels':torch.tensor(labels)})

        #  remove 0 class
        if remove_zero == True:
            inputs = self.remove_zero_class(inputs)

        return inputs

    def remove_zero_class(self, inputs):
        inputs_new = {}

        # index of nonzero class
        nonzero_class_list = []
        for i, label in enumerate(inputs['labels']):
            if label!= 0:
                nonzero_class_list.append(i)

        # remove zero class from input_ids
        inputs_new['input_ids'] = inputs['input_ids'][nonzero_class_list,]
        # remove zero class from attention_mask
        inputs_new['attention_mask'] = inputs['attention_mask'][nonzero_class_list,]
        # remove zero class from labels
        inputs_new['labels'] = inputs['labels'][nonzero_class_list]

        # count how many nonzero class left
        n_nonzero_class = len(nonzero_class_list)
        print("Number of non-zero ratio %.4f, origin total %d, nonzero total samples %d" % (n_nonzero_class/len(inputs['labels']), len(inputs['labels']), n_nonzero_class))

        return inputs_new
        
    def update_data_and_att(self, inputs, inputs_rep):
        input_c = {}
        input_c['input_ids'] = torch.cat([inputs['input_ids'], inputs_rep['input_ids']], dim=0)
        input_c['attention_mask'] = torch.cat([inputs['attention_mask'], inputs_rep['attention_mask']], dim=0)
        return input_c
        
    def gen_token_ids(self, dataset, sub_sample_size):
        # mask 0: is adding padding
        # mask 1: is adding token
        texts = []
        for sequence in tqdm(dataset['train']):
            texts.append(sequence['text'])
            if len(texts)>= sub_sample_size:
                break
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        return inputs
    
    def rep_data(self, dataset, keep_rep_i, sub_sample_size):
        texts = []
        for sequence in tqdm(dataset['train']):
            texts.append(sequence['text'])
            if len(texts)>= sub_sample_size:
                break
        #texts = [sequence['text'] for sequence in dataset['train']]
        rep_texts = [texts[index] for index in keep_rep_i]
        inputs_rep = self.use_tokenizer(text=rep_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        return inputs_rep
    
    # assign labels for these examples
    def gen_label(self, inputs):
        # examples: a batch of sentences
        # label_id_map: {dict_id_of_word: label}

        # labels for each sentence
        labels = [[] for _ in range(len(inputs))]
        # iterate over each sentence
        for i in range(len(inputs)):
            # iterate over each word
            for j in range(len(inputs[i])):
                # if word is in the
                word_id = inputs[i][j].item()
                if word_id in self.labels_encoder.keys():
                    # append fair-word id to that example.
                    labels[i].append(self.labels_encoder[word_id])
            # we need to group [2,2,2,2] to a set then a list                
            labels[i] = list(set(labels[i]))

        count_0, count_1, count_2, count_3, count_more = 0,0,0,0,0
        for i in range(len(labels)):           
            if len(labels[i]) == 0:
                count_0 += 1
                labels[i] = [0]
            elif len(labels[i]) == 1:
                count_1 += 1                
            elif len(labels[i]) == 2:
                count_2 += 1
            elif len(labels[i]) == 3:
                count_3 += 1
            else:
                count_more += 1
        
        keep_rep_i, keep_rep_y = [], []

        for i in range(len(labels)):
            if len(labels[i]) >= 2:
                for j in range(1, len(labels[i])):
                    keep_rep_i.append(i) 
                    keep_rep_y.append(labels[i][j])

        # calculate how percentage data have label
        print("Classification Task: 0L %.3f, 1L %.3f, 2L %.3f, 3L %.3f, >3L %.3f" % 
              (count_0/len(labels), count_1/len(labels), count_2/len(labels), count_3/len(labels), count_more/len(labels)))

        # rechange the label            
        label_flat = [z[0] for z in labels]
        label_flat.extend(keep_rep_y)
        labels = label_flat

        return labels, keep_rep_i


class OWTCDatasetMain(Dataset):
    """
    PyTorch Dataset class for loading data.

    This is where the data parsing happens.

    This class is built with reusability in mind: it can be used as is as.

    Arguments:

        path (:obj:`str`):
            Path to the data partition.
    """

    def __init__(self, Dataset):
        #super(Dataset, self).__init__()

        # Number of exmaples.
        self.texts = Dataset['input_ids']
        self.att_mask = Dataset['attention_mask']
        self.labels = Dataset['labels']
        self.fair_mask = Dataset['fair_mask']
        
        self.n_examples = len(self.labels)
        return

    def __len__(self):
        """When used `len` return the number of examples.
        """
        return self.n_examples

    def __getitem__(self, item):
        """Given an index return an example from the position.
        Arguments:
        item (:obj:`int`):
            Index position to pick an example to return.
        Returns:
        :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
        asociated labels.

        """
        return {'input_ids':self.texts[item],
                'attention_mask':self.att_mask[item],
                'labels':self.labels[item],
                'fair_mask': self.fair_mask[item]}


class GAN_Dataset(Dataset):
    """
    PyTorch Dataset class for loading data.

    This is where the data parsing happens.

    This class is built with reusability in mind: it can be used as is as.

    Arguments:

        path (:obj:`str`):
            Path to the data partition.

    """

    def __init__(self, Dataset_GEN, Dataset_CLS):
        #super(Dataset, self).__init__()

        # Number of exmaples.
        self.texts_GEN = Dataset_GEN['input_ids']
        self.att_mask_GEN = Dataset_GEN['attention_mask']
        self.labels_GEN = Dataset_GEN['labels']
        self.masked_pos_GEN = Dataset_GEN['masked_pos']
        
        self.texts_CLS = Dataset_CLS['input_ids']
        self.att_mask_CLS = Dataset_CLS['attention_mask']
        self.labels_CLS = Dataset_CLS['labels']
        self.masked_pos_CLS = Dataset_CLS['masked_pos']

        self.n_examples = len(self.labels_GEN)
        return

    def __len__(self):
        """When used `len` return the number of examples.
        """
        return self.n_examples

    def __getitem__(self, item):
        """Given an index return an example from the position.
        Arguments:
        item (:obj:`int`):
            Index position to pick an example to return.
        Returns:
        :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
        asociated labels.

        """
        return {'input_ids_gen':self.texts_GEN[item],
                'attention_mask_gen':self.att_mask_GEN[item],
                'labels_gen':self.labels_GEN[item],
                'fair_mask_gen':self.masked_pos_GEN[item],
                'input_ids_cls':self.texts_CLS[item],
                'attention_mask_cls':self.att_mask_CLS[item],
                'labels_cls':self.labels_CLS[item],
                'fair_mask_cls':self.masked_pos_CLS[item]
               }



class Gpt2ClassificationCollatorMain(object):
    """
    Data Collator used for GPT2 Adv Training.
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, cls_id_2_desc_id, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder
        self.max_sequence_len = max_sequence_len
        self.cls_id_2_desc_id = cls_id_2_desc_id


    def __call__(self, dataset_path, aim, sub_sample_size, remove_zero=True):
        """
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
        # step 1.3: convert this mask tokens into ids
        # Get all texts from sequences list.
        #texts = [sequence['text'] for sequence in dataset['train']]
        #inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        if aim == 'train_data':
            dataset = load_dataset(dataset_path)
            texts = []
            for sequence in tqdm(dataset['train']):
                texts.append(sequence['text'])
                if len(texts)>= sub_sample_size:
                    break
        elif aim == 'val_data': 
            texts = self.load_val_dataset(dataset_path)
        
        inputs = self.gen_token_ids(texts) # 10K*256
        labels, keep_rep_i = self.gen_label(inputs['input_ids'])

        # replicate inputs
        inputs_rep = self.rep_data(texts, keep_rep_i)
        # update x and mask
        inputs = self.update_data_and_att(inputs, inputs_rep)
        
        # add labels to data
        inputs.update({'labels':torch.tensor(labels)})

        # remove 0 class
        if remove_zero == True:
            inputs = self.remove_zero_class(inputs, self.max_sequence_len)
        
        # mask fair token
        inputs = self.fair_token_mask(inputs)
        
        return inputs
    
    def fair_token_mask(self, passages):
        n_samples = len(passages['labels'])
        mask_fair_token_matrix = torch.zeros((n_samples, self.max_sequence_len), dtype = torch.long)
        fair_list = []
        for key in range(1, 17):
            fair_list.extend(self.cls_id_2_desc_id[str(key)])

        for i in range(n_samples):
            key = passages['labels'][i].item()
            # fair_list = self.cls_id_2_desc_id[str(key)]
            if fair_list != []:
                for j in range(len(passages['input_ids'][i])):
                    token_id = passages['input_ids'][i][j].item()
                    if token_id in fair_list:
                        mask_fair_token_matrix[i][j] = torch.tensor(-1)      

        passages.update({'fair_mask':mask_fair_token_matrix})
        return passages
    
    def remove_zero_class(self, inputs):
        inputs_new = {}

        # index of nonzero class
        nonzero_class_list = []
        for i, label in enumerate(inputs['labels']):
            if label!= 0:
                nonzero_class_list.append(i)

        # remove zero class from input_ids
        inputs_new['input_ids'] = inputs['input_ids'][nonzero_class_list,]
        # remove zero class from attention_mask
        inputs_new['attention_mask'] = inputs['attention_mask'][nonzero_class_list,]
        # remove zero class from labels
        inputs_new['labels'] = inputs['labels'][nonzero_class_list]

        # count how many nonzero class left
        n_nonzero_class = len(nonzero_class_list)
        print("Number of non-zero ratio %.4f, origin total %d, nonzero total samples %d" % (n_nonzero_class/len(inputs['labels']), len(inputs['labels']), n_nonzero_class))

        return inputs_new
        
    def update_data_and_att(self, inputs, inputs_rep):
        input_c = {}
        input_c['input_ids'] = torch.cat([inputs['input_ids'], inputs_rep['input_ids']], dim=0)
        input_c['attention_mask'] = torch.cat([inputs['attention_mask'], inputs_rep['attention_mask']], dim=0)
        return input_c
        
    def gen_token_ids(self, texts):
        # mask 0: is adding padding
        # mask 1: is adding token
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        return inputs
    
    def rep_data(self, texts, keep_rep_i):
        rep_texts = [texts[index] for index in keep_rep_i]
        inputs_rep = self.use_tokenizer(text=rep_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        return inputs_rep
    
    # assign labels for these examples
    def gen_label(self, inputs):
        # examples: a batch of sentences
        # label_id_map: {dict_id_of_word: label}

        # labels for each sentence
        labels = [[] for _ in range(len(inputs))]
        # iterate over each sentence
        for i in range(len(inputs)):
            # iterate over each word
            for j in range(len(inputs[i])):
                # if word is in the sensitive word list
                word_id = inputs[i][j].item()
                if word_id in self.labels_encoder.keys():
                    # append fair-word id to that example.
                    labels[i].append(self.labels_encoder[word_id])
            # we need to group [2,2,2,2] to a set then a list                
            labels[i] = list(set(labels[i]))

        count_0, count_1, count_2, count_3, count_more = 0,0,0,0,0
        for i in range(len(labels)):           
            if len(labels[i]) == 0:
                count_0 += 1
                labels[i] = [0]
            elif len(labels[i]) == 1:
                count_1 += 1                
            elif len(labels[i]) == 2:
                count_2 += 1
            elif len(labels[i]) == 3:
                count_3 += 1
            else:
                count_more += 1
        
        keep_rep_i, keep_rep_y = [], []

        for i in range(len(labels)):
            if len(labels[i]) >= 2:
                for j in range(1, len(labels[i])):
                    keep_rep_i.append(i) 
                    keep_rep_y.append(labels[i][j])

        # calculate how percentage data have label
        print("Classification Task: 0L %.3f, 1L %.3f, 2L %.3f, 3L %.3f, >3L %.3f" % 
              (count_0/len(labels), count_1/len(labels), count_2/len(labels), count_3/len(labels), count_more/len(labels)))

        # rechange the label            
        label_flat = [z[0] for z in labels]
        label_flat.extend(keep_rep_y)
        labels = label_flat

        return labels, keep_rep_i
    
    def load_val_dataset(self, val_data_path):
        with open(val_data_path, encoding="utf-8") as f:
            text = f.read()
        texts = []
        count = 0
        temp_text = []
        for word in text.split(" "):
            temp_text.append(word)
            count += 1
            if count % self.max_sequence_len == 0:
                count = 0
                texts.append(" ".join(temp_text))
                temp_text = []
        return texts



def random_split_train_val_main(inputs, train_ratio=0.9):
    n_samples = len(inputs['input_ids'])
    val_ratio = 1 - train_ratio
    train_n = int(0.9 * n_samples)
    val_n = n_samples - train_n

    train_list = random.sample(range(n_samples), train_n)
    val_list = [x for x in range(n_samples) if x not in train_list]

    train_data = {}
    train_data['input_ids'] = inputs['input_ids'][train_list,]
    train_data['attention_mask'] = inputs['attention_mask'][train_list,]
    train_data['labels'] = inputs['labels'][train_list]
    train_data['fair_mask'] = inputs['fair_mask'][train_list, ]
    
    
    val_data = {}
    val_data['input_ids'] = inputs['input_ids'][val_list,]
    val_data['attention_mask'] = inputs['attention_mask'][val_list,]
    val_data['labels'] = inputs['labels'][val_list]
    val_data['fair_mask'] = inputs['fair_mask'][val_list, ]
    
    return train_data, val_data    


def Batches_gpt2(batch, device):
    '''
    inputs: batch
    outputs:
        batches_lm: for main model
        batches_aux: for aux model   
    '''
    batches_lm = {}
    batches_aux = {}
    
    batch = {k: v.type(torch.long).to(device) for k,v in batch.items()}
    
    for k,v in batch.items():
        if k == 'input_ids' or k == 'attention_mask': 
            batches_lm[k] = v
            batches_aux[k] = v
    
    batches_lm['labels'] = batches_lm['input_ids']
    batches_aux['labels'] = batch['labels']
    
    batches_lm['fair_mask'] = batch['fair_mask'] + batch['attention_mask']
    batches_aux['attention_mask'] = batch['fair_mask'] + batch['attention_mask']
             
    return batches_lm, batches_aux


def Batches_Bert(batch, device):
    '''
    inputs: batch
    outputs:
        batches_lm: for main model
        batches_aux: for aux model   
    '''
    batches_lm = {}
    batches_aux = {}
    
    batch = {k: v.type(torch.long).to(device) for k,v in batch.items()}
    
    for k,v in batch.items():
        if k == 'input_ids_gen' or k == 'attention_mask_gen': 
            batches_lm[k] = v
        elif k == 'input_ids_cls' or k == 'attention_mask_cls': 
            batches_aux[k] = v
    
    batches_lm['labels_gen'] = batches_lm['input_ids_gen']
    batches_aux['labels_cls'] = batch['labels_cls']
    
    batches_lm['fair_mask_gen'] = batch['fair_mask_gen']
    batches_aux['fair_mask_cls'] = batch['fair_mask_cls']
             
    return batches_lm, batches_aux


def label_id_mapping_gpt2(tokenizer, word_label, descriptors):
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
    for word in descriptors:
        # female = daughter
        class_name = word_label[word]
        # 1 = female class
        label_id = label_class[class_name]
        # 24724 (word dict_id) of female
        dict_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0]
        # 24724: 1
        labels_ids[dict_id] = label_id
    
    cls_id_2_desc_id = {}
    for k,v in labels_ids.items():
        if str(v) not in cls_id_2_desc_id.keys():
            cls_id_2_desc_id[str(v)] = []
            cls_id_2_desc_id[str(v)].append(k)
        else:
            cls_id_2_desc_id[str(v)].append(k)
    cls_id_2_desc_id['0'] = []
    
    #write_json("../data/mask_token/label_id_map.json", label_id_map)
    # not sure whether adjust the embedding will change the id or not
    #label_id_map = load_json("../data/mask_token/label_id_map.json")
    return labels_ids, cls_id_2_desc_id, label_class, class2kingdom



def label_id_mapping_bert(user_tokenizer, word_label, descriptors, tokenizer_method):
    """
    load the label id mapping for BERT/Roberta
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
    
        if tokenizer_method == 'roberta-base-aux':
        # add space for each tokens
            space_dict_id = user_tokenizer.convert_tokens_to_ids(user_tokenizer.tokenize(' '+word))[0]
            labels_ids[space_dict_id] = label_id  
        
    #write_json("../data/mask_token/label_id_map.json", label_id_map)
    # not sure whether adjust the embedding will change the id or not
    #label_id_map = load_json("../data/mask_token/label_id_map.json")
    return labels_ids, label_class, class2kingdom, labels_word




class RoBERTaClassificationCollator(object):
    """
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_word, descriptor, max_sequence_len=256):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # word: class_id
        self.labels_word = labels_word 
        # class_id: word
        self.classid_2_descriptors = self.label_words_transformation()
        # Label encoder used inside the class.
        self.descriptor = descriptor
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

        
    def label_words_transformation(self):
        classid_2_descriptors = {key: [] for key in range(1, 17)}
        for key, val in self.labels_word.items():
            class_id = self.labels_word[key]
            classid_2_descriptors[class_id].append(key)
        return classid_2_descriptors
    
    def __call__(self, dataset, sub_sample_size, remove_zero=True):
        """
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
        # step 1.3: convert this mask tokens into ids
        # Get all texts from sequences list.
        #texts = [sequence['text'] for sequence in dataset['train']]
        #inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        inputs, original_texts = self.gen_token_ids(dataset, sub_sample_size) # 10K*1024
        labels, keep_rep_i, initial_sentences = self.gen_label(inputs['input_ids'])
        print("Initial Tokenize Complete")
        #print('Labels %s' % labels)
        #print('Replicate i: %s' % keep_rep_i)

        # replicate inputs
        inputs_rep, replicate_sentences = self.rep_data(dataset, keep_rep_i, sub_sample_size)
        print("Replicate Data Complete")

        # concatenate full sentence 
        full_sentences = initial_sentences + replicate_sentences
        # concatenate full labels
        full_labels = labels
        # concatenate ids
        full_ids = list(range(sub_sample_size)) + keep_rep_i        
        
        # update x and mask
        inputs_for_CLS = self.update_data_and_att(inputs, inputs_rep)
        #print('Sentence number : %d'  % (len(inputs_for_CLS_and_GPT['input_ids'])))
        
        # transform full sentences to its corresponding sentence with just one mask class.
        inputs_mask_tokenized_for_gancls, mask_IDs = self.mask_transformation(inputs_for_CLS, full_labels, full_ids, original_texts)
        print("Mask Position and Data Create Complete")

        
        # add labels to for CLS GPT
        inputs_for_CLS.update({'labels':torch.tensor(full_labels)})
        inputs_for_CLS.update({'masked_pos': mask_IDs})
        
        #inputs_mask_tokenized_for_gancls.update({'labels':torch.tensor(full_labels)})
        #inputs_mask_tokenized_for_gancls.update({'masked_pos': mask_IDs})

        return inputs_for_CLS

    def remove_zero_class(self, inputs):
        inputs_new = {}

        # index of nonzero class
        nonzero_class_list = []
        for i, label in enumerate(inputs['labels']):
            if label!= 0:
                nonzero_class_list.append(i)

        # remove zero class from input_ids
        inputs_new['input_ids'] = inputs['input_ids'][nonzero_class_list,]
        # remove zero class from attention_mask
        inputs_new['attention_mask'] = inputs['attention_mask'][nonzero_class_list,]
        # remove zero class from labels
        inputs_new['labels'] = inputs['labels'][nonzero_class_list]

        # count how many nonzero class left
        n_nonzero_class = len(nonzero_class_list)
        print("Number of non-zero ratio %.4f, origin total %d, nonzero total samples %d" % (n_nonzero_class/len(inputs['labels']), len(inputs['labels']), n_nonzero_class))

        return inputs_new
        
    def update_data_and_att(self, inputs, inputs_rep):
        input_c = {}
        input_c['input_ids'] = torch.cat([inputs['input_ids'], inputs_rep['input_ids']], dim=0)
        input_c['attention_mask'] = torch.cat([inputs['attention_mask'], inputs_rep['attention_mask']], dim=0)
        return input_c
        
    def gen_token_ids(self, dataset, sub_sample_size):
        # mask 0: is adding padding
        # mask 1: is adding token
        texts = []
        for sequence in tqdm(dataset['train']):
            texts.append(sequence['text'])
            if len(texts)>= sub_sample_size:
                break
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        return inputs, texts
    
    def rep_data(self, dataset, keep_rep_i, sub_sample_size):
        texts = []
        for sequence in tqdm(dataset['train']):
            texts.append(sequence['text'])
            if len(texts)>= sub_sample_size:
                break
        rep_texts = [texts[index] for index in keep_rep_i]        
        inputs_rep = self.use_tokenizer(text=rep_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        
        replicate_sentences = []
        # replicate the sentence
        for i in range(len(inputs_rep['input_ids'])):
            sent = self.use_tokenizer.decode(inputs_rep['input_ids'][i])
            masked_sent = self.text_matching_fair_mask(sent)
            replicate_sentences.append(masked_sent)
        
        return inputs_rep, replicate_sentences
    
    def gen_label(self, inputs):
        # examples: a batch of sentences
        # label_id_map: {dict_id_of_word: label}
        # Transform back of the sentence to the block size text. make sure the sentence (in word level lenght is block_size)

        # labels for each sentence
        sentences = []
        labels = [[] for _ in range(len(inputs))]
        for i in range(len(inputs)):
            sent = self.use_tokenizer.decode(inputs[i])
            masked_sent = self.text_matching_fair_mask(sent)
            sentences.append(masked_sent)
            # find the labels for each sentence
            labels[i] = self.find_label_id(masked_sent)
            # we need to group [2,2,2,2] to a set then a list                
            labels[i] = list(set(labels[i]))
            #print("ID: %d ---  %s" % (i, labels[i]))
        count_0, count_1, count_2, count_3, count_more = 0,0,0,0,0
        for i in range(len(labels)):           
            if len(labels[i]) == 0:
                count_0 += 1
                labels[i] = [0]
            elif len(labels[i]) == 1:
                count_1 += 1                
            elif len(labels[i]) == 2:
                count_2 += 1
            elif len(labels[i]) == 3:
                count_3 += 1
            else:
                count_more += 1

        keep_rep_i, keep_rep_y = [], []

        for i in range(len(labels)):
            if len(labels[i]) >= 2:
                for j in range(1, len(labels[i])):
                    keep_rep_i.append(i) 
                    keep_rep_y.append(labels[i][j])

        # calculate how percentage data have label
        print("Classification Task: 0L %.3f, 1L %.3f, 2L %.3f, 3L %.3f, >3L %.3f" % 
              (count_0/len(labels), count_1/len(labels), count_2/len(labels), count_3/len(labels), count_more/len(labels)))

        # rechange the label            
        label_flat = [z[0] for z in labels]
        label_flat.extend(keep_rep_y)
        labels = label_flat

        return labels, keep_rep_i, sentences
    
    def find_label_id(self, mask_sent):
        mask_list = re.findall( r'MASK_\w+', mask_sent)
        if mask_list == []:
            return []
        else:
            label_id_list = [int(ids[5:len(ids)]) for ids in mask_list]
        #print(label_id_list)
        return label_id_list

    #Text matching
    def text_matching_fair_mask(self, text):
        text = text.lower()
        descriptors = [item.lower() for item in self.descriptor]
        for pattern in descriptors:
            big_regex = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, [pattern])))
            label_id = self.labels_word[pattern]
            text = big_regex.sub("[MASK_%d]" % label_id, text)    
        return text


    def text_matching_fair_mask_only(self, text, class_id):
        text = text.lower()
        # select the descriptors
        #print("MASK ID: %d" % class_id)
        if class_id != 0:
            descriptor = self.classid_2_descriptors[class_id]
            # add the mask token
            patterns = [item.lower() for item in descriptor]
            big_regex = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, patterns)))
            text = big_regex.sub("<mask>", text)    
        
        return text
    
    def mask_transformation(self, inputs, full_labels, full_ids, original_texts):
        n = len(inputs['input_ids'])
        assert n == len(full_labels)
        assert n == len(full_ids)

        # decode the sentence into words
        sentences = []
        for i in range(len(inputs['input_ids'])):
            sent = self.use_tokenizer.decode(inputs['input_ids'][i])
            masked_only_sent = self.text_matching_fair_mask_only(sent, full_labels[i])
            sentences.append(masked_only_sent)
            
        mask_tokenized_input = self.use_tokenizer(text = sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        # check the location id of the mask position
        mask_IDs = [[] for i in range(len(mask_tokenized_input['input_ids']))]
        for i in range(len(mask_tokenized_input['input_ids'])):
            mask_IDs[i] = (mask_tokenized_input['input_ids'][i] == self.use_tokenizer.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        
        return mask_tokenized_input, mask_IDs


class DataCollator(object):
    """
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer_gpt2, use_tokenizer_bert, method_name_gpt2, method_name_bert, labels_word, descriptor, max_sequence_len=256):

        # Tokenizer to be used inside the class.
        self.use_tokenizer_gpt2 = use_tokenizer_gpt2
        self.use_tokenizer_bert = use_tokenizer_bert
        self.method_name_gpt2 = method_name_gpt2
        self.method_name_bert = method_name_bert
        
        # word: class_id
        self.labels_word = labels_word 
        # class_id: word
        self.classid_2_descriptors = self.label_words_transformation()
        # Label encoder used inside the class.
        self.descriptor = descriptor
        # Check max sequence length.
        self.max_sequence_len = max_sequence_len

        
    def label_words_transformation(self):
        classid_2_descriptors = {key: [] for key in range(1, 17)}
        for key, val in self.labels_word.items():
            class_id = self.labels_word[key]
            classid_2_descriptors[class_id].append(key)
        return classid_2_descriptors
    
    def __call__(self, dataset_path, aim, sub_sample_size, remove_zero=True):
        """
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
        # step 1.3: convert this mask tokens into ids
        # Get all texts from sequences list.
        #texts = [sequence['text'] for sequence in dataset['train']]
        #inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        if aim == 'train_data':
            dataset = load_dataset(dataset_path)
            texts = [sequence['text'] for sequence in dataset['train']]
            texts = texts[0:sub_sample_size]
        elif aim == 'val_data': 
            texts = self.load_val_dataset(dataset_path)
            
        inputs, original_texts = self.gen_token_ids(texts) # 10K*1024
        
        labels, keep_rep_i, initial_sentences = self.gen_label(inputs['input_ids'])
        print("Initial Tokenize Complete! \n")
        #print('Labels %s' % labels)
        #print('Replicate i: %s' % keep_rep_i)
        

        # replicate inputs
        inputs_rep, replicate_sentences = self.rep_data(texts, keep_rep_i)
        #print("Replicate Data for CLS Complete! \n")
        
        # concatenate full sentence 
        full_sentences = initial_sentences + replicate_sentences
        # concatenate full labels
        full_labels = labels
        # concatenate ids
        full_ids = list(range(len(texts))) + keep_rep_i        
        
        #print(inputs['input_ids'].size())
        # update x and mask, and need to tokenize with GPT2 
        inputs_for_cls, inputs_for_gpt2 = self.update_data_and_att(inputs, inputs_rep)
        #print(inputs_for_gpt2['input_ids'].size())
        
        mask_IDS_gpt2 = self.mask_transformation_GPT2(inputs_for_gpt2, full_labels, full_ids)
        logger.info("Create Data For GPT2 Complete! \n")
        
        # transform full sentences to its corresponding sentence with just one mask class.
        inputs_mask_tokenized_for_cls, mask_IDs_cls = self.mask_transformation_bert(inputs_for_cls, full_labels, full_ids)
        logger.info("Create Data For RoBERTa Complete! \n")
        
        #for i in range(len(mask_IDS_gpt2)):
        #    if len(mask_IDS_gpt2[i]) != len(mask_IDs_cls[i]):
        #        print('IDS %d' % i)
        
        # add labels to for CLS GPT
        inputs_for_gpt2.update({'labels':torch.tensor(full_labels)})
        inputs_for_gpt2.update({'masked_pos': self.mask_ids_transform_matrix(mask_IDS_gpt2)})
        
        inputs_mask_tokenized_for_cls.update({'labels':torch.tensor(full_labels)})
        inputs_mask_tokenized_for_cls.update({'masked_pos': self.mask_ids_transform_matrix(mask_IDs_cls)})

        return inputs_for_gpt2, inputs_mask_tokenized_for_cls
        
    def mask_ids_transform_matrix(self, mask_IDs):
        # create a matrix for mask ids
        masked_matrix = torch.ones((len(mask_IDs), self.max_sequence_len))
        # transform the masked_pos to matrix
        for i in range(len(mask_IDs)):
            if mask_IDs[i] == []:
                continue
            else:
                for j in mask_IDs[i]: # list of mask ids for one sentence
                    masked_matrix[i][j] = 0
        return masked_matrix
    
    def remove_zero_class(self, inputs):
        inputs_new = {}

        # index of nonzero class
        nonzero_class_list = []
        for i, label in enumerate(inputs['labels']):
            if label!= 0:
                nonzero_class_list.append(i)

        # remove zero class from input_ids
        inputs_new['input_ids'] = inputs['input_ids'][nonzero_class_list,]
        # remove zero class from attention_mask
        inputs_new['attention_mask'] = inputs['attention_mask'][nonzero_class_list,]
        # remove zero class from labels
        inputs_new['labels'] = inputs['labels'][nonzero_class_list]

        # count how many nonzero class left
        n_nonzero_class = len(nonzero_class_list)
        print("Number of non-zero ratio %.4f, origin total %d, nonzero total samples %d" % (n_nonzero_class/len(inputs['labels']), len(inputs['labels']), n_nonzero_class))

        return inputs_new
        
    def update_data_and_att(self, inputs, inputs_rep):
        input_cls = {}
        input_cls['input_ids'] = torch.cat([inputs['input_ids'], inputs_rep['input_ids']], dim=0)
        input_cls['attention_mask'] = torch.cat([inputs['attention_mask'], inputs_rep['attention_mask']], dim=0)
        
        # create input for gpt2
        ori_sentences = []
        for i in range(len(input_cls['input_ids'])):
            sent = self.use_tokenizer_bert.decode(input_cls['input_ids'][i])
            # sentence cleaning
            ori_sent = self.cleaning_sentence(sent)
            ori_sentences.append(" " + ori_sent + "  ")
            #print("ID: %d \n Sentence for GTP2: %s" % (i, ori_sent))
        inputs_for_gpt2 = self.use_tokenizer_gpt2(text=ori_sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)

        return input_cls, inputs_for_gpt2
    
    def cleaning_sentence(self, sent):
        return re.sub(r"<s>|</s>|<pad>", "", sent)
        
    def gen_token_ids(self, texts):
        # mask 0: is adding padding
        # mask 1: is adding token
        inputs = self.use_tokenizer_bert(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        return inputs, texts
    
    def rep_data(self, texts, keep_rep_i):            
        rep_texts = [texts[index] for index in keep_rep_i]        
        inputs_rep = self.use_tokenizer_bert(text=rep_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        
        replicate_sentences = []
        # replicate the sentence
        for i in range(len(inputs_rep['input_ids'])):
            sent = self.use_tokenizer_bert.decode(inputs_rep['input_ids'][i])
            masked_sent = self.text_matching_fair_mask(sent)
            replicate_sentences.append(masked_sent)
        
        return inputs_rep, replicate_sentences
    
    def gen_label(self, inputs):
        # examples: a batch of sentences
        # label_id_map: {dict_id_of_word: label}
        # Transform back of the sentence to the block size text. make sure the sentence (in word level lenght is block_size)

        # labels for each sentence
        sentences = []
        labels = [[] for _ in range(len(inputs))]
        for i in range(len(inputs)):
            sent = self.use_tokenizer_bert.decode(inputs[i])
            masked_sent = self.text_matching_fair_mask(sent)
            sentences.append(masked_sent)
            # find the labels for each sentence
            labels[i] = self.find_label_id(masked_sent)
            # we need to group [2,2,2,2] to a set then a list                
            labels[i] = list(set(labels[i]))
            #print("ID: %d ---  %s" % (i, labels[i]))
            
        count_0, count_1, count_2, count_3, count_more = 0,0,0,0,0
        for i in range(len(labels)):           
            if len(labels[i]) == 0:
                count_0 += 1
                labels[i] = [0]
            elif len(labels[i]) == 1:
                count_1 += 1                
            elif len(labels[i]) == 2:
                count_2 += 1
            elif len(labels[i]) == 3:
                count_3 += 1
            else:
                count_more += 1

        keep_rep_i, keep_rep_y = [], []

        for i in range(len(labels)):
            if len(labels[i]) >= 2:
                for j in range(1, len(labels[i])):
                    keep_rep_i.append(i) 
                    keep_rep_y.append(labels[i][j])

        # calculate how percentage data have label
        print("Classification Task: 0L %.3f, 1L %.3f, 2L %.3f, 3L %.3f, >3L %.3f" % 
              (count_0/len(labels), count_1/len(labels), count_2/len(labels), count_3/len(labels), count_more/len(labels)))

        # rechange the label            
        label_flat = [z[0] for z in labels]
        label_flat.extend(keep_rep_y)
        labels = label_flat
        #print("Labels %s" % labels)
        return labels, keep_rep_i, sentences
    
    def load_val_dataset(self, val_data_path):
        with open(val_data_path, encoding="utf-8") as f:
            text = f.read()
        texts = []
        count = 0
        temp_text = []
        for word in text.split(" "):
            temp_text.append(word)
            count += 1
            if count % self.max_sequence_len == 0:
                count = 0
                texts.append(" ".join(temp_text))
                temp_text = []
        return texts
    
    def find_label_id(self, mask_sent):
        mask_list = re.findall( r'MASK_\w+', mask_sent)
        if mask_list == []:
            return []
        else:
            label_id_list = [int(ids[5:len(ids)]) for ids in mask_list]
        #print(label_id_list)
        return label_id_list

    #Text matching
    def text_matching_fair_mask(self, text):
        text = text.lower()
        descriptors = [item.lower() for item in self.descriptor]
        for pattern in descriptors:
            big_regex = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, [pattern])))
            label_id = self.labels_word[pattern]
            text = big_regex.sub("[MASK_%d]" % label_id, text)    
        return text


    def text_matching_fair_mask_only(self, text, class_id, method_name):
        text = text.lower()
        # select the descriptors
        #print("MASK ID: %d" % class_id)
        if method_name == 'gpt2-main':
            mask_token = "[MASK]"
        elif method_name == 'roberta-base-aux':
            mask_token = "<mask>"
        
        if class_id != 0:
            #Option 1:  Mask sensitive words in that category
            #descriptor = self.classid_2_descriptors[class_id]
            
            #Option 2:  Mask all sensitive word
            descriptor = self.descriptor
            # add the mask token
            patterns = [item.lower() for item in descriptor]
            big_regex = re.compile(r"\b%s\b" % r"\b|\b".join(map(re.escape, patterns)))
            text = big_regex.sub(mask_token, text)    
        
        return text

    
    def mask_transformation_bert(self, inputs, full_labels, full_ids):
        n = len(inputs['input_ids'])
        assert n == len(full_labels)
        assert n == len(full_ids)

        # decode the sentence into words
        sentences = []
        for i in range(len(inputs['input_ids'])):
            sent = self.use_tokenizer_bert.decode(inputs['input_ids'][i])
            masked_only_sent = self.text_matching_fair_mask_only(sent, full_labels[i], self.method_name_bert)
            sentences.append(masked_only_sent)
            #if i == 12:
            #print("ID: %d; \n Masked Sentence: %s" % (i, masked_only_sent))
            
        mask_tokenized_input = self.use_tokenizer_bert(text = sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        # check the location id of the mask position
        mask_IDs = [[] for i in range(len(mask_tokenized_input['input_ids']))]
        for i in range(len(mask_tokenized_input['input_ids'])):
            mask_IDs[i] = (mask_tokenized_input['input_ids'][i] == self.use_tokenizer_bert.mask_token_id).nonzero(as_tuple=True)[0].tolist()
            #if i == 12:
            #print(mask_tokenized_input['input_ids'][i])
            #print("ID: %d, Mask ID FOR BERT: %s" % (i, mask_IDs[i]))
        return mask_tokenized_input, mask_IDs
    
    
    def mask_transformation_GPT2(self, inputs, full_labels, full_ids):
        n = len(inputs['input_ids'])
        assert n == len(full_labels)
        assert n == len(full_ids)

        sentences = []
        for i in range(len(inputs['input_ids'])):
            sent = self.use_tokenizer_gpt2.decode(inputs['input_ids'][i])
            masked_only_sent = self.text_matching_fair_mask_only(sent, full_labels[i], self.method_name_gpt2)
            sentences.append(masked_only_sent)
            #if i == 12:
            #print("ID: %d; \n Masked Sentence: %s" % (i, masked_only_sent))
            
        mask_tokenized_input = self.use_tokenizer_gpt2(text = sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        # check the location id of the mask position
        mask_IDs = [[] for i in range(len(mask_tokenized_input['input_ids']))]
        for i in range(len(mask_tokenized_input['input_ids'])):
            mask_IDs[i] = (mask_tokenized_input['input_ids'][i] == self.use_tokenizer_gpt2.mask_token_id).nonzero(as_tuple=True)[0].tolist()
            #if i == 12:
            #print(mask_tokenized_input['input_ids'][i])
            #print("ID: %d, Mask ID FOR GPT2: %s" % (i, mask_IDs[i]))
        return mask_IDs