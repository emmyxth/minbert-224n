#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''


import csv
import json

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import random
import numpy as np


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class InferenceDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(self, string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]
        labels = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids1 = torch.LongTensor(encoding1['input_ids'])
        attention_mask1 = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids1 = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

        labels = torch.LongTensor(labels)

        return (token_ids1, token_type_ids1, attention_mask1,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids1, token_type_ids1, attention_mask1,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids1,
                'token_type_ids_1': token_type_ids1,
                'attention_mask_1': attention_mask1,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data

class SingleLineDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        encoding = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True)
        labels = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        # 15% of the token positions at random for prediction
        batch_size, _ = encoding["input_ids"].shape
        token_ids = []
        chosen = []
        
        for sent_id in range(batch_size):
            token_ids.append([])
            endOfSequence = int((labels[sent_id] == self.tokenizer.sep_token_id).nonzero())
            indicies = random.sample(range(1, endOfSequence), round((endOfSequence-1)*.15))
            for i in range(len(labels[sent_id])):
                if i not in indicies:
                    token_ids[sent_id].append(int(labels[sent_id][i]))
                else:
                    num = random.randint(1,10)
                    if num <=8:
                        # then in 80% of these cases the token is replaced [MASK],
                        token_ids[sent_id].append(self.tokenizer.mask_token_id)
                    elif num <= 9:
                        # in 10% of cases the token is replaced with a random token, 
                        token_ids[sent_id].append(np.random. randint(0, 30521))
                    else:
                        # and in another 10% of cases, the token will remain unchanged.
                        token_ids[sent_id].append(labels[sent_id][i])
            for val in indicies:
                chosen.append([sent_id, val])
            
        token_ids = torch.LongTensor(token_ids)
        token_ids = torch.reshape(token_ids, (batch_size,-1))
        chosen = torch.LongTensor(chosen)

        return labels, token_ids, attention_mask, data, chosen
    def collate_fn(self, all_data):
        labels, token_ids, attention_mask, sents, chosen = self.pad_data(all_data)

        #add in label array for original tokens that were masked out
        batched_data = {
                'labels' : labels,
                'token_ids': token_ids, #15 percent are masked , etc. 
                'attention_mask': attention_mask,
                'sents': sents,
                'chosen': chosen
            }

        return batched_data

class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression =False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data


def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'

    sentiment_data = []

    with open(sentiment_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data

#Loading data from JSON file as [(sent1, sent2, pairid, label)]
def load_inference_data(filename):
    data = []
    LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
    }
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            sent1 = loaded_example['sentence1'].lower().strip()
            sent2 = loaded_example["sentence2"].lower().strip()
            pairid = loaded_example["pairID"].lower().strip()
            label = LABEL_MAP[loaded_example["gold_label"]]
            data.append((sent1, sent2, pairid, label))
        random.seed(1)
        random.shuffle(data)
    print(f"load {len(data)} data from {filename}")
    return data


def load_pretrain_data(sentiment_filename,paraphrase_filename,similarity_filename):
    sentiment_data = []
    num_labels = {}
    with open(sentiment_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} train examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            paraphrase_data.append(preprocess_string(record['sentence1']))
            paraphrase_data.append(preprocess_string(record['sentence2']))

    print(f"Loaded {len(paraphrase_data)} train examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            similarity_data.append(preprocess_string(record['sentence1']))
            similarity_data.append(preprocess_string(record['sentence2']))

    print(f"Loaded {len(similarity_data)} train examples from {similarity_filename}")

    return sentiment_data + paraphrase_data + similarity_data

def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data
