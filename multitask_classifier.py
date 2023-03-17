import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from pcgrad import PCGrad

from datasets import SentenceClassificationDataset, SentencePairDataset, SingleLineDataset, InferenceDataset, \
    load_multitask_data, load_pretrain_data, load_inference_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask, model_eval_inference, model_eval_pretrain_domain

from lib2to3.pgen2.tokenize import tokenize
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from evaluation import model_eval_inference
from itertools import cycle

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
BERT_VOCAB_SIZE = 30522

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### IMPLEMENTED
        self.sentiment_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.sentiment_dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.paraphrase_linear = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.paraphrase_dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.similarity_linear = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.similarity_dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.pretrain_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.pretrain_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_VOCAB_SIZE) 

        self.inference_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.inference_linear = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 3) 


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### IMPLEMENTED
        return self.bert(input_ids, attention_mask)['pooler_output'] # this has cls token hidden state

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### IMPLEMENTED
        res = self.bert(input_ids, attention_mask)['pooler_output'] # this has cls token hidden state
        res = self.sentiment_dropout(res)
        return self.sentiment_linear(res)

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### IMPLEMENTED
        res1 = self.bert(input_ids_1, attention_mask_1)['pooler_output'] # this has cls token hidden state
        res2 = self.bert(input_ids_2, attention_mask_2)['pooler_output'] # this has cls token hidden state
        res1 = self.paraphrase_dropout(res1)
        res2 = self.paraphrase_dropout(res2)
        res = torch.cat((res1,res2),-1)
        return self.paraphrase_linear(res)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### IMPLEMENTED
        res1 = self.bert(input_ids_1, attention_mask_1)['pooler_output'] # this has cls token hidden state
        res2 = self.bert(input_ids_2, attention_mask_2)['pooler_output'] # this has cls token hidden state
        res1 = self.similarity_dropout(res1)
        res2 = self.similarity_dropout(res2)
        res = torch.cat((res1,res2),-1)
        return self.similarity_linear(res).squeeze(-1)
    
    def predict_domain_data(self, input_ids, attention_mask):
        res = self.bert(input_ids, attention_mask)['last_hidden_state'] # this has cls token hidden state
        res = self.pretrain_dropout(res)
        return self.pretrain_linear(res)
    
    def predict_inference(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        '''Given a pair of sentences, outputs logits for classifying inference.
        There are 3 inference classes, should have 3 logits for each pair of sentences'''
        #Returning first token of hidden state
        res1 = self.bert(input_ids1, attention_mask1)['pooler_output']
        res2 = self.bert(input_ids2, attention_mask2)['pooler_output'] 
        #Applying dropout layers
        res1 = self.inference_dropout(res1)
        res2 = self.inference_dropout(res2)
        res = torch.cat((res1,res2),-1)
        return self.inference_linear(res)

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args) 

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)
    if args.pretrained_weights_path:
        saved = torch.load(args.pretrained_weights_path)
        model.load_state_dict(saved['model'],  strict=False)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
       
        zip_list = zip(tqdm(cycle(sst_train_dataloader), desc=f'train-{epoch}', disable=TQDM_DISABLE), 
                        tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE),
                        tqdm(cycle(sts_train_dataloader), desc=f'train-{epoch}', disable=TQDM_DISABLE))
        for batch_sst, batch_para, batch_sts in zip_list: 
            average_loss = 0
            
            #SENTIMENT
            b_ids_sst, b_mask_sst, b_labels_sst = (batch_sst['token_ids'],
                                       batch_sst['attention_mask'], batch_sst['labels'])

            b_ids_sst = b_ids_sst.to(device)
            b_mask_sst = b_mask_sst.to(device)
            b_labels_sst = b_labels_sst.to(device)

            # optimizer.zero_grad()
            logits_sst = model.predict_sentiment(b_ids_sst, b_mask_sst)
            loss_sst = F.cross_entropy(logits_sst, b_labels_sst.view(-1), reduction='sum') / args.batch_size

            # loss_sst.backward()
            # optimizer.step()

            average_loss += loss_sst.item()
            
            #PARAPHRASING
            (b_ids1_para, b_mask1_para,
             b_ids2_para, b_mask2_para,
             b_labels_para, b_sent_ids_para) = (batch_para['token_ids_1'], batch_para['attention_mask_1'],
                          batch_para['token_ids_2'], batch_para['attention_mask_2'],
                          batch_para['labels'], batch_para['sent_ids'])

            b_ids1_para = b_ids1_para.to(device)
            b_mask1_para = b_mask1_para.to(device)
            b_ids2_para = b_ids2_para.to(device)
            b_mask2_para = b_mask2_para.to(device)
            b_labels_para = b_labels_para.to(device)

            # optimizer.zero_grad()
            logits_para = model.predict_paraphrase(b_ids1_para, b_mask1_para, b_ids2_para, b_mask2_para)
            loss_para = F.binary_cross_entropy(logits_para.sigmoid().view(-1), b_labels_para.view(-1).float(), reduction='mean')
            
            # loss_para.backward()
            # optimizer.step()

            average_loss += loss_para.item()

            #SIMILARITY
            (b_ids1_sts, b_mask1_sts,
             b_ids2_sts, b_mask2_sts,
             b_labels_sts, b_sent_ids_sts) = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
                          batch_sts['token_ids_2'], batch_sts['attention_mask_2'],
                          batch_sts['labels'], batch_sts['sent_ids'])

            b_ids1_sts = b_ids1_sts.to(device)
            b_mask1_sts = b_mask1_sts.to(device)
            b_ids2_sts = b_ids2_sts.to(device)
            b_mask2_sts = b_mask2_sts.to(device)
            b_labels_sts = b_labels_sts.to(device)

            optimizer.zero_grad()
            logits_sts = model.predict_similarity(b_ids1_sts, b_mask1_sts, b_ids2_sts, b_mask2_sts)
            # loss_sts = torch.nn.CosineEmbeddingLoss(logits_sts, b_labels_sts.float(),)
            # loss_sts = torch.nn.CosineSimilarity()
            loss_sts = F.mse_loss(logits_sts, b_labels_sts.float())

            # loss_sts.backward()
            # optimizer.step()
            losses = [loss_sst, loss_para, loss_sts]
            optimizer.pc_backward(losses) # calculate the gradient can apply gradient modification
            optimizer.step()  # apply gradient step

            average_loss += loss_sts.item()
            
            #For each batch, compute average of all losses
            train_loss += average_loss / 3
            num_batches += 1

        train_loss = train_loss / (num_batches)
        paraphrase_accuracy, _, _, sentiment_accuracy, _, _, sts_corr, _, _= model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_paraphrase_accuracy, _, _, dev_sentiment_accuracy, _, _, dev_sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        train_acc = paraphrase_accuracy+sentiment_accuracy+sts_corr
        dev_acc  = dev_paraphrase_accuracy+dev_sentiment_accuracy+dev_sts_corr
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)

def pretrain(args):
    assert(args.pretrained_weights_path)
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader

    #DOMAIN_DATASET
    domain_train_data = load_pretrain_data(args.sst_train,args.para_train,args.sts_train)
    domain_dev_data = load_pretrain_data(args.sst_dev,args.para_dev,args.sts_dev)

    domain_data = SingleLineDataset(domain_train_data, args)
    domain_data_dataloader = DataLoader(domain_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=domain_data.collate_fn)

    domain_dev_data = SingleLineDataset(domain_dev_data, args)
    domain_dev_data_dataloader = DataLoader(domain_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=domain_dev_data.collate_fn)
    
    #INFERENCE_DATASET
    inference_train_data = load_inference_data('multinli_1.0/multinli_1.0_train.jsonl')
    inference_dev_data = load_inference_data('multinli_1.0/multinli_1.0_dev_matched.jsonl')

    inference_train_dataset = InferenceDataset(inference_train_data, args)
    inference_dev_dataset = InferenceDataset(inference_dev_data, args)
    
    inference_train_dataloader = DataLoader(inference_train_dataset, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=inference_train_dataset.collate_fn)
    inference_dev_dataloader = DataLoader(inference_dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=inference_dev_dataset.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    optimizer = PCGrad(optimizer) 
    best_dev_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        if len(domain_data_dataloader) > len(inference_train_dataloader):
            zip_list = zip(tqdm(domain_data_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE), cycle(tqdm(inference_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE))) 
        else:
            zip_list = zip(cycle(tqdm(domain_data_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)), tqdm(inference_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE))
        for domain_batch, inference_batch in zip_list:
            
            average_loss = 0
            
            #DOMAIN
            domain_ids, domain_mask, domain_labels, domain_chosen = (domain_batch['token_ids'],
                                       domain_batch['attention_mask'], domain_batch['labels'], domain_batch['chosen'])
            
            domain_chosen = domain_chosen.to(device)
            domain_ids = domain_ids.to(device)
            domain_mask = domain_mask.to(device)
            domain_labels = domain_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_domain_data(domain_ids, domain_mask)
            logits = logits[domain_chosen[:,0], domain_chosen[:,1]]
            domain_labels = domain_labels[domain_chosen[:,0], domain_chosen[:,1]]
            loss1 = F.cross_entropy(logits, domain_labels.view(-1), reduction='sum') / args.batch_size

            average_loss += loss1.item()

            #INFERENCE
            (inference_ids1, inference_mask1,
             inference_ids2, inference_mask2,
             inference_labels, inference_sent_ids) = (inference_batch['token_ids_1'], inference_batch['attention_mask_1'],
                          inference_batch['token_ids_2'], inference_batch['attention_mask_2'],
                          inference_batch['labels'], inference_batch['sent_ids'])

            inference_ids1 = inference_ids1.to(device)
            inference_mask1 = inference_mask1.to(device)
            inference_ids2 = inference_ids2.to(device)
            inference_mask2 = inference_mask2.to(device)
            inference_labels = inference_labels.to(device)

            logits = model.predict_inference(inference_ids1, inference_mask1, inference_ids2, inference_mask2)
            loss2 = F.cross_entropy(logits, inference_labels.view(-1), reduction='sum') / args.batch_size

            num_batches += 1
            average_loss += loss2.item()
            train_loss += average_loss/2

            losses = [loss1, loss2]
            optimizer.pc_backward(losses) # calculate the gradient can apply gradient modification
            optimizer.step()  # apply gradient step

        domain_train_acc = model_eval_pretrain_domain(domain_data_dataloader, model, device)
        domain_dev_acc = model_eval_pretrain_domain(domain_dev_data_dataloader, model, device)

        inference_train_accuracy, _, _  = model_eval_inference(inference_train_dataloader, model, device)
        inference_dev_accuracy, _, _ = model_eval_inference(inference_dev_dataloader, model, device)

        train_acc = domain_train_acc + inference_train_accuracy
        dev_acc = domain_dev_acc + inference_dev_accuracy

        dev_acc = 1
        train_acc=1

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.pretrained_weights_path + f'full-pretrained-epoch{epoch}-lr{args.lr}.pt')

        train_loss = train_loss / (num_batches)
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--pretrained_weights_path", type=str)

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    if args.pretrained_weights_path and args.option == "finetune":
        pretrain(args)
    else:
        train_multitask(args)
        test_model(args)
