from lib2to3.pgen2.tokenize import tokenize
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import csv
import json
import collections

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from evaluation import model_eval_inference

# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

TQDM_DISABLE=False
BERT_HIDDEN_SIZE=768

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

#Loading data from JSON file as [(sent1, sent2, pairid, label)]
def load_data(filename):
    data = []
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

class BertInference(torch.nn.Module):
    def __init__(self, config):
        super(BertInference, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #Updating BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 3)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        '''Given a pair of sentences, outputs logits for classifying inference.
        There are 4 inference classes, should have 4 logits for each pair of sentences'''
        #Returning first token of hidden state
        res1 = self.bert(input_ids1, attention_mask1)['pooler_output']
        res2 = self.bert(input_ids2, attention_mask2)['pooler_output'] 
        #Applying dropout layers
        res1 = self.dropout(res1)
        res2 = self.dropout(res2)
        res = torch.cat((res1,res2),-1)
        return self.linear(res)

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    train_data = load_data(args.train)
    dev_data = load_data(args.dev)

    #print(train_data)

    train_dataset = InferenceDataset(train_data, args)
    dev_dataset = InferenceDataset(dev_data, args)
    
    inf_train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=train_dataset.collate_fn)
    inf_dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev_dataset.collate_fn)
    
    # Init model
    #saved = torch.load(args.pretrained_weights_path)

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': len(LABEL_MAP),
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)
    model = BertInference(config)
    model = model.to(device)
    #model.load_state_dict(saved['model'])

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(inf_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids1, b_mask1, b_ids2, b_mask2)
            print("LOGITS", logits.shape)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_accuracy, train_y_pred, train_sent_ids  = model_eval_inference(inf_train_dataloader, model, device)
        dev_accuracy, dev_y_pred, dev_sent_ids = model_eval_inference(inf_dev_dataloader, model, device)

        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_accuracy :.3f}, dev acc :: {dev_accuracy :.3f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
                                    

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=4)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    #args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt'

    print('Training inference on Inference Dataset...')
    config = SimpleNamespace(
        filepath='inf-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='multinli_1.0/multinli_1.0_train.jsonl',
        dev='multinli_1.0/multinli_1.0_dev_matched.jsonl',
        test='data/ids-sst-test-student.csv',
        option=args.option,
        dev_out = 'predictions/'+args.option+'-sst-dev-out.csv',
        test_out = 'predictions/'+args.option+'-sst-test-out.csv'
    )

    train(config)
