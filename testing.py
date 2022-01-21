import sys, os, codecs
import argparse

from sent2graph import SentenceDG

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GCN import RGCN, GAT, Mixed, Parallel

def parse_args():
    parser = argparse.ArgumentParser('GCN data loading and testing script\n')
    parser.add_argument('test_file', metavar='data/test/subtask-1/En-Subtask1-test.tsv',
                        help='Path to the file containing the test instances.')
    parser.add_argument('--language', '-l', dest='lang', default='en',
                        help='Language (en|fr|it)')
    parser.add_argument('--task', '-t', dest='task', default='1',
                        help='Task number (1: classification, 2: regression)')
    parser.add_argument('--model', '-m', dest='model', default='GAT',
                        help='model choice (GAT|RGCN|Mixed|Parallel) default:GAT')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

OPTS = parse_args()

filename=os.path.join(os.getcwd(),OPTS.test_file)

ids=[]
test_data=[]
i=0
with open(filename, 'r') as f:
    content=f.readlines()[1:]
    for c in content:
       (id, text)=c.strip().split('\t')
       ids.append(id)
       test_data.append(text)
       i+=1
       #if i > 10: break # for debug

gc=SentenceDG(OPTS.lang)

NRELS=gc.getNRels()
SIZE=len(test_data)

dataset=[] #array of tensors
i=0
for sent in test_data:
    #print(sent)
    processed = gc.process(sent)
    for sent in processed.sents: #we whould have just one sentence per each instance
        sent_data = gc.sentToPyG(sent, task_type=OPTS.task)
        dataset.append(sent_data)
        #print(sent_data)
    i+=1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

#note: for some reason .eval() is not having effect on the loaded model
#execution depend on random values that create fluctuations in the results
seed = 13
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

batch_size=1
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

num_node_features = dataset[0].num_node_features

if (OPTS.model =='RGCN'):
    model = RGCN(num_node_features, NRELS).to(device) #GCNConv
elif (OPTS.model == 'Mixed'):
    model = Mixed(num_node_features, NRELS).to(device)
elif (OPTS.model == 'Parallel'):
    model = Parallel(num_node_features, NRELS).to(device)
else:
    model = GAT(num_node_features).to(device) #GATConv

model_filename='./state_dict_'+OPTS.model+'_'+OPTS.lang+'_t'+OPTS.task+'.pt'

print("loading state dict...")
model.load_state_dict(torch.load(model_filename))

model.eval()
with torch.no_grad():
    i=0
    for test_data in test_loader:
        test_data.to(device)
        out = model(test_data)
        if OPTS.task == '1':
            res = (torch.round(torch.sigmoid(out)).long()).item()
        else:
            res = np.round(out.item(), 2)
        print(ids[i], res, sep="\t")
        i+=1
