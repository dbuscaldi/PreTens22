import sys, os, codecs
import argparse
import spacy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torchtext
from torchtext.vocab import *

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GCN import GCN


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def getDeps(root_token, numerical=True):
    deps=[]
    for child in root_token.children:
        if (child.dep_ != "punct"):
            if numerical:
                deps.append((labdict[child.dep_], child.text, root_token.text))
            else:
                deps.append((child.dep_, child.text, root_token.text))
        deps=deps+getDeps(child)
    return (deps)

def getPyGDeps(root_token, nodelist):
    edges_ts=[] # edges list (to be transposed)
    edges_attrs=[] # edges attrs (to be transposed)

    if root_token.pos_ != "PUNCT":
        root_id=nodelist.index(root_token.text)
        for child in root_token.children:
            if (child.dep_ != "punct"):
                edges_attrs.append(labdict[child.dep_])
                chld_id=nodelist.index(child.text)
                edges_ts.append((chld_id, root_id))

            (a_u, e_u) = getPyGDeps(child, nodelist)
            edges_attrs = edges_attrs + a_u
            edges_ts = edges_ts + e_u

    return (edges_attrs, edges_ts)

def sentToPyG(sent, wvecs, label):
    #transforms a parsed sentence into a PyTorch Geometric Data object
    nodelist=[]
    for tok in sent:
        if tok.pos_ != "PUNCT":
            nodelist.append(tok.text)
        #print(tok, tok.pos_)
    sz=len(nodelist)

    # associate to each node their word vector
    node_ts=[]
    for node in nodelist:
        emb=wvecs.get_vecs_by_tokens(node)
        node_ts.append(emb)

    x = torch.stack([n for n in node_ts])

    (edges_attrs, edges_ts) = getPyGDeps(sent.root, nodelist)
    e_attr = torch.tensor(edges_attrs, dtype=torch.long)
    e_attr = torch.reshape(e_attr, (len(e_attr), 1)) #reshape to fit the desired format

    edges = torch.tensor(edges_ts, dtype=torch.long)
    edges = edges.t().contiguous() #transpose from original format

    y = torch.tensor(label, dtype=torch.long)
    y = torch.reshape(y, (1,1))

    data = Data(x=x, edge_index=edges, edge_attr=e_attr, y=y)

    return data

def parse_args():
    parser = argparse.ArgumentParser('GCN data loading and prep script\n')
    parser.add_argument('train_dir', metavar='data/en',
                        help='Directory containing the training data.')
    parser.add_argument('--language', '-l', dest='lang', default='en',
                        help='Language (en|fr|it)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

OPTS = parse_args()

folder=os.path.join(os.getcwd(),OPTS.train_dir)
folder=os.path.join(folder,OPTS.lang) #automatically switch directory depending on language

#load training data
ids=[]
train_data=[]
#labels=[]
i=0
for filename in os.listdir(folder):
   with open(os.path.join(folder, filename), 'r') as f:
       content=f.readlines()[1:]
       for c in content:
           (id, text, label)=c.strip().split('\t')
           ids.append(id)
           train_data.append((text, int(label)))
           #labels.append(int(label))
           i+=1
           #if i > 128: break # for debug

#vectorize data using a BERT model (RoBERTa for English, BARThez for French, ...? for Italian)

print('loading SpaCy...')
if OPTS.lang=="fr":
    nlp = spacy.load("fr_core_news_sm")
elif OPTS.lang=="it":
    nlp = spacy.load("it_core_news_sm")
else:
    nlp = spacy.load('en_core_web_sm')

dplabels=nlp.get_pipe("parser").labels
#make label dictionary that maps labels into a numeric value
id=0
labdict={}
for lab in dplabels:
    labdict[lab]=id
    id+=1

NRELS=id-1
print(NRELS)
#loading word embeddings:
wvecs=torchtext.vocab.FastText(language=OPTS.lang)

SIZE=len(train_data)
dataset=[] #array of tensors
i=0
for sent, label in train_data:
    print(sent)
    processed = nlp(sent)
    for sent in processed.sents: #we whould have just one sentence per each instance
        sent_data = sentToPyG(sent, wvecs, label)
        dataset.append(sent_data)
        #print(sent_data)
    i+=1

#loader = DataLoader(dataset, batch_size=64, shuffle=True)
#for batch in loader:
#    print(batch)

split_frac = 0.7 # 70% train, 30% test
split_id = int(split_frac * SIZE)

dataset_train = dataset[0:split_id]
dataset_test = dataset[split_id:]

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
print(device)

batch_size=32 #note: to fix loader as it groups graphs into the same batch
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

num_node_features = dataset_train[0].num_node_features

model = GCN(num_node_features, NRELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss()

epochs = 2
counter = 0
print_every = 10 #print info each 10 batches
test_loss_min = np.Inf

model.train()
i=0
for epoch in range(epochs):
    epoch_acc=[]
    for data in train_loader:
        counter += 1
        #print(data)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()

        train_acc = binary_acc(out, data.y.float())
        epoch_acc.append(train_acc.item())

        if counter%print_every == 0:
            test_losses = []
            test_accs=[]
            model.eval()
            for test_data in test_loader:
                out = model(test_data)
                lab = test_data.y
                test_loss = criterion(out, lab.float())
                test_acc = binary_acc(out, lab.float())
                test_losses.append(test_loss.item())
                test_accs.append(test_acc.item())

            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Test Loss: {:.6f}".format(np.mean(test_losses)),
                  "Acc: {:.6f}...".format(np.mean(epoch_acc)),
                  "Test Acc: {:.6f}".format(np.mean(test_accs)),
                  )
    i+=1

"""

            if np.mean(test_losses) <= test_loss_min:
                torch.save(nnmodel.state_dict(), './state_dict.pt')
                print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,np.mean(test_losses)))
                test_loss_min = np.mean(test_losses)
"""
