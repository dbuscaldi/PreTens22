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

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def parse_args():
    parser = argparse.ArgumentParser('GCN data loading and training script\n')
    parser.add_argument('train_dir', metavar='data/en',
                        help='Directory containing the training data.')
    parser.add_argument('--language', '-l', dest='lang', default='en',
                        help='Language (en|fr|it)')
    parser.add_argument('--task', '-t', dest='task', default='1',
                        help='Task number (1: classification, 2: regression)')
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
           if OPTS.task == '1':
               train_data.append((text, int(label)))
           else:
               train_data.append((text, float(label)))
           #labels.append(int(label))
           i+=1
           #if i > 128: break # for debug

gc=SentenceDG(OPTS.lang)

NRELS=gc.getNRels()
SIZE=len(train_data)
dataset=[] #array of tensors
i=0
for sent, label in train_data:
    #print(sent)
    processed = gc.process(sent)
    for sent in processed.sents: #we whould have just one sentence per each instance
        sent_data = gc.sentToPyG(sent, label, OPTS.task)
        dataset.append(sent_data)
        #print(sent_data)
    i+=1

split_frac = 0.9 # 70% train, 30% test
split_id = int(split_frac * SIZE)

dataset_train = dataset[0:split_id]
dataset_test = dataset[split_id:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
print(device)

batch_size=16
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

num_node_features = dataset_train[0].num_node_features

#model = RGCN(num_node_features, NRELS).to(device) #GCNConv
#model = GAT(num_node_features).to(device) #GATConv
#model = Mixed(num_node_features, NRELS).to(device)
model = Parallel(num_node_features, NRELS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
if OPTS.task == '1':
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.MSELoss()

epochs = 5
counter = 0
print_every = 10 #print info each 10 batches
test_loss_min = np.Inf

model.train()
i=0
for epoch in range(epochs):
    epoch_acc=[]
    for data in train_loader:
        data.to(device)
        counter += 1
        #print(data)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()

        if OPTS.task == '1':
            train_acc = binary_acc(out, data.y.float())
            epoch_acc.append(train_acc.item())

        if counter%print_every == 0:
            test_losses = []
            test_accs=[]
            model.eval()
            for test_data in test_loader:
                test_data.to(device)
                out = model(test_data)
                lab = test_data.y
                test_loss = criterion(out, lab.float())
                test_losses.append(test_loss.item())
                if OPTS.task == '1':
                    test_acc = binary_acc(out, lab.float())
                    test_accs.append(test_acc.item())

            model.train()
            if OPTS.task == '1':
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Test Loss: {:.6f}".format(np.mean(test_losses)),
                      "Acc: {:.6f}...".format(np.mean(epoch_acc)),
                      "Test Acc: {:.6f}".format(np.mean(test_accs)),
                      )
            else:
                print("Epoch: {}/{}...".format(i+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Test Loss: {:.6f}".format(np.mean(test_losses))
                      )

            if np.mean(test_losses) <= test_loss_min:
                model_filename='./state_dict_'+OPTS.lang+'_t'+OPTS.task+'.pt'
                torch.save(model.state_dict(), model_filename)
                print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,np.mean(test_losses)))
                test_loss_min = np.mean(test_losses)

    i+=1
