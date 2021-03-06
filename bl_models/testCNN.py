import sys, os, codecs
import argparse
from transformers import RobertaTokenizer, RobertaModel, BarthezTokenizer, AutoModel, AutoTokenizer

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from CNN1d import CNN1d

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def parse_args():
    parser = argparse.ArgumentParser('CNN test script\n')
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
train_datatxt=[]
labels=[]
for filename in os.listdir(folder):
   with open(os.path.join(folder, filename), 'r') as f:
       content=f.readlines()[1:]
       for c in content:
           (id, text, label)=c.strip().split('\t')
           ids.append(id)
           train_datatxt.append(text)
           labels.append(int(label))
           #print(id, label)

#vectorize data using a BERT model (RoBERTa for English, BARThez for French, ...? for Italian)

print('loading model...')
if OPTS.lang=="fr":
    tokenizer = BarthezTokenizer.from_pretrained("moussaKam/barthez")
    model = AutoModel.from_pretrained("moussaKam/barthez")
elif OPTS.lang=="it":
    tokenizer = AutoTokenizer.from_pretrained("idb-ita/gilberto-uncased-from-camembert", do_lower_case=True)
    model = AutoModel.from_pretrained("idb-ita/gilberto-uncased-from-camembert")
else:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained('roberta-base')

SIZE=len(labels)
dataset=[] #array of tensors
i=0
MAXLEN=25 #set max sentence length to 20 (saw 19 in the training set for en, 21 for fr, 23 for it)
for sent in train_datatxt:
    #print(sent)
    inputs = tokenizer(sent, return_tensors="pt")
    row=inputs['input_ids'][0]
    a = torch.zeros(MAXLEN-len(row), dtype=torch.int)
    new_row = torch.cat((row,a), 0) #padding to MAXLEN with zeros
    #print(row)
    #print(new_row)
    dataset.append(new_row)
    i+=1

split_frac = 0.7 # 70% train, 30% test
split_id = int(split_frac * SIZE)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
print(device)

tdataset=torch.stack([d for d in dataset]).to(device)
tlabels=torch.tensor(labels).to(device)

full_dataset = TensorDataset(tdataset, tlabels)
train_data, test_data = torch.utils.data.random_split(full_dataset, [split_id, SIZE-split_id])
print("train size:", len(train_data))
print("test size:", len(test_data))

batch_size = 32

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

N_FILTERS = 32
FILTER_SIZES = [2,3]
hidden_size=16

nnmodel=CNN1d(model, hidden_size, N_FILTERS, FILTER_SIZES)
nnmodel.to(device)

lr=0.1
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 2 #eval every 10 batches (640 input sents)
clip = 5
test_loss_min = np.Inf

nnmodel.train()
for i in range(epochs):
    epoch_acc=[]

    for inputs, labels in train_loader:
        counter += 1

        nnmodel.zero_grad()
        output = nnmodel(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(nnmodel.parameters(), clip)
        optimizer.step()

        train_acc = binary_acc(output.squeeze(), labels.float())
        epoch_acc.append(train_acc.item())

        if counter%print_every == 0:
            test_losses = []
            test_accs=[]
            nnmodel.eval()
            for inp, lab in test_loader:
                out = nnmodel(inp)
                test_loss = criterion(out.squeeze(), lab.float())
                test_acc = binary_acc(out.squeeze(), lab.float())
                test_losses.append(test_loss.item())
                test_accs.append(test_acc.item())

            nnmodel.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Test Loss: {:.6f}".format(np.mean(test_losses)),
                  "Acc: {:.6f}...".format(np.mean(epoch_acc)),
                  "Test Acc: {:.6f}".format(np.mean(test_accs)),
                  )
            if np.mean(test_losses) <= test_loss_min:
                torch.save(nnmodel.state_dict(), './state_dict.pt')
                print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,np.mean(test_losses)))
                test_loss_min = np.mean(test_losses)
