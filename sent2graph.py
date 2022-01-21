import sys, os, codecs
import spacy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torchtext
from torchtext.vocab import *

import torch_geometric
from torch_geometric.data import Data

class SentenceDG:

    def __init__(self, language='en', directed=True):
        self.DIRECTED=directed
        print('loading SpaCy...')
        if language=="fr":
            self.nlp = spacy.load("fr_core_news_sm")
        elif language=="it":
            self.nlp = spacy.load("it_core_news_sm")
        else:
            self.nlp = spacy.load('en_core_web_sm')

        dplabels=self.nlp.get_pipe("parser").labels
        #make label dictionary that maps labels into a numeric value
        id=0
        self.labdict={}
        for lab in dplabels:
            self.labdict[lab]=id
            id+=1
        self.NRELS=id-1
        print("loading vectors...")
        self.wvecs=torchtext.vocab.FastText(language=language)

    def process(self, sent):
        return self.nlp(sent)

    def getNRels(self):
        return self.NRELS

    def getDeps(self, root_token, numerical=True):
        deps=[]
        for child in root_token.children:
            if (child.dep_ != "punct"):
                if numerical:
                    deps.append((self.labdict[child.dep_], child.text, root_token.text))
                else:
                    deps.append((child.dep_, child.text, root_token.text))
            deps=deps+self.getDeps(child)
        return (deps)

    def getPyGDeps(self, root_token, nodelist):
        edges_ts=[] # edges list (to be transposed)
        edges_attrs=[] # edges attrs (to be transposed)

        if root_token.pos_ != "PUNCT":
            root_id=nodelist.index(root_token.text)
            for child in root_token.children:
                try:
                    chld_id=nodelist.index(child.text)
                except ValueError:
                    continue
                if (child.dep_ != "punct"):
                    edges_attrs.append(self.labdict[child.dep_])
                    if(not self.DIRECTED): edges_attrs.append(self.labdict[child.dep_])
                    edges_ts.append((chld_id, root_id))
                    if(not self.DIRECTED): edges_ts.append((root_id, chld_id))

                (a_u, e_u) = self.getPyGDeps(child, nodelist)
                edges_attrs = edges_attrs + a_u
                edges_ts = edges_ts + e_u

        return (edges_attrs, edges_ts)

    def sentToPyG(self, sent, label, task_type='1'):
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
            emb=self.wvecs.get_vecs_by_tokens(node)
            node_ts.append(emb)

        x = torch.stack([n for n in node_ts])

        (edges_attrs, edges_ts) = self.getPyGDeps(sent.root, nodelist)
        e_attr = torch.tensor(edges_attrs, dtype=torch.long)
        e_attr = torch.reshape(e_attr, (len(e_attr), 1)) #reshape to fit the desired format

        edges = torch.tensor(edges_ts, dtype=torch.long)
        edges = edges.t().contiguous() #transpose from original format

        if task_type == '1':
            y = torch.tensor(label, dtype=torch.long)
        else:
            y = torch.tensor(label, dtype=torch.double)
        y = torch.reshape(y, (1,1))

        data = Data(x=x, edge_index=edges, edge_attr=e_attr, y=y)

        return data
