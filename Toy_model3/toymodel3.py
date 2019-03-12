#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:30:34 2019

@author: hoshu
"""

"""
Code that predicts a sentence given a random words as input. 

"""

import torch
from torch.autograd import Variable
import numpy as np
#import torch.functional as F
import torch.nn.functional as F
import torch.nn as N
import matplotlib.pyplot as plt
import string

# %%
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',   
]

# %%


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(corpus)

# %%

### Tokenize movie text
###
f=open("movie_lines.txt", "r")

def tokenize_corpus2(corpus2):
    tokens2=[]
    counter = 0
    
    for x in corpus2:
        s = x.split(" +++$+++ ")
        part_S = s[-1]        
        part_S = part_S.translate(str.maketrans('','',string.punctuation))
        print(part_S)
        separate = part_S.split()
        #separate = separate.split(".")
        print('separate words:',separate)
        tokens2.append(separate)
        counter+=1
        if counter>=70: #5404:
            break
    return tokens2
    
tokenized_corpus2=tokenize_corpus2(f)
tokenized_corpus=tokenized_corpus2

#print(*tokenized_corpus)

# %%


vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)
            
print("\n full vocabulary: \n", vocabulary)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)
print("dictionary of words: \n", word2idx)
print("number of words in the dictionary", vocabulary_size)
#print(vocabulary)
#print(list(enumerate(vocabulary)))
#print(word2idx)
#print(idx2word)
#print(word2idx["a"])
#Center word context words pairs, with symmetric witndow=2

# %%


window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array 
#reproducing pairs
#for pair in idx_pairs:
 #  print(pair)
  # print(idx2word[pair[0]], idx2word[pair[1]])
  
# %%

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 101
learning_rate = 0.001
loss_history=[]
for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        ###satutday#print(data, target)
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())
        #z3=np.array([target]) #target is a number so do z3
        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0) #exp/sum of exp of each element of z2
        ####print(log_softmax)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true) #view(1,-1) shape 1rowxn columns
        #print(loss.item())
        loss_val+=loss.item()
        #print(type(loss))
        #loss_val += loss.data[0]
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
        #if epo % 10 == 0:  
         #   print("\n x=",x, "\n y_true=", y_true, "\n target=",type( target))
    loss_history.append(loss_val/len(idx_pairs))    
    if epo % 10 == 0:    
        print("Loss at epo",epo,":"  ,loss_val/len(idx_pairs))
       # print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

# %%

epoch_count=range(1,num_epochs+1) 
plt.plot(epoch_count,loss_history,'b--')
plt.legend(["Training Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()   
#print(W2.data)

"""
Here we produce a sentence given an input word that is found in the dictionary, 
the model will select the next word that is more likely to be close to the input word
"""

def sentence_production(first_word):
    hotvector=get_input_layer(word2idx[first_word])
    z1=torch.matmul(W1,hotvector)
    z2=torch.matmul(W2,z1)
    log_softmax=F.log_softmax(z2,dim=0)
    values, indices = torch.max(torch.exp(log_softmax), 0)
    next_word=idx2word[indices.item()]
    return(next_word)

size_of_sentence=5
first_word=" "
while first_word!="ESC":
    first_word=input("Please type a word , I will make a sentence or type ESC to exit \n")
    if first_word=="ESC": continue
    elif first_word not in word2idx:
        print("not in my dictionary, write another word: \n")
        continue 
    else:
        sentence=[]
        counter=0
        while counter<=size_of_sentence:
            sentence.append(first_word)
            first_word=sentence_production(first_word)
            counter+=1
print("Your sentence is: \n ", *sentence)
