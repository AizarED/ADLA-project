"""
Author: Aizar E. D. and Nicholas
Aizar: Added 2 new ways to calculate the probabilities to predict sentences.
all the words are in lower case.
New graph to show the projection of the embbeded vectors in 3 dimensions. 
"""

import torch
from torch.autograd import Variable
import numpy as np
#import torch.functional as F
import torch.nn.functional as F
import torch.nn as N
import matplotlib.pyplot as plt
import string
from mpl_toolkits.mplot3d import Axes3D

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
        part_S = part_S.translate(str.maketrans('','',string.punctuation)).lower()
        print("Part_S:", part_S)
        separate = part_S.split()
        #separate = separate.split("\n")
        ##print('separate words:',separate)
        tokens2.append(separate)
        counter+=1
        if counter>=1000:#80:# 70: #10:# 200:#70: #5404:
            break
    return tokens2
    
tokenized_corpus2=tokenize_corpus2(f)
tokenized_corpus=tokenized_corpus2

#print(*tokenized_corpus)

# %%

# Create vocab

vocabulary = []

# For each sentence
for sentence in tokenized_corpus:
    # for each word/token in sentence
    for token in sentence:
        if token not in vocabulary: # append new words to vocab
            vocabulary.append(token)
            
print("\n full vocabulary: \n", vocabulary)

# Use enumerate to generate index for vocabulary 
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)} # word and number
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)} # number and word

vocabulary_size = len(vocabulary)
print("dictionary of words: \n", word2idx)
print("number of words in the dictionary", vocabulary_size)
#print(vocabulary)
#print(list(enumerate(vocabulary)))
#print(word2idx)
#print(idx2word)
#print(word2idx["a"])
#Center word context words pairs, with symmetric witndow=2

# %% Generate pairs, centre and context word

window_size = 2 #2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence] # e.g. word2idx["aboard"]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1): # shift window across. Works well for window_size = 2 with a four letter sentence
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue # i.e. forget lines below and loop again. i.e. one time break
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx)) # create pairs

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array 
#reproducing pairs
#for pair in idx_pairs:
 #  print(pair)
  # print(idx2word[pair[0]], idx2word[pair[1]])
  
# %% Input layer
  
def get_input_layer(word_idx): # use word_idx("King") to input index of word
    x = torch.zeros(vocabulary_size).float() # Create zeros vector which has the size of the entire vocab
    x[word_idx] = 1.0 # For the particular word. Set value of array element to 1 => i.e. create one-hot.
    return x

# %% define neural network
    
embedding_dims = 5 # 5 embedded neurons
    
class SG_NN(torch.nn.Module):
    def __init__(self):
        #initialize weights matrices
        super().__init__()
        # input is a 1x(vocab length)
        # 2 layers because there are 2 weights?????
       self.dense1 = torch.nn.Linear(vocabulary_size,?) # don't know dimensions of second layer
       self.dense2 = torch.nn.Linear(?, embedding_dims) # why linear??????
    
    def forward(self,a):
        a = F.relu((self.dense1(a.view(-1,vocabulary_size))))
        a = F.log_softmax(self.dense2(a),dim=1) # log_softmax instead of softmax
        return a

# %% Instantiate model and optimizer 
lr = 0.001 # learning rate
mymodel = SG_NN() 
optimizer = torch.optim.Adam(mymodel.parameters(),lr=lr)

plt.ion()

# %% Define training loop

epochs = 101
data_in = idx_pairs

def train(epochs,data_in):
    plt.close()
    mymodel.train() # no argruments in train?
    
    for e in range(epochs):
        loss_val = 0
        for data, target in idx_pairs:
            x = get_input_layer(data).float()
            h = mymodel.forward(x) # hypothesis, i.e. get log_softmax
            
            y_true = torch.from_numpy(np.array([target])).long()
            loss = F.nll_loss(log_softmax.view(1,-1),y_true)
            
            loss_val+=loss.item()
            
            loss.backward()
            optimizer.step() # step????
            optimizer.zero_grad()       
        


# %%

embedding_dims = 5 # 5 embedded neurons
# W1 is center word weights vector
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True) 

# W2 is the context weights vector
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
        ##saturday##print(log_softmax)

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
fig=plt.figure()
plt.plot(epoch_count,loss_history,'b--')
plt.legend(["Training Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()   
#print(W2.data)

# %%

"""
Here we produce a sentence given an input word that is found in the dictionary, 
the model will select the next word that is more likely to be close to the input word
"""
"""
 Method1: Given first word obtain hotvector associated V,
        then we calculate output vector U=W2*W1*V obtain probabilities using 
        exponential and logsoftmax and  find the component of U with highest probability.
        that component represents the position of the most likely word to be close to the first word.
Method2: Given first word (word1) obtain hotvector associated V, obtain next word (word2)
        like in method 1 and obtain its associated hotvector. Then add the two hotvectors =HV
        use it to calculate output vector U=W2*W1*HV, repeat.
        
Method3: From first word (word1) get hotvector V1 and calculate embbeded vector U1=W1*V1 then calculate
        all embbeded vectors in the vocabulary that represent the context vectors Ui=W1*Vi.
        Calculate all inner products U1*Ui use softmax and exp to get all the probabilities, the highest
        represent the next vector.
        
Note: for Method1 and Method3 we are eliminating the possibility that the next word is the same first word.
        
        
"""
def sentence_production2(hotvector): # Method 2
        zz1=torch.matmul(W1,hotvector)
        zz2=torch.matmul(W2,zz1) # perform W2*W1*HV
        log_softmax=F.log_softmax(zz2,dim=0) # Compute Log softmax
        values2, indices2 = torch.max(torch.exp(log_softmax), 0) 
        next_word2=idx2word[indices2.item()] #Type 2 Sentece Generator
        return(next_word2,indices)

def sentence_production(first_word): # Method 1
        hotvector=get_input_layer(word2idx[first_word]) # extract index from dictionary and obtain 1 hot vector
        z1=torch.matmul(W1,hotvector)
        z2=torch.matmul(W2,z1) # Implement U = W2*W1*HV. Size of z2 = 15 (for small corpus), determined by print
#        print("-"*50)
#        print("Z2",z2)
#        print("x"*50)
#        print(z2.shape)
        # Find softmax of input vector. Normalizes values along axis 1
        log_softmax=F.log_softmax(z2,dim=0) # Input vector is equivalent z2 = Ucontext * Vcenter?
#        print("LOG",log_softmax)
        # multiplication with hotvector fixes dimensions. Becaues hotvector all but one element will be 0
        v_substract=torch.exp(log_softmax[word2idx[first_word]])*hotvector#Reduce/negate the prob of getting same word next. Expononential to eliminate neg vals
#        print(word2idx[first_word])
#        print(log_softmax[word2idx[first_word]])
#        print(torch.exp(log_softmax[word2idx[first_word]]))
#        print("v_sutract",v_substract)
        values, indices = torch.max(torch.exp(log_softmax)-v_substract, 0) #Take maximum softmaxtorch.tensor([[1., -1.], [1., -1.]]) 
#        print("Y"*50)
#        print(torch.exp(log_softmax))
#        print(torch.exp(log_softmax)-v_substract)
#        print("T"*50)
#        print("values",values)
        next_word1=idx2word[indices.item()] #Type 1 Sentence Generator 
        return(next_word1, indices)
        
embbeded_vectors=[torch.matmul(W1,get_input_layer(i)) for i in range(vocabulary_size)] # For each word in vocab, obtain one-hot vec and multiply with W1.

def sentence_production3(first_word):
        hotvector=get_input_layer(word2idx[first_word]) # obtain one-hot vector
        z1=torch.matmul(W1,hotvector) # W1 * HV
        inner_products=torch.Tensor([torch.dot(z1,u) for u in embbeded_vectors]) # dot product W1*HV with one-hot vector
        log_softmax2=F.log_softmax(inner_products,dim=0) #obtain softmax
        v_substract2=torch.exp(log_softmax2[word2idx[first_word]])*hotvector
        values3, indices3 = torch.max(torch.exp(log_softmax2)-v_substract2,0) # Negate repeated word
        next_word3=idx2word[indices3.item()] #Type 3 Sentence Generator
        return(next_word3, indices3)
 
fig1=plt.figure()
ax=Axes3D(fig1) 
x=[x[0].item() for x in embbeded_vectors]
y=[x[1].item() for x in embbeded_vectors]
z=[x[2].item() for x in embbeded_vectors]
ax.scatter(x,y,z)
plt.legend(["Projected embbeded vectors"])
plt.xlabel("x")
plt.ylabel("y")
plt.show()


size_of_sentence=7
first_word=" "
while first_word!="ESC":
    first_word=input("Please type a word , I will make a sentence or type ESC to exit \n")
    if first_word=="ESC": continue
    elif first_word not in word2idx:
        print("not in my dictionary, write another word: \n")
        continue 
    else:
        sentence=[]
        sentence2=[]
        sentence3=[]
        counter=0
        dummy_vector=torch.zeros(vocabulary_size).float()
        first_word2=first_word
        first_word3=first_word
        while counter<=size_of_sentence:
            
            #First type of generator
            sentence.append(first_word)
            first_word=sentence_production(first_word)[0]
            #Second type of generator
            sentence2.append(first_word2)
            dummy_vector=get_input_layer(word2idx[first_word2])+dummy_vector # add hot_vectors together to "remember" previous words
            first_word2=sentence_production2(dummy_vector)[0]
            #Third type of generator
            sentence3.append(first_word3)
            first_word3=sentence_production3(first_word3)[0]
            counter+=1
        print("Sentence with 1st generator is: \n ", *sentence)
        print("\nSentence with 2nd generator is: \n", *sentence2)
        print("\nSentence with 3rd generator is: \n", *sentence3)