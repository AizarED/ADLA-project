# -*- coding: utf-8 -*-
"""
Author: Aizar E. D. and Nicholas
Updates: 
1) using the skip gram model for generation. Now it uses the "forward" function 
that takes a word index and returns a logsoftmax vector
2)Uses dataset.unloader(logsoftmax_vector) in the generation to get the most likely word from the models output
3)Added <EOL> token to each line and implemented flow control (While statement)
in generation to make variable length sentences
4)Tested time of computation on various sections using Timer() class  from Harry's code 
instantiated as: TimeCop function :P
5)Modifed  function _getitem_ from the class WordWindowDataset: The selection of context words now is chosen from 
 the elements "after" the chosen central word, having as a result only forward probabilities in the soft max vector instead of a mix between forward and back probabilities
this could avoid sentences like: "I am father your" instead of the correct: "I am your father"
6) Set to False the Bias from Linear layer.
7) Add clean_words() function to separate words like: i'm= i am, don't= do not

"""

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import string
#from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import random
import training                                     # training library from AI_Utils (Harry's library)
from timer import Timer                             # Timer library from AI_Utils (Harry's library)

TimeCop=Timer() #To test computing time
def small_corpus_tokeniser(corpus):
    """Take in the list of sentences (lines as strings) and split those strings into a list of lowercase words"""
    tokens = [(x.lower()+" <EOL>").split() for x in corpus]
    return tokens
#[x.lower().split() for x in f]

def movie_corpus_tokeniser(corpus):
    ### IMPLEMENT
    #pass
    tokens=[(x.split(" +++$+++ ")[-1].translate(str.maketrans('','',string.punctuation)).lower()+" <EOL>").split() for i,x in enumerate(corpus,1) if i<133]
    return tokens

"""
#This section is used for testing function movie_corpus_tokeniser.
Nick version  uses for loop, append method, and f object straight into the function,
I  use comprehension list,f.read() and splitlines(). 
#to create the separated sentences containing words separated.

text_file='movie_lines_test.txt'
with open(text_file, 'r') as f:                         # open the file whilst you read it's contents
#with open('movie_lines_test.txt','rt',encoding='latin1') as f:
        lines = f.read()                                    # read the file its a class string 
        #print(type(lines))
        type(lines.splitlines())
        lines = lines.splitlines()                          #its a class list, in N version this is done by append() method
        print(lines)
        print(type(lines))

dic = {"BIANCA":"GIRL", "CAMERON":"BOY", "They":"THEY" , "holA": "HOLA"}
#[x.replace(z,dic.get(z))  for x in lines.splitlines() for z in dic if z in x ] #

newline=[]
for x in lines:
    for z in dic:
        #if z in x:
            x=x.replace(z,dic.get(z)) 
    newline.append(x.replace(z,dic.get(z)))      
print(newline)  
   
[x for x in lines.splitlines() if "BIANCA" in x ]
        movie_tokens=movie_corpus_tokeniser(f)             #Nick version: f straing into tokeniser
print('movie_tokens: ', movie_tokens)
print("tokenised", movie_corpus_tokeniser(lines))         #Aiz version lines is preprocessed with splitlines and readfile()
print("tokenised", movie_corpus_tokeniser(newline))


"""
def clean_words(lines_splitlines):
   dic = { "she's":"she is",  "i'm": "i am", "you’re": "you are",  "you’d": "you would","they're":"they are", 
   "you're":"you are", "don't":"do not", "i've":"i have", "she'd": "she would", 
   "workin'": "working", "goin' ": "going", "where've": "where have"}
   newline=[]
   for x in lines_splitlines:
      for z in dic:
        if z in x:
            x=x.replace(z,dic.get(z)) 
      newline.append(x.replace(z,dic.get(z)))      
   return(newline)    

def clean_text(text):
    text = text.lower()
    ### IMPLEMENT OTHER  CLEANING FUNCTIONALITY (PUNCTUATION)
    return text

# should we have words made from tokenised text (list of tokenised lines) or from raw text (e.g. open(file).read())?

def make_windows(text, window_size=3):
    """Takes in a list of tokenised messages and returns all the windows of the specified size"""
    windows = []
    for msg_idx, line in enumerate(text):                       # count through each line of text
        print()
        print('Message index:', msg_idx)
        print('Message:', line)
        for idx in range(len(line) - window_size + 1):          # slide a window along the line until it reaches the end
            window = line[idx: idx + window_size]               # get the words that fall into that window as a list
            print('Window idx:', idx, '\twindow:', window)
            windows.append(window)                              # add that list of tokens in a window to the list of windows
    print("All windows:", windows)
    return windows
#print(make_windows(movie_corpus_tokeniser(lines))) 
'''
Aizar observation: 
#all windows are of size: window_size. That means that you are missing out cases for sentences smaller 
than windowsize. ex: line= "I am" "she okay" "lets go" is lost for case of window_size=3 worse for 
bigger windows sizes.
'''

#testv=output_softmax_vector1("they",dataset)
#unloader(testv,dataset.idx2word)

def unloader(data, dict):
    # data is the output softmax vector
    y_hat = data.detach()                                       # detach output vector from computational graph
    y_hat = y_hat.numpy()                                       # turn torch tensor into numpy
    y_hat = np.argmax(y_hat)                                    # get idx of most likely predicted word
    y_hat = dict[y_hat]                                         # use idx2word to get a word
    #print(y_hat)
    return y_hat

def loader(word, dict):
    word = dict[word]                                           # use word2idx to convert word into index
    return word

# BUILD DATASET
class WordWindowDataset(Dataset):
    def __init__(self, text_file='small_corpus.txt', tokeniser=small_corpus_tokeniser, loader=loader, unloader=unloader, transform=None):
        self.loader = loader
        self.unloader = unloader
        self.transform = transform
 
        # put tokeniser in here and use it to convert raw txt into windows
        with open(text_file, 'rt', encoding="latin1") as f:                         # open the file whilst you read it's contents
            lines = f.read()                                    # read the file
        #function to read first N lines: ? 
        #lines=lines.readonly(N)
        #print(lines)
        lines = clean_text(lines)                               # clean the text
        #print(lines)
        lines = lines.splitlines()                              # split the lines of the text
        #print(type(lines))
        #print(lines)
        lines=clean_words(lines)
        TimeCop.start("tokeniser")
        tokens = tokeniser(lines)                               # convert text into tokens
        TimeCop.stop("tokeniser")
        print(tokens)

        vocab = [word for line in tokens for word in line]         # flatten out corpus into list of words
        vocab = set(vocab)                                                      # remove duplicates
        print("vocabulary", vocab)
        print('Length of vocab:', len(vocab))
        self.len_vocab = len(vocab)

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}           # map words to indexes
        self.idx2word= {v: k for k, v in self.word2idx.items()}                 # make dict to map the inverse

        #MAKE WINDOWS
        TimeCop.start("Make_Windows")
        self.windows = make_windows(tokens)                                     # make windows from the tokenised lines
        print('Number of windows:', len(self.windows))
        TimeCop.stop("Make_Windows")

    def __getitem__(self, idx):
        window = self.windows[idx]
        #print(self.windows)
        ####print('Window:', window)
        #original# centre_idx = len(window) // 2                                           # get centre idx of window
        #original# centre_word = window[centre_idx]                                        # get the centre word using that idx
        #original# context_word = random.choice(window[:centre_idx] + window[centre_idx + 1:])     # choose a random word from the context
        
        #######Improvement to pick only fwd words Aizar
        #centre_word=window[0]
        #context_word=random.choice(window[1:])
        
        ######attempt to pick random pairs with forward probabilities.Aizar, look if loss gets better/worse
        centre_idx=random.choice(range(len(window)-1))
        centre_word=window[centre_idx]
        context_word=random.choice(window[centre_idx+1:])
        
        #attempt incomplete#
        #for centre_idx in range(len(window)-1):
         #   centre_word=window[centre_idx]
          #  context_word=random.choice(window[centre_idx+1:])
            
        ####print('Centre word:', centre_word)
        ####print('Context word:', context_word)
        #if self.transform:
        #    centre_word, context_word = self.transform(centre_word, context_word)       # convert from word to idx

        # load the words into torch.tensor form using the loader
        ####print("types before: ", type(centre_word), type(context_word))  #types before:  <class 'str'> <class 'str'>
        centre_word = self.loader(centre_word, self.word2idx)
        context_word = self.loader(context_word, self.word2idx)
        ####print("types: ", type(centre_word), type(context_word))   #types:  <class 'int'> <class 'int'>

        return centre_word, context_word

    def __len__(self):
        return len(self.windows)


"""

window_size = 1# 4 #2
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
  

"""

text_file3 = 'movie_lines.txt'                       ### IMPLEMENT (we could write a helper function to format the movie dset like the simple one)
text_file = 'small_corpus.txt'
#text_file2='movie_lines_test.txt'  
tokeniser2=movie_corpus_tokeniser
tokeniser=small_corpus_tokeniser

dataset = WordWindowDataset(text_file3, tokeniser2)          # instantiate dataset
vocab_size = dataset.len_vocab                  # get length of dataset
#print(vocab_size, len(dataset))
for item in dataset:
    print(item)             # show an example
    break

# HYPERPARAMETERS
embedding_dims = 50                              # dimensionality of word embeddings
lr = 0.001                                      # learning rate
batch_size = 16

# CREATE DATALOADER
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)          # to shuffle and batch the examples
#Create sample: 
'''
for batch_idx, batch in enumerate(data_loader):      # for each batch in the dataloader
            x, y = batch                                    # unpack the batch
            print(batch)
            print(x, type(x))
            print(y)
            break
mymodel.forward(x)

z=torch.LongTensor([32])
z2=torch.LongTensor([24, 10,  7, 14, 24, 19, 12,  1,  1, 24, 28,  6,  3,  9, 26, 16])
print(type(z2))
mymodel.forward(z)
NOTE: check if each element of the pair x,y is one of the pairs created for centralword/contextword i.e.
xi,yi belong to the total set of pairs created.
NOTE: x,y=batch separates the  pairs central word, context word  ( xi,yi) in to vectors x,y each element xi,yi correspos
to the index of the central/contex words.
'''

### Tokenize movie text

def tokenize_movie_corpus(corpus2):
    tokens2 = []
    counter = 0
    for x in corpus2:
        s = x.split(" +++$+++ ")
        part_S = s[-1]
        part_S = part_S.translate(str.maketrans('', '', string.punctuation)).lower()
        print("Dialogue:", part_S)
        separate = part_S.split()
        #separate = separate.split("\n")
        ##print('separate words:',separate)
        tokens2.append(separate)
        counter += 1
        if counter >= 1000:
            break
    return tokens2

'''
with open("movie_lines.txt", "r") as f:
    tokenized_corpus2=tokenize_movie_corpus(f)

tokenized_corpus=tokenized_corpus2
'''
'''
# HOW TORCH MODULES WORK - by the Berg
import torch.nn.functional as F
# each nn.layer is a class like the following, 
# and it has a __call__ function so that when we write mylayer(x) it runs the __call__ function

class LinearLayer():
    def __init__(self, input_units, output_units):
        self.weights = torch.randn(output_units, input_units)
        
    def __call__(input):
        return torch.matmul(self.weights, input)
        

class LogSoftmax():

    def __call__(self, input):
        o = F.softmax(torch.tensor(input).double())
        return o.item()
        
class Sequential():
    def __init__(self, list_of_layers):
        self.lol = list_of_layers
        
    def __call__(self, input):
        for layer in list_or_layers:        # e.g layer = torch.nn.Linear(input_units, output_units)
            input = layer(input)            # input goes through each layer which is a callable torch module like Linear or Embedding or LogSoftmax
        return input
            

s = LogSoftmax()     # instantiate class
print(s(10))    # call function is defined by __call__, so we can
'''

# MAKE MODEL
class SG_NN(torch.nn.Module):                                       # create class and inherit from torch.nn.Module
    def __init__(self):                                             # function run on instantiation
        super().__init__()                                          # instantiate parent class
        self.layers = torch.nn.Sequential(                          # a set of layers that are applied sequentially
            torch.nn.Embedding(vocab_size, embedding_dims),         # embedding layer (word representation lookup table. Takes in an index and returns the embedding of the word with that index.
            torch.nn.Linear(embedding_dims, vocab_size, False),            # linear transformation
            torch.nn.LogSoftmax()                                   # softmax to make a distribution, log to stailise and make faster (logsoftmax and softmax are maximised by the same thing)
        )

    def forward(self,x):                # the forward pass
        x = self.layers(x)              # pass input, x, through the sequential layers
        return x

# %% Instantiate model and optimizer
    # data
    # model
    # optimize
    # loss

mymodel = SG_NN()                                               # instantiate skip gram model
optimiser = torch.optim.Adam(mymodel.parameters(),lr=lr)        # create optimiser
criterion = torch.nn.NLLLoss()                                  # create instance of NLLLoss criterion (it's callable)

# TRAINING CODE
epochs = 1

loss_fig, loss_ax = training.getLossPlot()                      # get a plot to show the loss from a library i wrote
train_losses = []

def train(model, epochs, dataloader):
    #print("what is this", model.train())
    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):      # for each batch in the dataloader
            x, y = batch                                    # unpack the batch
            ####print(x)
            ####print(y)
            TimeCop.start("train_prediction")
            prediction = model(x)                            # forward pass
            TimeCop.stop("train_prediction")
            #print(prediction, prediction.size())       #NAZ suspected prediction output but why change dimensions??
            #print('Input shape:', x.shape)
            #print('Prediction shape:', prediction.shape)
            loss = criterion(prediction, y)                 # compute the loss
            optimiser.zero_grad()                           # zero gradients so they dont accumulate
            loss.backward()                                 # backward pass (compute gradients in all tensors contributing to loss)
            optimiser.step()                                # update the parameters that we told the optimiser to when we initialised it
            if epoch % 10 == 0: 
                print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
            train_losses.append(loss.item())
            loss_ax.plot(train_losses, 'b')
            loss_fig.canvas.draw()                          # update plot

            if batch_idx == 10:
                #break
                pass
    model.eval()
    #print("Preprint parameters in training: \n" , list(model.parameters()))
    #return(prediction)
#output_y=
TimeCop.start("Exec_train")
train(mymodel, 100, data_loader) #it is 100 not 4
TimeCop.stop("Exec_train")
#print('Predicted vector',output_y)
#print("Predicted vector size: ", output_y.size())

#SAVING MODEL
#torch.save(mymodel.state_dict(), "/Users/aizar/Documents/NLP/skip_gram_v2.pth")

#LOADING MODEL
"""
mymodel1 = SG_NN(*args, **kwargs)
mymodel1.load_state_dict(torch.load("/Users/aizar/Documents/NLP/skip_gram_v2.pth"))
mymodel1.eval()

checkpoint = torch.load("/Users/aizar/Documents/NLP/skip_gram_v2.pth")
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['mymodel']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

mymodel = load_checkpoint('/Users/aizar/Documents/NLP/skip_gram_v2.pth')
"""

'''
#Exploring how to obtain the parameters of the model
#print("Is this what you are looking for? ", list(mymodel.parameters()))
#for param in mymodel.parameters(): 
#    print("List of parameters are: " ,type(param.data), param.size(), param.data)
#for param2 in mymodel.layers.parameters():
#    print("alternative \n", type(param2.data), param2.data)
#print( mymodel.layers[0], list(mymodel.layers[0].parameters())[0].detach(), mymodel.layers[0].parameters())
'''
W1=list(mymodel.layers[0].parameters())[0].detach()
#shape(W1)
W2=list(mymodel.layers[1].parameters())[0].detach()
if len(list(mymodel.layers[1].parameters()))<2:
    Bias=torch.zeros(dataset.len_vocab )
else:     
    Bias=list(mymodel.layers[1].parameters())[1].detach()
#shape(W2)

#output_softmax_vector("i")
def get_input_layer(word_idx):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x
#get_input_layer(1)    
def output_softmax_vector(word):
    hotvector=get_input_layer(dataset.word2idx[word])
    Embebbed_vector=torch.matmul(hotvector,W1)
    Output_vector=torch.matmul(W2, Embebbed_vector.view(-1,1))+Bias.view(-1,1)
    y = F.log_softmax(Output_vector, dim=0) #exp/sum of exp of each element of z2
    y=torch.exp(y)  #obtain probability distribution
    return(y)
def output_softmax_vector1(word, dataset):
    word_index=torch.LongTensor([dataset.word2idx[word]]) #get input index from word
    y=mymodel.forward(word_index)
    y=torch.exp(y)
    return(y)
#output_softmax_vector1("do", dataset)



"""
word="learn" 
word="they" 
print(word, dataset.word2idx[word])
Yout=output_softmax_vector(word)
print(Yout)
print(torch.exp(Yout), torch.exp(Yout)[ dataset.word2idx[word]])
print(sentence_production(word))
hotvector=get_input_layer(dataset.word2idx[word])
print(mymodel.forward(hotvector))
F.log_softmax(torch.matmul(W2,W1.t()),dim=0)
torch.matmul(W2,W1.t())
torch.matmul(W1,W2.t())
EmV=torch.matmul(get_input_layer(dataset.word2idx[word]),W1)
print(EmV)
EmV2=torch.Tensor([  0.1859, -0.6175, -0.1403 ])
matmul(EmV,W2.t())
matmul(W2,EmV.view(-1,1))
vs
zz=torch.LongTensor([dataset.word2idx[word]])
print(mymodel.forward(zz))
z=torch.LongTensor([32])
z2=torch.LongTensor([24, 10,  7, 14, 24, 19, 12,  1,  1, 24, 28,  6,  3,  9, 26, 16])
print(type(z2))
mymodel.forward(z)

torch.nn.Embedding()
mymodel.layers[0](zz)
mymodel.layers[0:2](zz)
mymodel.layers[0:3](zz)
mymodel.forward(zz)
F.log_softmax(mymodel.layers[0:2](zz), dim=1)

mymodel.layers[1](mymodel.layers[0](zz))

torch.matmul(mymodel.layers[0](zz),W2.t())
a=mymodel.layers[1](torch.Tensor([1,0,0])).detach().view(1,4)
b=mymodel.layers[1](torch.Tensor([0,1,0])).detach().view(1,4)
c=mymodel.layers[1](torch.Tensor([0,0,1])).detach().view(1,4)
d=torch.cat((a,b,c),0)
torch.matmul(mymodel.layers[0](zz),d)
torch.matmul(torch.Tensor([1,0,0]),d)
shape(c)
shape(mymodel.layers[0](zz))
shape(torch.LongTensor([[1,0,0]]))
"""

"""
A=torch.matmul(W1,W2)
print(A)

print(vector_word)
B=torch.mm(vector_word,W1)
print(list(W1) )
print(type(vector_word))

#Given a word:
word="they" 
print(word, dataset.word2idx[word])  #print the word and the asociated index.
Yout=output_softmax_vector(word)
print(Yout)
#Embebbed vector from W1*hotvector
hotvector=get_input_layer(dataset.word2idx[word]) #From word index get hotvector
shape(hotvector)
shape(W1)
Embebbed_vector=torch.matmul(hotvector,W1)
shape(Embebbed_vector)
shape(W2)
Output_vector=torch.matmul(W2,Embebbed_vector )+Bias
F.log_softmax(Output_vector)
F.log_softmax(mymodel.layers[1](L1), dim=1)
#For same input word:
zz=torch.LongTensor([dataset.word2idx[word]]) #get input index from word
list(mymodel.parameters())
L1=mymodel.layers[0](zz)
L2=mymodel.layers[1](L1)
L3=mymodel.layers[2](L2)
mymodel.forward(zz)

#Reverse engineering matrix W2: apply e1,e2,e3 in R^3
mymodel.layers[1]()


test1=mymodel.layers[1](torch.Tensor([[0.0237,-0.9448,-1.6834]])).detach().view(1,4) #reproduces same as L2
test2=mymodel.layers[1](0.0237*torch.Tensor([1,0,0])-.9448*torch.Tensor([0,1,0])-1.6834*torch.Tensor([0,0,1]))
a=mymodel.layers[1](torch.Tensor([1,0,0])).detach().view(1,4)
b=mymodel.layers[1](torch.Tensor([0,1,0])).detach().view(1,4)
c=mymodel.layers[1](torch.Tensor([0,0,1])).detach().view(1,4)
d=torch.cat((a,b,c),0)
torch.matmul(L1,d-Bias)+Bias                #This should be the same as L2
torch.matmul(torch.Tensor([1,0,0]),d)
shape(0.0237*a-.9448*b-1.6834*c+Bias-(0.0237-.9448-1.6834)*Bias)
shape(test2)
mymodel.layers[0:2](torch.LongTensor([0]))
10*a
d-Bias
"""
TimeCop.show()
# GENERATE
def generate(firt_word, dataset):             #this is the same as sentence production0
    y=output_softmax_vector1(first_word, dataset)
    next_word=unloader(y,dataset.idx2word) 
    return(next_word)
    ### IMPLEMENT
    pass

plt.ion()
#first_word="i"
def sentence_production(first_word):
        hotvector=get_input_layer(dataset.word2idx[first_word]).view(-1,1)
        #z1=torch.matmul(W1,hotvector)
        #z2=torch.matmul(W2,z1)
        #log_softmax=F.log_softmax(z2,dim=0)
        y=output_softmax_vector(first_word)
        #print(torch.mul(torch.exp(y),hotvector))
        v_substract=torch.mul(torch.exp(y),hotvector)#REduce the prob of getting same word next
        values, indices = torch.max(torch.exp(y)-v_substract, 0)
        next_word1=dataset.idx2word[indices.item()] #Type 1 Sentence Generator 
        return(next_word1, indices.item())
def sentence_production0(first_word, dataset): 
         y=output_softmax_vector1(first_word, dataset)
         next_word=unloader(y,dataset.idx2word) 
         return(next_word)
#sentence_production0("they", dataset)      
def sentence_production2(first_word):
        hotvector=get_input_layer(dataset.word2idx[first_word]).view(-1,1)
        #z1=torch.matmul(W1,hotvector)
        #z2=torch.matmul(W2,z1)
        #log_softmax=F.log_softmax(z2,dim=0)
        y=output_softmax_vector(first_word)
        #print(torch.mul(torch.exp(y),hotvector))
        v_substract=torch.mul(torch.exp(y),hotvector)#REduce the prob of getting same word next
        #values, indices = torch.max(torch.exp(y)-v_substract, 0)
        set_likely_words=5
        indices=random.choice(torch.topk(torch.exp(y)-v_substract,set_likely_words,0)[1])
        next_word1=dataset.idx2word[indices.item()] #Type 2 Sentence Generator 
        return(next_word1, indices.item())
def sentence_production3(first_word, dataset):
    y=output_softmax_vector1(first_word, dataset)
    set_likely_words=4
    indices=random.choice(torch.topk(y,set_likely_words,0)[1])
    next_word1=dataset.idx2word[indices.item()] #Type 2 Sentence Generator 
    return(next_word1, indices.item())
    
#sentence_production("i")
#sentence_production2("she")[0]
#torch.topk(torch.exp(y)-v_substract, 4)
#sentence_production("she")[0]

size_of_sentence=27
first_word=" "
while first_word!="ESC":
    first_word=input("Please type a word , I will make a sentence or type ESC to exit \n")
    if first_word=="ESC": continue
    elif first_word not in dataset.word2idx:
        print("not in my dictionary, write another word: \n")
        continue 
    else:
        sentence=[]
        sentence2=[]
        #sentence3=[]
        counter=0
        counter2=0
        #dummy_vector=torch.zeros(vocabulary_size).float()
        first_word2=first_word
        sentence.append(first_word)
        sentence2.append(first_word2)
        #first_word3=first_word
   # while counter<=size_of_sentence:
        while first_word!="<EOL>" and counter<=size_of_sentence: 
                #First type of generator
                ####sentence.append(first_word)
                first_word=sentence_production0(first_word, dataset)
               # print(first_word)
                sentence.append(first_word)
                counter+=1
            #else: break
        while first_word2!="<EOL>" and counter2<=size_of_sentence:
                #Second type of generator
                ####sentence2.append(first_word2)
                first_word2=sentence_production2(first_word2)[0]
               # print(first_word2)
                sentence2.append(first_word2)
                counter2+=1
           # else: break
                
    print("Sentence with 1st generator is: \n ", *sentence)
    print("\nSentence with 2nd generator is: \n", *sentence2)
        

"""           
            #Second type of generator
            sentence2.append(first_word2)
            dummy_vector=get_input_layer(word2idx[first_word2])+dummy_vector
            first_word2=sentence_production2(dummy_vector)[0]
            #Third type of generator
            sentence3.append(first_word3)
            first_word3=sentence_production3(first_word3)[0]
            
"""
       # print("\nSentence with 2nd generator is: \n", *sentence2)
       # print("\nSentence with 3rd generator is: \n", *sentence3)


'''
#Nick attempt to model: 

# %% Define training loop

epochs = 1

def train(epochs, ):
    plt.close()
    mymodel.train() # put model into training mode
    
    for e in range(epochs):
        loss_val = 0
        for data, target in idx_pairs:
            x = get_input_layer(data).float()
            h = mymodel(x) # hypothesis, i.e. get log_softmax
            
            y_true = torch.from_numpy(np.array([target])).long()
            print(h.shape)
            print(target.shape)
            loss = criterion(h, target) #tensor
            
            loss_val+=loss.item() #value inside tensor
            
            loss_val.backward()
            optimizer.step() # step????
            optimizer.zero_grad()
            print(loss)
        
#train(2, data_in)
'''
"""
#Our origianl model: 
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
#Our original model
"""
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
"""
