"""
Author: Aizar E. D. and Nicholas
Nicholas: Fixed and optimised loading of movie_corpus
Also inserted tokenise function with Class WordWindow Dataset
Edited make_windows function such that short sentences are NOT missed!
"""

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import string
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import random
#import training                                     # training library from AI_Utils (Harry's library)


# %%

#def small_corpus_tokeniser(corpus):
#    """Take in the list of sentences (lines as strings) and split those strings into a list of lowercase words"""
#    tokens = [x.translate(str.maketrans('', '', string.punctuation)).lower().split() for x in corpus]
#    tokens
#    return tokens
# %% Open file and read, explicit encoding to prevent encoding/decoding error

    #currentFile = open(filename, 'rt', encoding='latin1')
'''with open('movie_lines.txt','rt',encoding='latin1') as f: #closes file after all the lines have been processed
    #for line in f: #not using readlines(), as this consumes the memory
        #process(line)            tokenised_corpus = tokeniser(f)    

       ' movie_tokens = movie_corpus_tokeniser(f)    '''


# Will not use function below

#  Clean text 

def clean_text(line):
    # text = text.lower() NOT needed!
    ### IMPLEMENT OTHER CLEANING FUNCTIONALITY (PUNCTUATION)
    
    uncontract = {"i'm":"i am","i'll":"i will","i'd":"i would","i've":"i have",
                  "you're":"you are","you'll":"you will","you'd":"you would","you've":"you have",
                  "he's":"he is","he'll":"he will","he'd":"he would",#"he's":"he has",
                  "she's":"she is","she'll":"she will","she'd":"she would",#"she's":"she has",
                  "it's":"it is","it'll":"it will","it'd":"it would",#"it's":"it has",
                  "we're":"we are","we'll":"we will","we'd":"we would","we've":"we have",
                  "they're":"they are","they'll":"they will","they'd":"they would","they've":"they have",
                  "that's":"that is","that'll":"that will","that'd":"that would","that's":"that has",
                  "who's":"who is","who'll":"who will","who'd":"who would",#"who's":"who has",
                  "what's":"what is","what're":"what are","what'll":"what will","what'd":"what would",
                  "where's":"where is","where'll":"where will","where'd":"where would",
                  "when's":"when is","when'll":"when will","when'd":"when would",
                  "why's":"why is","why'll":"why will","why'd":"why would",
                  "how's":"how is","how'll":"how will","how'd":"how would",
                  "can't":"cannot","don't":"do not"} 
   
    for z in uncontract:
        if z in line:
            #print("suuuuuuuuuuuup")
            line=line.replace(z,uncontract.get(z)) # not equivalent to x=z
            
    #newline.append(line)
    #print("linessssssssssssssss",line)
    return line


# Function to tokenise corpus
def movie_corpus_tokeniser(corpus):
    ### IMPLEMENT
    tokens = []
    #counter = 0
    for x in corpus:
        
        listed = x.split(' +++$+++ ')
        sentence = listed[-1].lower()
        #print("sentenceeeeeeeeeeeeeeee",sentence)
        sentence = clean_text(sentence)
        #print("sentenceeeeeeeeeeeeeeee",sentence)
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
        #print("sentenceeeeeeeeeeeeeeee",sentence)
        words = sentence.split()
        #print(words)
        tokens.append(words)
        
        #if counter == 3767: Do not break
           #break
        #counter +=1

    return tokens
# %%

def make_windows(text, window_size=5):
    """Takes in a list of tokenised messages and returns all the windows"""
    windows = []
    for msg_idx, line in enumerate(text):                       # count through each line of text
        print()
        print('Message index:', msg_idx)
        print('Message:', line)
        if len(line)<5:
            window_size = 2
        else:
            window_size = 5
        for idx in range(len(line) - window_size + 1):          # slide a window along the line until it reaches the end
            #print("In the loop\n")
            window = line[idx: idx + window_size]               # get the words that fall into that window as a list
            print('Window idx:', idx, '\twindow:', window)
            windows.append(window)                              # add that list of tokens in a window to the list of windows
    return windows

def unloader(data, dict):
    # data is the output softmax vector
    y_hat = data.detach()                                       # detach output vector from computational graph
    y_hat = y_hat.numpy()                                       # turn torch tensor into numpy
    y_hat = np.argmax(y_hat)                                    # get idx of most likely predicted word
    y_hat = dict[y_hat]                                         # use idx2word to get a word
    print(y_hat)
    return y_hat

def loader(word, dict):
    word = dict[word]                                           # use word2idx to convert word into index
    return word

# BUILD DATASET
class WordWindowDataset(Dataset):
    def __init__(self, text_file='small_corpus.txt', tokeniser=movie_corpus_tokeniser, loader=loader, unloader=unloader, transform=None):
        self.loader = loader
        self.unloader = unloader
        self.transform = transform
        
        #currentFile = open(filename, 'rt', encoding='latin1')
        with open(text_file,'rt',encoding='latin1') as f: #closes file after all the lines have been processed
        #for line in f: #not using readlines(), as this consumes the memory
        #process(line)
            tokenised_corpus = tokeniser(f)    

#        # put tokeniser in here and use it to convert raw txt into windows
#        with open(text_file, 'r') as f:                         # open the file whilst you read it's contents
#            lines = f.read()                                    # read the file
#        
#        #print(lines)
        #lines = clean_text(tokenised_corpus)                    # clean the text
        #print(tokenised_corpus)
#        lines = lines.splitlines()                              # split the lines of the text
#        #print(type(lines))
#        #print(lines)
#        tokens = tokeniser(lines)                               # convert text into tokens
#        #print(tokens)
        vocab = []
        print(tokenised_corpus)
        zxcxzc
        #vocab = [word for message in lines for word in message.split()]         # flatten out corpus into list of words
        for sentence in tokenised_corpus:
            for words in sentence:
                vocab.append(words)
        
        vocab = set(vocab)  # remove duplicates
        print(vocab)                                                     
        print('Length of vocab:', len(vocab))
        self.len_vocab = len(vocab)

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}           # map words to indexes
        self.idx2word= {v: k for k, v in self.word2idx.items()}                 # make dict to map the inverse

        #MAKE WINDOWS
        print('-'*70)
        self.windows = make_windows(tokenised_corpus)                                     # make windows from the tokenised lines
        print('-'*70)
        print('Number of windows:', len(self.windows))


    def __getitem__(self, idx):
        window = self.windows[idx]
        print(self.windows)
        print('Window:', window)
        centre_idx = len(window) // 2                                           # get centre idx of window
        centre_word = window[centre_idx]                                        # get the centre word using that idx
        context_word = random.choice(window[:centre_idx] + window[centre_idx + 1:])     # choose a random word from the context
        print('Centre word:', centre_word)
        print('Context word:', context_word)
        #if self.transform:
        #    centre_word, context_word = self.transform(centre_word, context_word)       # convert from word to idx

        # load the words into torch.tensor form using the loader
        centre_word = self.loader(centre_word, self.word2idx)
        context_word = self.loader(context_word, self.word2idx)

        return centre_word, context_word

    def __len__(self):
        return len(self.windows)
        
text_file = 'movie_lines.txt'
#text_file = 'small_corpus.txt'

dataset = WordWindowDataset(text_file)          # instantiate dataset

# %%
vocab_size = dataset.len_vocab                  # get length of dataset

#for item in dataset:
#    print(item)             # show an example
#    break

# %%

# HYPERPARAMETERS
embedding_dims = 50                              # dimensionality of word embeddings
lr = 0.001                                      # learning rate
batch_size = 16
# %%
# CREATE DATALOADER
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)          # to shuffle and batch the examples

# %%

### Tokenize movie text
##################################################################################################################################

#def tokenize_movie_corpus(corpus2):
#    tokens2=[]
#    counter = 0
#    for x in corpus2:
#        s = x.split(" +++$+++ ")
#        part_S = s[-1]
#        part_S = part_S.translate(str.maketrans('', '', string.punctuation)).lower()
#        print("Part_S:", part_S)
#        separate = part_S.split()
#        #separate = separate.split("\n")
#        ##print('separate words:',separate)
#        tokens2.append(separate)
#        counter += 1
#        if counter >= 1000:
#            break
#    return tokens2

#################################################################################################################################
'''
with open("movie_lines.txt", "r") as f:
    tokenized_corpus2=tokenize_movie_corpus(f)

tokenized_corpus=tokenized_corpus2
#
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
# %%
# MAKE MODEL
class SG_NN(torch.nn.Module):                                       # create class and inherit from torch.nn.Module
    def __init__(self):                                             # function run on instantiation
        super().__init__()                                          # instantiate parent class
        self.layers = torch.nn.Sequential(                          # a set of layers that are applied sequentially
            torch.nn.Embedding(vocab_size, embedding_dims),         # embedding layer (word representation lookup table. Takes in an index and returns the embedding of the word with that index.
            torch.nn.Linear(embedding_dims, vocab_size),            # linear transformation
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
    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):      # for each batch in the dataloader
            x, y = batch                                    # unpack the batch
            print(x)
            print(y)
            prediction = model(x)                           # forward pass
            #print('Input shape:', x.shape)
            #print('Prediction shape:', prediction.shape)
            loss = criterion(prediction, y)                 # compute the loss
            optimiser.zero_grad()                           # zero gradients so they dont accumulate
            loss.backward()                                 # backward pass (compute gradients in all tensors contributing to loss)
            optimiser.step()                                # update the parameters that we told the optimiser to when we initialised it
            print('Epoch:', epoch, '\tBatch:', batch_idx, '\tLoss:', loss.item())
            train_losses.append(loss.item())
            loss_ax.plot(train_losses, 'b')
            loss_fig.canvas.draw()                          # update plot

            if batch_idx == 10:
                #break
                pass
    model.eval()

train(mymodel, 100, data_loader)

# GENERATE
def generate():
    ### IMPLEMENT
    pass

plt.ion()

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
        
train(2, data_in)

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
