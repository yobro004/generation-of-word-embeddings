import torch
import os
from string import ascii_letters
import nltk
import numpy as np
import torch.nn as nn

def index_max(ip):
    index = 0
    for i in range(1, len(ip)):
        if ip[i] > ip[index]:
            index = i 
    return index


def make_cv(context, word2vec):
    idxs = [word2vec[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)



def prob_max(ip, vec2word):
    return vec2word[index_max(ip)]

#declarations
CONTEXT_SIZE = 2  #window size
EMDEDDING_DIM = 100

word2vec = {}
vec2word = {}
raw_text =open(r"C:\Users\ROHITH MANDA\Desktop\work\dataset.txt","r+")
raw_text=raw_text.read()
raw_text=raw_text.split()
#print(raw_text)

vocab = set(raw_text)
vocab_size = len(vocab)

for i, word in enumerate(vocab):
    word2vec[word] = i
    vec2word[i] = word

data = []
#format according to nn
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))


#nn
class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        #out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.linear1 = nn.Linear(embedding_dim, 128)

        self.activation_function1 = nn.ReLU()
        
        #out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)

        self.activation_function2 = nn.LogSoftmax(dim = -1)
        

    def forward(self, ips):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.LongTensor([word2vec[word]])
        return self.embeddings(word).view(1,-1)


model = CBOW(vocab_size, EMDEDDING_DIM)

loss_function = nn.NLLLoss()
opzer = torch.optim.SGD(model.parameters(), lr=0.001)#learning rate set 



for epoch in range(50):
    total_loss = 0
    for context, target in data:
        context_vector = make_cv(context, word2vec)  
        model.zero_grad()
        log_probs = model(context_vector)
        loss = loss_function(log_probs, torch.tensor([word2vec[target]], dtype=torch.long))
        loss.backward()
        opzer.step()

        total_loss += loss.data

#test cases::::::::::::::::::::::::::-------
#context = ['People','create','to', 'direct']
#context = ['they' ,'are' ,'composed' ,'from']
#context = ['programmers','must','to','understand']
#context = ['A','process','is', 'like']
#context = ['learning' ,'to' ,'is','considerably']
#context = ['One' ,'kind' ,'of' ,'expression']
#context = ['It' ,'can','intellectual', 'work.']
#context = ['If','you','Lisp','with']
#context =['One' ,'easy' ,'get', 'started']
context  =['about','to','the','idea']
#context = ['to' ,'the', 'collapse', 'of']
#context = ['he' ,'likes', 'going', 'there']
#context = ['We' ,'are', 'to', 'study']
#context = ['have' ,'three', 'for', 'India']
#context = ['tried' ,'to', 'our', 'way']
#context = ['as' ,'a', 'power', 'but']
        



context_vector = make_cv(context, word2vec)
a = model(context_vector).data.numpy()
print('Raw text: {}\n'.format(' '.join(raw_text)))
whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

answer = ''.join(filter(whitelist.__contains__, context))
print(context[0:10])        
print('Context: {}\n'.format(context))
#print (str)
print('Prediction: {}'.format(prob_max(a[0], vec2word)))
