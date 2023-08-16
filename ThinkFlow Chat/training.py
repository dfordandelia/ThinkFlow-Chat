# Creating the bot here
# What kind of bot?

import json
import nltk_utils 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import myNeuralNet



with open('intents.json','r') as f:
    intents = json.load(f)

intents = intents['intents']

all_words = []
tags = []
Xy = []

for intent in intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk_utils.tokenize(pattern)
        all_words.extend(w)
        Xy.append((w,tag))

ignore_words = ['?','!','.',',','-'] # unecessary characters

#print(all_words)

all_words = [nltk_utils.stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
#print(tags)

X_train = [] # bag of words here
y_train = [] # labels here

for (pattern_sentence, tag) in Xy:
    bag = nltk_utils.bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)

    labels = tags.index(tag)
    y_train.append(labels) # One hot encoded vector?

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.X_data = X_train
        self.y_data = y_train
    
    def __getitem__(self,index):
        return (self.X_data[index],self.y_data[index])
    
    def __len__(self):
        return self.n_samples

#Hyperparameters
batch_size = 6
hidden_size = 16
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 1e-3
num_epochs = 500

print(input_size,len(all_words))
print(output_size),len(tags)
dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)

model = myNeuralNet(input_size,  hidden_size, output_size) 

lossfn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr = learning_rate)

# The training

train_loss = []

for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words#.to(device)
        labels = labels#.to(device)

        # Forward pass
        y_pred = model(words)
        loss = lossfn(y_pred,labels.long())

        # Backward and opitimization pass
        optim.zero_grad()
        loss.backward()
        optim.step()
    train_loss.append(loss.item())
    if (epoch+1)%20==0:
        print(f'epoch number {epoch+1}/{num_epochs}, loss = {loss.item()}')
        

print(f'final loss is {loss.item()}')

plt.plot(np.arange(num_epochs),np.array(train_loss))
plt.show()


data = {
    'model_state' : model.state_dict(),
    'input_size' : input_size,
    'output_size' : output_size,
    'hidden_size' : hidden_size,
    'all_words' : all_words,
    'tags' : tags
}

FILE = 'chatbot_data.pth'
torch.save(data,FILE)

print(f'Training complete and file saved to {FILE}')






