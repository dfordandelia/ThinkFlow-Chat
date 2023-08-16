import numpy as np
import json
import torch
from model import myNeuralNet
from nltk_utils import bag_of_words, tokenize

with open('intents.json','r') as f:
    intents = json.load(f)

FILE = "chatbot_data.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags= data['tags']
model_state = data['model_state']

model = myNeuralNet(input_size,  hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot" #input("What would you like your bot's name to be: ")
print("Let's chat! I will assist you in your order. Type 'quit' to quit the chat")

while(True):
    sentence = input("You: ")
    if sentence=='quit': 
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _,predicted = torch.max(output,dim=0)
    
    pred_tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=-1)
    prob = probs[predicted.item()]

    if prob>0.8:
        for intent in intents['intents']:
            if pred_tag == intent['tag']:
                #print(f"Predicted as {pred_tag}")
                print(f"{bot_name}: {np.random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Sorry, I do not understand..")

    
