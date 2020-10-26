
# Imports

import torch
from torchvision import datasets, models, transforms # All torchvision modules
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, Loss functions,..
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam,...
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (DataLoader,Dataset)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchtext # Makes it easy to work with sequence data 
from torchtext.data import get_tokenizer

import re # regex library
import os # Doing operating system operations
import cv2 # Computer vision tasks with OpenCV
import numpy as np # Powerful arrray computation library
from PIL import Image # WOrking with image files
import pandas # Extracting data from csv
import math # Math package
import pickle # Saving variables for later usage.

from torchsummary import summary # Make understanding of models easier
import torch # PyTorch library
from time import time # Using timer in code


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use Cuda if GPU available!


model_vgg = models.vgg16(pretrained=True)# obtain pretrained VGG16 model
model_vgg.classifier = nn.Sequential(*list(model_vgg.classifier.children())[:-2]) # remove last linear layer of VGG16 model

### Sequence to sequence model

class Encoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
                                    
        return h_n, c_n
    
class Decoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, number_of_words):
        super(Decoder_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x, h_n, c_n):
        output, _ = self.lstm(x.float(),(h_n,c_n))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        output = self.fc(output)                            
        
        return output
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X_1, X_2):
        h_n, c_n = self.encoder(X_1)
        output = self.decoder(X_2, h_n, c_n)
        return output