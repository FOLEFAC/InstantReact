
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


class TextProcessor:
    '''
    This class contains methods which help in processing text data
    Args: 
        freq_threshold: Get the maximum frequency above which a word is not considered to be part of our vocabulary
        VOCAB_SIZE: the vocabulary size

    '''
    
    def __init__(self, freq_threshold = None, VOCAB_SIZE = None):
        
        self.word_to_index = {"<unk>":0, "<pad>":1, "<bos>": 2, "<eos>": 3}
        self.freq_threshold = freq_threshold
        self.VOCAB_SIZE = VOCAB_SIZE
        self.get_tokenizer = get_tokenizer("basic_english")

    def __len__(self):
        
        return len(self.itos)

    def get_output(self, sentence_to_indices, NUMBER_OF_WORDS):
         
        '''
        Purpose: Generate one - hot representation of sentence, ready for model training
        Input(s): 
            sentence_to_indices: A dictionary which contains the words and indices as key, value pairs
            NUMBER_OF_WORDS: The maximum number of words a sentence can contain
        Outputs(s):
            One-hot vectors stacked into an array
        
        '''
        
        arr = np.zeros((NUMBER_OF_WORDS, self.VOCAB_SIZE))
        pad_number = 1 # The pad in sentence to index is seen as 1
        for i in range(len(arr)):
            if(i<len(sentence_to_indices)):
                arr[i][sentence_to_indices[list(sentence_to_indices.keys())[i]]] = 1 # set a given key to 1, while leaving the others at zero
            else:
                arr[i][pad_number] = 1 # pad to complete the remaining words to make up the NUMBER OF WORDS needed for the model
                
        return arr
    def sentence_to_indices(self, sentence, dictionary):
         
        '''
        Purpose: Take an input sentence and convert it to a dictionary which has words and their corresponding indices in the vocabulary as key, value pairs
        Input(s):
            sentence: The sentence whose words have to be linked to indices
            dictionary: The dictionary which will contain the word to indices
        Outputs(s):
            sentence_to_index: word to index pair dictionary
        
        '''
        
        sentence_to_index = {}
        
        tokenizer = self.get_tokenizer 
        for word in tokenizer(sentence):# go tgrough all the words formed after tokenizing the sentence
            try:
                if dictionary[word] < self.VOCAB_SIZE: # if word is part of vocabulary

                    sentence_to_index[word] = dictionary[word] 
                else: # else it isn't added to the sentence_to_index
                    continue
            except:
                sentence_to_index[word] = 0 # in case the word isn't found in the dictionary, we consider it to be unknown
        return sentence_to_index
    
    
    def vocab_creator(self,sentence_list):
         
        '''
        Purpose: From a give corpus, generate a WORD vocabulary which maps a givenn word to a given index
        Input(s): 
            sentence_list: A corpus of all sentences extracted from the videos in the dataset
        Outputs(s):
            word_to_index: the word to index of all words contained in the textual corpus
        
        '''
        frequencies = {}
        idx = 4
        stoi = {}

        tokenizer = self.get_tokenizer
        for sentence in sentence_list:
            try:
                for word in tokenizer(sentence):
                    if word not in frequencies:
                        frequencies[word] = 1

                    else:
                        frequencies[word] += 1

                    if frequencies[word] == self.freq_threshold:
                        self.word_to_index[word] = idx
                        idx += 1
            except:
                pass 
        return self.word_to_index
