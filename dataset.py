
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

from utils import Utils
from text_processor import TextProcessor
import models


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use Cuda if GPU available!


class CustomDataset(Dataset):

    def __init__(self, train_dir, train_corpus, device, dictionary, VOCAB_SIZE, NUMBER_OF_WORDS, INPUT_SIZE, number_of_frames, transform, model = None, pre_data = None):
        
        self.train_dir = train_dir
        self.train_dir_list = os.listdir(train_dir)
        self.model = model
        self.transform = transform
        self.number_of_frames = number_of_frames
        self.utils = Utils()
        self.word_to_index = dictionary
        self.VOCAB_SIZE = VOCAB_SIZE
        self.NUMBER_OF_WORDS = NUMBER_OF_WORDS
        self.INPUT_SIZE = INPUT_SIZE
        self.pre_data = pre_data
        self.device = device
        self.train_corpus = train_corpus
        
    def __len__(self):
        return len(self.train_dir_list)
    

    def __getitem__(self, idx):
        
        textprocessor = TextProcessor(VOCAB_SIZE = self.VOCAB_SIZE)
        utils = Utils()
        
        
        video_file = self.train_dir_list[idx] # get video file corresponding to the id, idx
        
        
        output_text = self.utils.output_text(self.train_corpus, video_file) # get the text contained in the video file
        
        
        #### generate input 2,  from the output_text
        sentence_to_index = textprocessor.sentence_to_indices(utils.tagger_input(utils.clean_text(output_text)), self.word_to_index)
        X_2 = textprocessor.get_output(sentence_to_index, self.NUMBER_OF_WORDS)
        
        #### generate output,  from the output_text
        sentence_to_index = textprocessor.sentence_to_indices(utils.tagger_output(utils.clean_text(output_text)), self.word_to_index) 
        y = textprocessor.get_output(sentence_to_index, self.NUMBER_OF_WORDS)
        
        video_path = self.train_dir + video_file
        
        # generate input 1
        X_1 = utils.video_to_frames(video_path, self.number_of_frames, self.device, self.INPUT_SIZE, self.model, self.transform)
        #X_1 = pre_data[idx]
        return (X_1,torch.tensor(X_2)), torch.tensor(y)