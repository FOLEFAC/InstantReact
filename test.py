
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

import argparse
from torchsummary import summary # Make understanding of models easier
import torch # PyTorch library
from time import time # Using timer in code


from utils import Utils
from text_processor import TextProcessor
from dataset import CustomDataset
from models import Encoder_LSTM, Decoder_LSTM, Seq2Seq

import models as md

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use Cuda if GPU available!




parser = argparse.ArgumentParser()
parser.add_argument('--NUMBER_OF_FRAMES', type=int, default=40) 
parser.add_argument('--LEARNING_RATE', type=float, default=1e-3)
parser.add_argument('--BATCH_SIZE', type=int, default=1) 
parser.add_argument('--EPOCH', type=int, default=10) 
parser.add_argument('--TRAINING_DEVICE', type=str, default='cuda') 
parser.add_argument('--VOCAB_SIZE', type=int, default=200) 
parser.add_argument('--NUMBER_OF_WORDS', type=int, default=10) 
parser.add_argument('--HIDDEN_SIZE', type=int, default=300) 
parser.add_argument('--INPUT_SIZE', type=int, default=4096) 
parser.add_argument('--NUMBER_OF_LAYERS', type=int, default=1) 
parser.add_argument('--video_file', type=str)
parser.add_argument('--train_corpus', type=str)
parser.add_argument('--load_weights', type=str)

FLAGS = parser.parse_args()







def main(argv = None):

    """
    Training.
    """

    ### parametres

    LEARNING_RATE = FLAGS.LEARNING_RATE
    NUMBER_OF_FRAMES = FLAGS.NUMBER_OF_FRAMES
    BATCH_SIZE = FLAGS.BATCH_SIZE
    EPOCH = FLAGS.EPOCH
    TRAINING_DEVICE = FLAGS.TRAINING_DEVICE
    VOCAB_SIZE = FLAGS.VOCAB_SIZE
    NUMBER_OF_WORDS = FLAGS.NUMBER_OF_WORDS
    HIDDEN_SIZE = FLAGS.HIDDEN_SIZE
    INPUT_SIZE = FLAGS.INPUT_SIZE
    NUMBER_OF_LAYERS = FLAGS.NUMBER_OF_LAYERS
    tsfm = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_corpus = FLAGS.train_corpus
    utils = Utils()
    all_text = utils.output_text(train_corpus)
    text_processor = TextProcessor(freq_threshold = 10)
    dictionary = text_processor.vocab_creator(all_text)



    ### Model definition
    encoder = Encoder_LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE , num_layers = NUMBER_OF_LAYERS)
    decoder = Decoder_LSTM(input_size = VOCAB_SIZE, hidden_size = HIDDEN_SIZE , num_layers = NUMBER_OF_LAYERS,number_of_words = NUMBER_OF_WORDS)
    model_seq_to_seq = Seq2Seq(encoder, decoder).to(device)
    model = model_seq_to_seq


    ### load the state_dict of model if model has been pretrained.
    model.load_state_dict(torch.load(FLAGS.load_weights))




    #### Model Testing
    model.eval();
    from random import randint
    import matplotlib.pyplot as plt

    utils = Utils()

    video_path = FLAGS.video_file

    video_pre_data = utils.video_to_frames(video_path,frame_number = NUMBER_OF_FRAMES, device = 'cuda', INPUT_SIZE = INPUT_SIZE , model = md.model_vgg, transform = tsfm)
    
    X_2  = torch.zeros([NUMBER_OF_WORDS,VOCAB_SIZE])

    for i in range(NUMBER_OF_WORDS):
        if (i == 0):
            
            X_2[i][2] = 1
        else:
            X_2[i][1] = 1

    input_data = video_pre_data.unsqueeze(0)

    final_sentence = []

    X_2 = X_2.unsqueeze(0)
    X_2 = X_2.to(device)
    input_data = input_data.to(device)




    for i in range(NUMBER_OF_WORDS-1):
        with torch.no_grad():
            predicted = model(input_data, X_2)
            predicted = predicted.squeeze(0)

            final_sentence.append(next((key for key, value in dictionary.items() if value == torch.argmax(predicted[i])), None))
            X_2[0][i+1][torch.argmax(predicted[i])] = 1
            X_2[0][i+1][1] = 0
    print(final_sentence)


    

if __name__ == '__main__':
    main()