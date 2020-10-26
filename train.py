
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
import argparse # for training using cli
import re # regex library
import os # Doing operating system operations
import cv2 # Computer vision tasks with OpenCV
import numpy as np # Powerful arrray computation library
from PIL import Image # WOrking with image files
import pandas# Extracting data from csv
import math # Math package
import pickle # Saving variables for later usage.

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
parser.add_argument('--train_dir', type=str)
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
    train_dir = FLAGS.train_dir#'D:/Machine_Learning/datasets/YouTubeClips_2/YouTubeClips/'
    train_corpus = FLAGS.train_corpus#'D:/Machine_Learning/datasets/video_corpus/video_corpus.csv'

    print("train_dir is =", train_dir)
    print("train_corpus =", train_corpus)

    utils = Utils()
    all_text = utils.output_text(train_corpus)
    text_processor = TextProcessor(freq_threshold = 10)
    dictionary = text_processor.vocab_creator(all_text)




    ### training data preparation
    train_ds = CustomDataset(train_dir, train_corpus, device, dictionary, VOCAB_SIZE, NUMBER_OF_WORDS, INPUT_SIZE,  NUMBER_OF_FRAMES, tsfm, model = md.model_vgg)
    train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE)

    ### Model definition
    encoder = Encoder_LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE , num_layers = NUMBER_OF_LAYERS)
    decoder = Decoder_LSTM(input_size = VOCAB_SIZE, hidden_size = HIDDEN_SIZE , num_layers = NUMBER_OF_LAYERS,number_of_words = NUMBER_OF_WORDS)
    model_seq_to_seq = Seq2Seq(encoder, decoder).to(device)
    model = model_seq_to_seq


    ### load the state_dict of model if model has been pretrained.
    if(FLAGS.load_weights):
        print("there are weights to be loaded")

        model.load_state_dict(torch.load(FLAGS.load_weights))

    ### optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()



    #### Model Training
    
    import time
    print_feq = 1
    best_loss = np.inf
    for epoch in range(1, EPOCH+1):
        model.train()
        epoch_loss = 0
        
        for step, (img,label) in enumerate(train_dl):
            
            
            time_1 = time.time() ## timing
            
            X_1, X_2 = img ### get inputs
            
            X_1 = X_1.to(device) # Set device 
            X_2 = X_2.to(device) # Set device
            
            
            label = label.to(device) # Set output device
            
            ### zero the parameter gradients
            optimizer.zero_grad()
            
            ### forward
            prediction = model(X_1, X_2)
            
            ### Optimize
            prediction = prediction.to(device)
            prediction = torch.squeeze(prediction,0)
            label = torch.squeeze(label,0)
            
            new_label = torch.zeros([label.shape[0]])
            for l in range(label.shape[0]):
                new_label[l] = np.argmax(label[l].cpu())
            new_label = new_label.to(device)
            loss = criterion(prediction, new_label.long())
            
            # Backward prop.
            loss.backward()
            optimizer.step()
            
            ### print out statistics
            epoch_loss += loss.item()
            if step % print_feq == 0:
                print('epoch:', epoch,
                      '\tstep:', step+1, '/', len(train_dl) + 1,
                      '\ttrain loss:', '{:.4f}'.format(loss.item()),
                      '\ttime:', '{:.4f}'.format((time.time()-time_1)*print_feq), 's')
                
        ### save best model
        if(epoch_loss < best_loss):
            best_loss = epoch_loss

            model_name = 'MODEL_SEQ2SEQ'+ 'VOCAB_SIZE_' + str(VOCAB_SIZE) + 'NUMBER_OF_WORDS_' + str(NUMBER_OF_WORDS)+ 'HIDDEN_SIZE_' + str(HIDDEN_SIZE)+ 'INPUT_SIZE_' + str(INPUT_SIZE)+ 'NUMBER_OF_LAYERS_' + str(NUMBER_OF_LAYERS) 
            torch.save(model.state_dict(), model_name +'.pth')

        print("The loss for this epoch is = :", epoch_loss/len(train_dl))



    

if __name__ == '__main__':
    main()