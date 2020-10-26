
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
import pandas as pd # Extracting data from csv
import math # Math package
import pickle # Saving variables for later usage.

from torchsummary import summary # Make understanding of models easier
import torch # PyTorch library
from time import time # Using timer in code


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use Cuda if GPU available!


class Utils:
    '''
    This class contains methods which help in processing our dataset
    Args: No arguments
    
    '''
    def __init__(self):
        pass

    def output_text(self, train_corpus, video = None):
        
        '''
        Purpose: Generate all text present in video using the csv file which contains all the text in the videos
        Input(s):
            train_corpus: The file path to the csv file
            video: A video file whose text is to be generated
        Outputs(s):
            Final description: the text representing a video caption
        
        '''
    
        df = pd.read_csv(train_corpus)
        if (video):
            video_id,start,end = self.get_video_id(video) # 

            final_description=''
            for i in range(len(df)):

                if df['VideoID'][i]==str(video_id) and df['Start'][i]==int(start) and df['End'][i]==int(end) and df['Language'][i]=='English':

                    final_description=df['Description'][i]
        else:
            
            final_description = []
            for i in range(len(df)):
                if (df['Language'][i]=='English'):
                    final_description.append(df['Description'][i])
        return final_description
            

    def get_video_id(self, video_path):
            
        '''
        Purpose: Extract video name (without extension) and also remove the start and end tags from the video file name
        Input(s): video file path EX: videoname_xx_yy.avi
        Outputs(s): extracted videoname, xx = start tag, yy=end tag
        
        '''
        video_id=None
        start=None
        end=None
        video_path=video_path[0:len(video_path)-4]
        counter=0
        for i in reversed(range(len(video_path))):
            if (video_path[i]=='_' and counter<2):

                if (counter == 0):
                    end=video_path[i+1:]
                    video_path=video_path[0:i]
                else:
                    start=video_path[i+1:]
                    video_path=video_path[0:i]
                counter+=1
        video_id=video_path

        return video_id,start,end
    @staticmethod
    def tagger_input(text):    
            
        '''
        Purpose: Add the beginning of sentence tag on a text
        Input(s): 
            text: a String which represents a sentence from a video
        Outputs(s): 
            text: A tagged String
        
        '''
    
        bos="<bos> "
        text= bos+text 
        return text
    
    @staticmethod
    def tagger_output(text):  
           
        '''
        Purpose: Add the end of sentence tag on a text
        Input(s): 
            text: a String which represents a sentence from a video
        Outputs(s): 
            text: A tagged String
        
        '''
        eos=" <eos>"
        text= text+eos
        
        return text

    @staticmethod
    def clean_text(texts):
            
        '''
        Purpose:Clean text by removing unnecessary characters and altering the format of words.
        Input(s):
            texts: Texts which contain several symbols which aren't used by our model
        Outputs(s):
            texts: Texts which have been cleaned
        
        '''
        for i in range(len(texts)):

            if(texts=="Commands[195]part4 of 9"):
                texts="commands 195 part 4 of 9"

            texts = texts.lower()
            texts = re.sub(r"i'm", "i am", texts)
            texts = re.sub(r"he's", "he is", texts)
            texts = re.sub(r"she's", "she is", texts)
            texts = re.sub(r"it's", "it is", texts)
            texts = re.sub(r"that's", "that is", texts)
            texts = re.sub(r"what's", "that is", texts)
            texts = re.sub(r"where's", "where is", texts)
            texts = re.sub(r"how's", "how is", texts)
            texts = re.sub(r"\'ll", " will", texts)
            texts = re.sub(r"\'ve", " have", texts)
            texts = re.sub(r"\'re", " are", texts)
            texts = re.sub(r"\'d", " would", texts)
            texts = re.sub(r"\'re", " are", texts)
            texts = re.sub(r"won't", "will not", texts)
            texts = re.sub(r"\n","",texts)
            texts = re.sub(r"\r","",texts)
            texts = re.sub(r"_"," ",texts)
            texts = re.sub(r"can't", "cannot", texts)
            texts = re.sub(r"n't", " not", texts)
            texts = re.sub(r"n'", "ng", texts)
            texts = re.sub(r"'bout", "about", texts)
            texts = re.sub(r"'til", "until", texts)
            texts = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,&]", "", texts)

        return texts
    
    def video_to_frames(self, video_path,frame_number, device, INPUT_SIZE , model, transform):
         
        '''
        Purpose: Take a video file and produce coded frames out of it
        Input(s):
            video_path: The video file to be processed
            frame_number: The number of frames we want to extract (In our example, it is 40)
            device: The device on which the inference will be done
            INPUT_SIZE: The dimension of the output array of each frame
            model: The CNN Model used for inference
            transform: The transform object, which will process all images before they are passed to the model
        Outputs(s):
            The coded frames of dimension frame_number X  INPUT_SIZE (Ex: 40 X 2850)
        
        '''
        print(video_path)
        cap=cv2.VideoCapture(video_path) # read the video file
        number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # get the number of frames
        get_filter=int(number_of_frames/frame_number) # obtain the factor of video number of frames to the number of frames
        print("gett filter = :",get_filter)
        #we want to extract, so that There is equal spacing between the frames which make up the videos 
        
        current_frame=0
        total_features = torch.zeros([frame_number, INPUT_SIZE]) # initialize the total_features 
        total_features.to(dtype = torch.float16)
        t=0
        while (current_frame<number_of_frames):
            ret,frame = cap.read()
            
            if ((current_frame%get_filter) == 0 and t<frame_number):
                with torch.no_grad(): 
                    
                    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # read the image using OpenCV library
                    frame = Image.fromarray(cv2_im)

                    frame = transform(frame) # Use transform to process the image before inference
                    
                    frame = frame.to(device) # set frame to be inferred using the set device
                    model = model.to(device) # set model to infer using the set device
                    
                    model.eval() # put model in evaluation mode

                    frame_feature = model(frame[None])
                    
                    frame_feature = torch.squeeze(frame_feature,0)
                    
                    total_features[t] = frame_feature

                
                t+=1
            current_frame+=1
            
        cap.release()
        cv2.destroyAllWindows()
        
        return total_features

    def get_pre_data(self, train_dir, frame_number, INPUT_SIZE, model , transform ):
         
        '''
        Purpose: Could be used to obtain the coded frames, and stored in a pickle file, such that training can be faster
        Input(s): 
            train_dir: The directory containing all  video files to be processed
            frame_number: The number of frames we want to extract (In our example, it is 40)
            INPUT_SIZE: The dimension of the output array of each frame
            model: The CNN Model used for inference
            transform: The transform object, which will process all images before they are passed to the model
        Outputs(s):
            All the coded frames in one output
        
        '''
        print(train_dir)
        train_video_list=os.listdir(train_dir) # get list of all files in the train directory
        i=0
        all_output = torch.zeros([len(train_video_list), frame_number, INPUT_SIZE])
        
        for video_path in train_video_list:
            
            video_path=train_dir+video_path
            
            output=self.video_to_frames(video_path,frame_number, 'cuda', INPUT_SIZE, model, transform)
            
            all_output[i] = output
            i += 1
            
        return all_output

    