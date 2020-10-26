# InstantReact
This repo contains code for the <b>InstantReact</b> project. This project helps people and organizations with <b>CCTV cameras</b> take quick decisions automatically depending on what the Camera sees.
In this project, we shall use <b>Facebook's PyTorch, Wit.ai and Docusaurus</b> to train and save a deep learning model which does Video Captioning, to extract sound from a video footage, and document a library which we shall develop and which will permit other <b> developers</b> easily use their own data and achieve great results!!!

<p>Basic knowledge of PyTorch, convolutional Neural Networks, Recurrent Neural Networks and Long Short Term Memories is assumed.</p>
<p>If you're new to PyTorch, first read <a href="https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html" rel="nofollow">Deep Learning with PyTorch: A 60 Minute Blitz</a> and <a href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html" rel="nofollow">Learning PyTorch with Examples</a>.</p>
<p>Questions, suggestions, or corrections can be posted as issues.</p>
<p>I'm using <code>PyTorch 1.6.0</code> in <code>Python 3.7.4</code>.</p>

<h1><a id="user-content-contents" class="anchor" aria-hidden="true" href="#contents"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Contents</h1>
<p><a href="https://github.com/FOLEFAC/InstantReact#objective"><em><strong>Objective</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#overview"><em><strong>Overview</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#implementation"><em><strong>Implementation</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#dataset"><em><strong>Dataset</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#requirements"><em><strong>Requirements</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#training"><em><strong>Training</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#inference"><em><strong>Inference</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#documentation"><em><strong>Documentation</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#next"><em><strong>Next step</strong></em></a></p>
<h1><a id="user-content-objective" class="anchor" aria-hidden="true" href="#objective"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Objective</h1>
<p><strong>Build a system, which can extract useful information from a video like actions and sounds.</strong> <br> This will for example permit security agents be more efficient as the computer can be trained to automatically see what is contained in the video, and alert them so that they can take quick and instant measures. 
 <br>Also we shall see how to build a package and quickly document it so that other developers can make use of it with ease. 
 
 Then finally we shall see how to deploy these models in production using ONNX and how to do a demo app in ReactJs (This will be in the next version of this tutorial)
 
 <br>  After going through this tutorial, you shall learn how to use the InstantReact Package and also how to create and document yours :)</p>
 <p> <strong> It is worth noting that this project can be used in many other environments apart from security. Imagine a patient who is being monitored, so that in case of a fall, he/she can be rescued very quickly</strong>. So don't limit yourself feel free to use these technologies to solve real world problems you encounter</p>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/images/fighting.gif"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/images/fighting.gif" style="max-width:100%;"></a>
</p>
<hr>

<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Overview</h1>


<p align="center">
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/images/generaloverview.jpg"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/images/generaloverview.jpg" style="max-width:100%;"></a>
</p>

<p>From the figure above, we see that a video file can be broken down into several frames. A frame can be considered as a single image or a single capture from the video at a given point in time. Video files can be broken down into the <strong>frames (images)</strong> and <strong>sound</strong>.
 </p>
 <p>
 In our case, we use the Popular Computer vision library <strong>OpenCV</strong>to break down the video file into frames and the <strong>Moviepy</strong> package to extract the sound from the video. Once the frames are extracted, We use a Deeplearning model (Seq2Seq LSTM) to extract actions made by the different entities present in the selected frames. PyTorch library helps us train this model and do inference very easily (we shall see subsequently in detail how this is done). 
 </p>
 <p>
 We also use Wit.ai powerful speech recognition module to extract speech from the sound. In the next subsections, we shall go in more detail to understand how <strong> PyTorch and Wit.ai</strong> help us solve our problem and also how <strong> Docusaurus</strong> helps document our solution, so that other developers can use our product in other more interesting and important applications.


<h3><a id="user-content-single-shot-detector-ssd" class="anchor" aria-hidden="true" href="#single-shot-detector-ssd"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Video Captioning </h3>

<p align="center">
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/images/videocaptioning.jpg"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/images/videocaptioning.jpg" style="max-width:100%;"></a>
</p>


<p><strong>Video captioning</strong> entails extracting visual information contained in a video and putting it out in the form of text</p>
<p> The diagram above describes a video captioning pipeline <strong>(from extraction of frames in the video to textual description of actions in the video). </strong> </p>
Deep learning algorithms are best suited for solving such problems, since they can help understand image data and also understand patterns existing between the different frames which are present in a video over a given period of time.
<p> To extract the information from the frames (image data), we make use of a Convolutional neural network (CNN), while to understand different actions and generate text over time we use a Recurrent Neural Network (RNN).</p>
<p> Infact, we are using a variant or a more sophisticated CNN, which is the <strong> VGG16</strong>, and a more sophisticated RNN, called LSTMs. These are not the most advanced models, as other advanced models and algorithms exist which can be used to better learn from a given dataset (Videos and their captions). 
</p>

<p> We have a dataset comprised of videos (<strong> inputs</strong>) and their captions (in English) (<strong> ouputs</strong>). We develop a model which will predict the captions when a new video is inputted.</p>
<p> For each video in the dataset, we chop it into several frames using <strong> OpenCV </strong>, to make <code><strong> N </strong></code> number of frames. We considered  <code><strong> N = 40 </strong></code>. Each image which is part of the  <code><strong> N </strong></code> frames is resized into a  <code><strong> 224 x 224 x 3 </strong></code> image. This notation is the <strong>NHWC </strong> notation, whereas <strong> PyTorch </strong> uses the <strong>NCHW</strong> notation. So each frame is resized into a <code><strong> 3 x 224 x 224  </strong></code> image. Fortunately for us, <strong>PyTorch</strong> helps us do such transformations very easily. So after the resizing, we have a <code><strong> N X 3 X 224 x 224</strong></code> tensor, where we take <code><strong> N = 40 </strong></code> Each of the <strong> 3 X 224 x 224</strong> image tensors are fed into the VGG16 network and produces a <strong> 1000 </strong> dimensional matrix. In this tutorial, we modify the VGG16 model, such that we instead output a <strong> 4096 </strong> dimensional matrix. This now leads to a <code><strong> N x 4096 </strong></code></p> dimensional matrix or tensor. You could increase or reduce this value and see how it affects your results.</p>
 
 <p> Now we pass this <code><strong> N x 4096 </strong></code> tensor into a Sequence to Sequence LSTM model. This is a good <a href  = "https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"> tutorial </a>which explains why and how this model is used. </p>
 
 <p> To understand the Sequence to sequence models, we need to start by understanding the reason why we use basic RNN models and even why we need RNN models. In our problem, we want a model which can generate a sentence from actions in the video. A sentence is made of a sequence of words, hence the need for a sequence model, which takes decisions based on events occuring over a given period of time (in this case over the time taken by the video).

<p> That said, RNNs are the most appropriate for such tasks, as they are designed to capture information when given a sequence of inputs in a given time frame. Generally RNNs have equal number of inputs and outputs. This poses a problem, since our inputs and outputs musn't always have the same length or size (ex: we can have an input consisting of 1000 frames, but we want to generate only sentences of maximum 50 words).</p>
 
 <p>A very intuitive example is with Machine Translation. When you want to translate the sentence: <strong> I love Deeplearning and Football</strong> in French we get: <strong>J'aime le Deeplearning et le football</strong> which is made of <code> 6 > 5 words</code> as compared to the english sentence. We see clearly that the standard RNN isn't the most appropriate model for such tasks, hence the need for Sequence to Sequence Models.</p>
 
 <p>In Sequence to Sequence models, the inputs and the outputs are 'separated'. We have a RNN layer which gets the inputs and which is configured to get that type of input and another RNN layer which produces the outputs and this time around can be configured to have a different sequence length as compared to the inputs. Now, we replace the RNNs on both side of the sequence to sequence model with an LSTM layer, which is composed of several LSTM units, as you can see in the diagram above.</p>
 
 <p> In order to model our data appropriately using the sequence to sequence models, we must first understand a single LSTM layer. A single LSTM layer is made of several LSTM units , which in turn has <code> 3 inputs (h = hidden state, c = cell state, x = input data)</code> and <code> 2 outputs (h = hidden state = output data, c = cell state) </code>. For the first part of the sequence to sequence model, only the outputs of the last LSTM block is useful and help to carry learned information to the 2nd part. In the 2nd part, we model our data such that for training, it takes as input the caption and also outputs the caption, with a little offset of one word.
</p>
<p> We can see on the schematic above that the sentence Many People are fighting is passed in the input as <strong> BOS Many people are fighting EOS<PAD>...<PAD></strong>, while the output is the same word, but starting with <strong> Many </strong> and not <strong> BOS </strong>. During testing of the model, once one LSTM unit produces an output, the output is used as input in the next LSTM unit and this continues till the end. This technique may also be used during training.
 </p>
<p> The words which form the captions are passed into the model in their <strong> one - hot </strong> notation. So if we have a vocabulary of 2850 words (although not very practical), each words can be transformed into a 1- D array with all zeros and having 1 only at one position which is unique to that number. Some examples were taken in the figure above.
 </p>
 <h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#implementation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Implementation</h1>
 
 <p>After understanding the concepts between the Video captioning module, we start with the implementation code. </p>
 
<p> We start by importing necessary packages and libraries</p>

<code>

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
      
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 </code>

<p> Before beginning with explaining each class or function, lets agree on a <strong> code structure</strong> </p>
<code>

            -------- Helper Classes
                               ++++ Utils class ( Which uses torchtext and other packages to treat images, videos and text)
                               ++++ TextProcessor class (which uses torchtext to treat text)
            --------  Custom Dataset class: PyTorch offers this possibility of creating and processing our custom datasets with much ease
            --------  Model definition classes: 
                                         +++++  VGG16 pretrained model will be used, although a slight modification will be made
                                         +++++  LSTM SEQ2SEQ again, we shall use PyTorch modules to easily create this class (from scratch, unlike the VGG PRETRAINED MODEL)
             It should be noted that one can rewrite a class for a pretrained model (but for our problem, that wasn't necessary)

            --------  Key parameter definition: Here parameters like the learning rate,number of epochs, ... are defined 
            --------  Optimizer definition
            --------  Training
            --------  Testing

 </code>
 <p> In order to convert the video file into an N number of frames, we have the following function
 <code>

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
           print("when we getfilter = :", get_filter)
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
 </code>
 Let's see how to process the text giving to us. First we see the formatting of the video text corpus data:
 <p align="center">
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/images/text_file.PNG"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/images/text_file.PNG" style="max-width:100%;"></a>
</p>
Taking the first row, we should note that the video  with videoId: <code>mv89psg6zh4</code>will come as <code>mv89psg6zh4_33_46.avi</code>, so we simply have to remove the 33 and 46 which correspond to start and end. Also we are only working in the English language, and thats why in the code we only select rows which have English as language. Thats what the function below does. You can see the code for the get_video_id function in the notebook file.
<strong>Pandas</strong> library is used to extract the information from csv file.
<p><strong> Take note that you can simply modify this csv file and input your own data, then use it to train the model to solve your specific problem </strong></p>
<code>


             def output_text(self, train_corpus, video = None):

                    '''
                    Purpose: Generate all text present in video using the csv file which contains all the text in the videos
                    Input(s):
                        train_corpus: The file path to the csv file
                        video: A video file whose text is to be generated
                    Outputs(s):
                        Final description: the text representing a video caption

                    '''

                    df = pandas.read_csv(train_corpus)
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

 </code>

<p>Now, we create a vocabulary of words with a given size, while limiting its size depending on the problem we want to solve. torchtext tokenizer helps us in tokenizing sentences very easily. A tokenizer simply takes in a sentence and separates it into words.</p>
<code>
 
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

 </code>
 <p> We then use a sentences to indices method to convert a caption e.g. I LIKE FOOTBALL ---> {'I': 23, 'LIKE': 234, 'FOOTBALL': 189}, then using this method, we can easily generate one-hot outputs for each video caption using the code below</p>
 
 <code>
 
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
</code>
 
 After treating the data, we proceed to creating our dataset. The Dataset class from PyTorch helps us easily create datasets, and also helps us avoid loading all our dataset in memory during training, as we just need to pass in the index as parameter in the __getitem__ function.<strong>(notice how idx is passed)</strong>
 
 <code>

       def __getitem__(self, idx):

              textprocessor = TextProcessor(VOCAB_SIZE = self.VOCAB_SIZE)
              utils = Utils()


              video_file = self.train_dir_list[idx] # get video file corresponding to the id, idx


              output_text = self.utils.output_text(self.train_corpus, video_file) # get the text contained in the video file


              #### generate input 2,  from the output_text
              sentence_to_index = textprocessor.sentence_to_indices(utils.tagger_input(utils.clean_text(output_text)), self.word_to_index)
              X_2 = textprocessor.get_output(sentence_to_index, NUMBER_OF_WORDS)

              #### generate output,  from the output_text
              sentence_to_index = textprocessor.sentence_to_indices(utils.tagger_output(utils.clean_text(output_text)), self.word_to_index) 
              y = textprocessor.get_output(sentence_to_index, NUMBER_OF_WORDS)

              video_path = self.train_dir + video_file

              # generate input 1
              X_1 = utils.video_to_frames(video_path, self.number_of_frames, self.device, self.INPUT_SIZE, self.model, self.transform)
              #X_1 = pre_data[idx] ### this may be used if we collect the pretrained video frames from a pickle file instead of doing inference during training
              return (X_1,torch.tensor(X_2)), torch.tensor(y)
                         
 </code>
 Next, we define paramters which shall be used in training/inference. We again use PyTorch Transforms to avoid re-writing them. there are more advanced ways of using PyTorch transforms which permit us do more data augmentation
 <code>
        
        ### parametres

       LEARNING_RATE = 1e-3
       NUMBER_OF_FRAMES = 40
       BATCH_SIZE = 1
       EPOCH = 10
       TRAINING_DEVICE = 'cuda'
       VOCAB_SIZE = 200
       NUMBER_OF_WORDS = 10
       HIDDEN_SIZE = 300
       INPUT_SIZE = 4096
       NUMBER_OF_LAYERS = 1
       tsfm = transforms.Compose([
           transforms.Resize([224, 224]),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
       ])
       train_dir = 'D:/Machine_Learning/datasets/YouTubeClips_2/YouTubeClips/'
       train_corpus = 'D:/Machine_Learning/datasets/video_corpus/video_corpus.csv'

       ### optimizer and loss function
       optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
       criterion = nn.CrossEntropyLoss()
       
</code>

<p>We shall define a seq to seq model using 2 Pytorch LSTMs. But its important you understand how to use the nn.LSTM class. See: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html for more details.
<strong> When trying to code a Model or develop any module in PyTorch, always refer to the documentation. </strong>For example in the documentation, by default 
 Inputs take the form <code>sequence_length, batch_size, input_size</code>, and to use the notation <code> batch_size, sequence_length, input_size</code> as in our code, we use the <code>batch_first</code> parametre.</p>
 Below, we see how we define three separate classes for the seq to seq model.
 <p><strong> TAKE NOTE OF THE DIMENSIONS!!!</strong></p>
 <code>


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


</code>
 
 
 <p> Finally we train and test our models</p>
 
 <code>
     
     #### Model Training

    EPOCH = 10
    import time
    print_feq = 100
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
            torch.save(model.state_dict(), 'model_lstm_2.pth')
        ### save best model
        if(epoch_loss < best_loss):
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'model_lstm_best_loss.pth')
        print("The loss for this epoch is = :", epoch_loss/lent(train_dl))
 </code>
<p>
<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#dataset"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Dataset</h1>

 <p>
 <strong>Note</strong>: This tutorial is meant to help you use your own custom dataset in solving a more precise problem (e.g. patient surveillance,...) and the dataset we used isn't directly linked to people fighting, but contains day to day actions instead.
  </p>
 </p>
 <p>
 <ul>
  
 <li> Video data: http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar (It contains .avi files)</li>
 <li>Text data: video_corpus.csv file in this repo </li>
  </ul>
 
 </p>
 </p>
 
 <p>
<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#requirements"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Requirements</h1>


 </p>
 <p>
 <ul>
  
 <li> Pandas</li>
 <li> OpenCV</li>
 
 <li> PyTorch</li>
 <li> Torchtext</li>
 <li> Numpy</li>
 <li> Re</li>
 <li> Pickle</li>
 
  </ul>
 </p>
 
 <p>
<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#training"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Training</h1>

 <p>
 <strong>Note</strong>: This tutorial is meant to help you use your own custom dataset in solving a more precise problem (e.g. patient surveillance,...) and the dataset we used isn't directly linked to people fighting, but contains day to day actions instead.
  </p>
 
 <p>
Training:
 <code>
            
        --NUMBER_OF_FRAMES', type=int, default=40
        --LEARNING_RATE', type=float, default=1e-3
        --BATCH_SIZE', type=int, default=1
        --EPOCH', type=int, default=10
        --TRAINING_DEVICE', type=str, default='cuda'
        --VOCAB_SIZE', type=int, default=200
        --NUMBER_OF_WORDS', type=int, default=10
        --HIDDEN_SIZE', type=int, default=300
        --INPUT_SIZE', type=int, default=4096 
        --NUMBER_OF_LAYERS', type=int, default=1 
        --train_dir', type=str
        --train_corpus', type=str
        --load_weights', type=str

 </code>
 
 <strong>Example of usage:</strong>
 <code> 
 
     python train.py --NUMBER_OF_WORDS 32 --train_dir D:/Machine_Learning/datasets/YouTubeClips_2/YouTubeClips/ --train_corpus  D:/Machine_Learning/datasets/video_corpus/video_corpus.csv --load_weights model_lstm_2.pth

 </code>
 
 </p>
 
 <p>
<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#inference"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Inference</h1>

 <p>
 Inference:
 <code>
  

            --NUMBER_OF_FRAMES', type=int, default=40
            --LEARNING_RATE', type=float, default=1e-3
            --BATCH_SIZE', type=int, default=1
            --EPOCH', type=int, default=10
            --TRAINING_DEVICE', type=str, default='cuda'
            --VOCAB_SIZE', type=int, default=200
            --NUMBER_OF_WORDS', type=int, default=10
            --HIDDEN_SIZE', type=int, default=300
            --INPUT_SIZE', type=int, default=4096 
            --NUMBER_OF_LAYERS', type=int, default=1 
            --video_file', type=str
            --train_corpus', type=str
            --load_weights', type=str

 </code>
 <strong>Example of usage:</strong>
 <code>
 
       python test.py --HIDDEN_SIZE 300 --video_file D:/Machine_Learning/datasets/YouTubeClips_2/YouTubeClips/_O9kWD8nuRU_45_49.avi --load_weights model_lstm_best_loss.pth --train_corpus D:/Machine_Learning/datasets/video_corpus/video_corpus.csv

 </code>
 
 
 </p>
 
 
 
 
 
 <h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#documentation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Documentation</h1>
<p>
To document our project, so that other developers can make use of our project to develop very cool applications, we shall use <strong> Docusaurus </strong>. It is pretty easy to use and helps a great deal in reducing effort needed to document our Open Source projects; while maintaining a certain standard.
 </p>
 <p>
 To install Docusaurus, you can follow this tutorial: https://docusaurus.io/docs/en/installation.
 To deploy Docusaurus, you can follow this tutorial: https://v2.docusaurus.io/docs/deployment/
 
 </p>
 
 
 
 <p>
 
<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#next"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a> What's next?</h1>
 </p>
 <p><ul>
 <li>
  Using more adapted deep learning models to increase accuracy (e.g. attention models,..)
 </li>
 
 <li>
  Modifying the parametres to obtain better results
 </li>
 <li>
 Setting up project with Messenger Platform to allow automatic Messages sent via messenger to a security agent.
 </li>
 <li>
 Producing the weights in the ONNX format and then developing a ReactJs app using ONNX.js
</li>
 <li>
Deploying the Documentation on github pages
</li>
 
 
 </ul>
 </p>
 
 
 
