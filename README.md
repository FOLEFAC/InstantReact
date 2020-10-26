# InstantReact
This repo contains code for the <b>InstantReact</b> project. This project helps people and organizations with <b>CCTV cameras</b> take quick decisions automatically depending on what the Camera sees.
In this project, we shall use <b>Facebook's PyTorch, Wit.ai, Messenger Platform, ReactJs and Docusaurus</b> to train and save a deep learning model which does Video Captioning, to extract sound from a video footage, signal in case of necessity to security or any person who may be of help and  do a demo app using <b>ONNX.js</b> and document a library which we shall develop and which will permit other <b> developers</b> easily use their own data and achieve great results!!!

<p>Basic knowledge of PyTorch, convolutional Neural Networks, Recurrent Neural Networks and Long Short Term Memories is assumed.</p>
<p>If you're new to PyTorch, first read <a href="https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html" rel="nofollow">Deep Learning with PyTorch: A 60 Minute Blitz</a> and <a href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html" rel="nofollow">Learning PyTorch with Examples</a>.</p>
<p>Questions, suggestions, or corrections can be posted as issues.</p>
<p>I'm using <code>PyTorch 1.6.0</code> in <code>Python 3.7.4</code>.</p>

<h1><a id="user-content-contents" class="anchor" aria-hidden="true" href="#contents"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Contents</h1>
<p><a href="https://github.com/FOLEFAC/InstantReact#objective"><em><strong>Objective</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#overview"><em><strong>Overview</strong></em></a>

<ul>
<li><p><em><strong>Video Captioning</strong></em>
</li>
</ul>

<ul>
<li><p><em><strong> Sound Recognition </strong></em>
</li>
</ul>

<ul>
<li><p><em><strong> Alert Message</strong></em>
</li>
</ul>


<ul>
<li><p><em><strong> Package Documentation </strong></em>
</li>
</ul>

<ul>
<li><p><em><strong> Demo </strong></em>
</li>
</ul>

</li>
</ul>
</p>
<p><a href="https://github.com/FOLEFAC/InstantReact#implementation"><em><strong>Implementation</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#training"><em><strong>Training</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#evaluation"><em><strong>Evaluation</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#inference"><em><strong>Inference</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#faqs"><em><strong>Frequently Asked Questions</strong></em></a></p>
<h1><a id="user-content-objective" class="anchor" aria-hidden="true" href="#objective"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Objective</h1>
<p><strong>Build a system, which can extract useful information from a video like actions and sounds.</strong> <br> This will for example permit security agents be more efficient as the computer can be trained to automatically see what is contained in the video, and alert them so that they can take quick and instant measures. 
 <br>Also we shall see how to build a package and quickly document it so that other developers can make use of it with ease. Then finally we shall see how to deploy these models in production using ONNX and how to do a demo app in ReactJs.
 
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
 
 <p>
 
 <code>
   
    
    import torch
    from torchvision import datasets, models, transforms
    import torch.nn as nn 
    import torch.optim as optim 
    import torch.nn.functional as F 
    from torch.utils.data import (DataLoader,Dataset)

    import torchvision.datasets as datasets 

    import torchvision.transforms as transforms

    import torchtext 

    from torchtext.data import get_tokenizer

    import re

    import os

    import cv2

    import numpy as np 

    from PIL import Image

    import pandas

    import math 

    import pickle 

    from torchsummary import summary

    import torch

    from time import time

    def __init__(self, input_size, hidden_size, num_layers):
 
        super(Encoder_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        
        out, (h_n, c_n) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)   
        
        return h_n, c_n
 </code>
 
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
<p><strong> Take note that you can simply modify this csv file and input your own data, then use it to train the model to solve your specific problem <strong></p>
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

 
