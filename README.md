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
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/fighting.gif"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/fighting.gif" style="max-width:100%;"></a>
</p>
<hr>

<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Overview</h1>


<p align="center">
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/generaloverview.jpg"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/generaloverview.jpg" style="max-width:100%;"></a>
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
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/videocaptioning.jpg"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/videocaptioning.jpg" style="max-width:100%;"></a>
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
 
 


<ul>
<li>
<p> derived from an existing image classification architecture that will provide lower-level feature maps.</p>
</li>
<li>
<p><strong>Auxiliary convolutions</strong> added on top of the base network that will provide higher-level feature maps.</p>
</li>
<li>
<p><strong>Prediction convolutions</strong> that will locate and identify objects in these feature maps.</p>
</li>
</ul>
<p>The paper demonstrates two variants of the model called the SSD300 and the SSD512. The suffixes represent the size of the input image. Although the two networks differ slightly in the way they are constructed, they are in principle the same. The SSD512 is just a larger network and results in marginally better performance.</p>
<p>For convenience, we will deal with the SSD300.</p>
<h3><a id="user-content-base-convolutions--part-1" class="anchor" aria-hidden="true" href="#base-convolutions--part-1"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Base Convolutions – part 1</h3>
<p>First of all, why use convolutions from an existing network architecture?</p>
<p>Because models proven to work well with image classification are already pretty good at capturing the basic essence of an image. The same convolutional features are useful for object detection, albeit in a more <em>local</em> sense – we're less interested in the image as a whole than specific regions of it where objects are present.</p>
<p>There's also the added advantage of being able to use layers pretrained on a reliable classification dataset. As you may know, this is called <strong>Transfer Learning</strong>. By borrowing knowledge from a different but closely related task, we've made progress before we've even begun.</p>
<p>The authors of the paper employ the <strong>VGG-16 architecture</strong> as their base network. It's rather simple in its original form.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/vgg16.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/vgg16.PNG" alt="" style="max-width:100%;"></a></p>
<p>They recommend using one that's pretrained on the <em>ImageNet Large Scale Visual Recognition Competition (ILSVRC)</em> classification task. Luckily, there's one already available in PyTorch, as are other popular architectures. If you wish, you could opt for something larger like the ResNet. Just be mindful of the computational requirements.</p>
<p>As per the paper, <strong>we've to make some changes to this pretrained network</strong> to adapt it to our own challenge of object detection. Some are logical and necessary, while others are mostly a matter of convenience or preference.</p>
<ul>
<li>
<p>The <strong>input image size</strong> will be <code>300, 300</code>, as stated earlier.</p>
</li>
<li>
<p>The <strong>3rd pooling layer</strong>, which halves dimensions, will use the mathematical <code>ceiling</code> function instead of the default <code>floor</code> function in determining output size. This is significant only if the dimensions of the preceding feature map are odd and not even. By looking at the image above, you could calculate that for our input image size of <code>300, 300</code>, the <code>conv3_3</code> feature map will be of cross-section <code>75, 75</code>, which is halved to <code>38, 38</code> instead of an inconvenient <code>37, 37</code>.</p>
</li>
<li>
<p>We modify the <strong>5th pooling layer</strong> from a <code>2, 2</code> kernel and <code>2</code> stride to a <code>3, 3</code> kernel and <code>1</code> stride. The effect this has is it no longer halves the dimensions of the feature map from the preceding convolutional layer.</p>
</li>
<li>
<p>We don't need the fully connected (i.e. classification) layers because they serve no purpose here. We will toss <code>fc8</code> away completely, but choose to <strong><em>rework</em> <code>fc6</code> and <code>fc7</code> into convolutional layers <code>conv6</code> and <code>conv7</code></strong>.</p>
</li>
</ul>
<p>The first three modifications are straightforward enough, but that last one probably needs some explaining.</p>
<h3><a id="user-content-fc--convolutional-layer" class="anchor" aria-hidden="true" href="#fc--convolutional-layer"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>FC → Convolutional Layer</h3>
<p>How do we reparameterize a fully connected layer into a convolutional layer?</p>
<p>Consider the following scenario.</p>
<p>In the typical image classification setting, the first fully connected layer cannot operate on the preceding feature map or image <em>directly</em>. We'd need to flatten it into a 1D structure.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/fcconv1.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/fcconv1.jpg" alt="" style="max-width:100%;"></a></p>
<p>In this example, there's an image of dimensions <code>2, 2, 3</code>, flattened to a 1D vector of size <code>12</code>. For an output of size <code>2</code>, the fully connected layer computes two dot-products of this flattened image with two vectors of the same size <code>12</code>. <strong>These two vectors, shown in gray, are the parameters of the fully connected layer.</strong></p>
<p>Now, consider a different scenario where we use a convolutional layer to produce <code>2</code> output values.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/fcconv2.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/fcconv2.jpg" alt="" style="max-width:100%;"></a></p>
<p>Here, the image of dimensions <code>2, 2, 3</code> need not be flattened, obviously. The convolutional layer uses two filters with <code>12</code> elements in the same shape as the image to perform two dot products. <strong>These two filters, shown in gray, are the parameters of the convolutional layer.</strong></p>
<p>But here's the key part – <strong>in both scenarios, the outputs <code>Y_0</code> and <code>Y_1</code> are the same!</strong></p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/fcconv3.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/fcconv3.jpg" alt="" style="max-width:100%;"></a></p>
<p>The two scenarios are equivalent.</p>
<p>What does this tell us?</p>
<p>That <strong>on an image of size <code>H, W</code> with <code>I</code> input channels, a fully connected layer of output size <code>N</code> is equivalent to a convolutional layer with kernel size equal to the image size <code>H, W</code> and <code>N</code> output channels</strong>, provided that the parameters of the fully connected network <code>N, H * W * I</code> are the same as the parameters of the convolutional layer <code>N, H, W, I</code>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/fcconv4.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/fcconv4.jpg" alt="" style="max-width:100%;"></a></p>
<p>Therefore, any fully connected layer can be converted to an equivalent convolutional layer simply <strong>by reshaping its parameters</strong>.</p>
<h3><a id="user-content-base-convolutions--part-2" class="anchor" aria-hidden="true" href="#base-convolutions--part-2"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Base Convolutions – part 2</h3>
<p>We now know how to convert <code>fc6</code> and <code>fc7</code> in the original VGG-16 architecture into <code>conv6</code> and <code>conv7</code> respectively.</p>
<p>In the ImageNet VGG-16 <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#base-convolutions--part-1">shown previously</a>, which operates on images of size <code>224, 224, 3</code>, you can see that the output of <code>conv5_3</code> will be of size <code>7, 7, 512</code>. Therefore –</p>
<ul>
<li>
<p><code>fc6</code> with a flattened input size of <code>7 * 7 * 512</code> and an output size of <code>4096</code> has parameters of dimensions <code>4096, 7 * 7 * 512</code>. <strong>The equivalent convolutional layer <code>conv6</code> has a <code>7, 7</code> kernel size and <code>4096</code> output channels, with reshaped parameters of dimensions <code>4096, 7, 7, 512</code>.</strong></p>
</li>
<li>
<p><code>fc7</code> with an input size of <code>4096</code> (i.e. the output size of <code>fc6</code>) and an output size <code>4096</code> has parameters of dimensions <code>4096, 4096</code>. The input could be considered as a <code>1, 1</code> image with <code>4096</code> input channels. <strong>The equivalent convolutional layer <code>conv7</code> has a <code>1, 1</code> kernel size and <code>4096</code> output channels, with reshaped parameters of dimensions <code>4096, 1, 1, 4096</code>.</strong></p>
</li>
</ul>
<p>We can see that <code>conv6</code> has <code>4096</code> filters, each with dimensions <code>7, 7, 512</code>, and <code>conv7</code> has <code>4096</code> filters, each with dimensions <code>1, 1, 4096</code>.</p>
<p>These filters are numerous and large – and computationally expensive.</p>
<p>To remedy this, the authors opt to <strong>reduce both their number and the size of each filter by subsampling parameters</strong> from the converted convolutional layers.</p>
<ul>
<li>
<p><code>conv6</code> will use <code>1024</code> filters, each with dimensions <code>3, 3, 512</code>. Therefore, the parameters are subsampled from <code>4096, 7, 7, 512</code> to <code>1024, 3, 3, 512</code>.</p>
</li>
<li>
<p><code>conv7</code> will use <code>1024</code> filters, each with dimensions <code>1, 1, 1024</code>. Therefore, the parameters are subsampled from <code>4096, 1, 1, 4096</code> to <code>1024, 1, 1, 1024</code>.</p>
</li>
</ul>
<p>Based on the references in the paper, we will <strong>subsample by picking every <code>m</code>th parameter along a particular dimension</strong>, in a process known as <a href="https://en.wikipedia.org/wiki/Downsampling_(signal_processing)" rel="nofollow"><em>decimation</em></a>.</p>
<p>Since the kernel of <code>conv6</code> is decimated from <code>7, 7</code> to <code>3,  3</code> by keeping only every 3rd value, there are now <em>holes</em> in the kernel. Therefore, we would need to <strong>make the kernel dilated or <em>atrous</em></strong>.</p>
<p>This corresponds to a dilation of <code>3</code> (same as the decimation factor <code>m = 3</code>). However, the authors actually use a dilation of <code>6</code>, possibly because the 5th pooling layer no longer halves the dimensions of the preceding feature map.</p>
<p>We are now in a position to present our base network, <strong>the modified VGG-16</strong>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/modifiedvgg.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/modifiedvgg.PNG" alt="" style="max-width:100%;"></a></p>
<p>In the above figure, pay special attention to the outputs of <code>conv4_3</code> and <code>conv_7</code>. You will see why soon enough.</p>
<h3><a id="user-content-auxiliary-convolutions" class="anchor" aria-hidden="true" href="#auxiliary-convolutions"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Auxiliary Convolutions</h3>
<p>We will now <strong>stack some more convolutional layers on top of our base network</strong>. These convolutions provide additional feature maps, each progressively smaller than the last.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/auxconv.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/auxconv.jpg" alt="" style="max-width:100%;"></a></p>
<p>We introduce four convolutional blocks, each with two layers. While size reduction happened through pooling in the base network, here it is facilitated by a stride of <code>2</code> in every second layer.</p>
<p>Again, take note of the feature maps from <code>conv8_2</code>, <code>conv9_2</code>, <code>conv10_2</code>, and <code>conv11_2</code>.</p>
<h3><a id="user-content-a-detour" class="anchor" aria-hidden="true" href="#a-detour"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>A detour</h3>
<p>Before we move on to the prediction convolutions, we must first understand what it is we are predicting. Sure, it's objects and their positions, <em>but in what form?</em></p>
<p>It is here that we must learn about <em>priors</em> and the crucial role they play in the SSD.</p>
<h4><a id="user-content-priors" class="anchor" aria-hidden="true" href="#priors"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Priors</h4>
<p>Object predictions can be quite diverse, and I don't just mean their type. They can occur at any position, with any size and shape. Mind you, we shouldn't go as far as to say there are <em>infinite</em> possibilities for where and how an object can occur. While this may be true mathematically, many options are simply improbable or uninteresting. Furthermore, we needn't insist that boxes are pixel-perfect.</p>
<p>In effect, we can discretize the mathematical space of potential predictions into just <em>thousands</em> of possibilities.</p>
<p><strong>Priors are precalculated, fixed boxes which collectively represent this universe of probable and approximate box predictions</strong>.</p>
<p>Priors are manually but carefully chosen based on the shapes and sizes of ground truth objects in our dataset. By placing these priors at every possible location in a feature map, we also account for variety in position.</p>
<p>In defining the priors, the authors specify that –</p>
<ul>
<li>
<p><strong>they will be applied to various low-level and high-level feature maps</strong>, viz. those from <code>conv4_3</code>, <code>conv7</code>, <code>conv8_2</code>, <code>conv9_2</code>, <code>conv10_2</code>, and <code>conv11_2</code>. These are the same feature maps indicated on the figures before.</p>
</li>
<li>
<p><strong>if a prior has a scale <code>s</code>, then its area is equal to that of a square with side <code>s</code></strong>. The largest feature map, <code>conv4_3</code>, will have priors with a scale of <code>0.1</code>, i.e. <code>10%</code> of image's dimensions, while the rest have priors with scales linearly increasing from <code>0.2</code> to <code>0.9</code>. As you can see, larger feature maps have priors with smaller scales and are therefore ideal for detecting smaller objects.</p>
</li>
<li>
<p><strong>At <em>each</em> position on a feature map, there will be priors of various aspect ratios</strong>. All feature maps will have priors with ratios <code>1:1, 2:1, 1:2</code>. The intermediate feature maps of <code>conv7</code>, <code>conv8_2</code>, and <code>conv9_2</code> will <em>also</em> have priors with ratios <code>3:1, 1:3</code>. Moreover, all feature maps will have <em>one extra prior</em> with an aspect ratio of <code>1:1</code> and at a scale that is the geometric mean of the scales of the current and subsequent feature map.</p>
</li>
</ul>
<table>
<thead>
<tr>
<th align="center">Feature Map From</th>
<th align="center">Feature Map Dimensions</th>
<th align="center">Prior Scale</th>
<th align="center">Aspect Ratios</th>
<th align="center">Number of Priors per Position</th>
<th align="center">Total Number of Priors on this Feature Map</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><code>conv4_3</code></td>
<td align="center">38, 38</td>
<td align="center">0.1</td>
<td align="center">1:1, 2:1, 1:2 + an extra prior</td>
<td align="center">4</td>
<td align="center">5776</td>
</tr>
<tr>
<td align="center"><code>conv7</code></td>
<td align="center">19, 19</td>
<td align="center">0.2</td>
<td align="center">1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior</td>
<td align="center">6</td>
<td align="center">2166</td>
</tr>
<tr>
<td align="center"><code>conv8_2</code></td>
<td align="center">10, 10</td>
<td align="center">0.375</td>
<td align="center">1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior</td>
<td align="center">6</td>
<td align="center">600</td>
</tr>
<tr>
<td align="center"><code>conv9_2</code></td>
<td align="center">5, 5</td>
<td align="center">0.55</td>
<td align="center">1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior</td>
<td align="center">6</td>
<td align="center">150</td>
</tr>
<tr>
<td align="center"><code>conv10_2</code></td>
<td align="center">3,  3</td>
<td align="center">0.725</td>
<td align="center">1:1, 2:1, 1:2 + an extra prior</td>
<td align="center">4</td>
<td align="center">36</td>
</tr>
<tr>
<td align="center"><code>conv11_2</code></td>
<td align="center">1, 1</td>
<td align="center">0.9</td>
<td align="center">1:1, 2:1, 1:2 + an extra prior</td>
<td align="center">4</td>
<td align="center">4</td>
</tr>
<tr>
<td align="center"><strong>Grand Total</strong></td>
<td align="center">–</td>
<td align="center">–</td>
<td align="center">–</td>
<td align="center">–</td>
<td align="center"><strong>8732 priors</strong></td>
</tr>
</tbody>
</table>
<p>There are a total of 8732 priors defined for the SSD300!</p>
<h4><a id="user-content-visualizing-priors" class="anchor" aria-hidden="true" href="#visualizing-priors"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Visualizing Priors</h4>
<p>We defined the priors in terms of their <em>scales</em> and <em>aspect ratios</em>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/wh1.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/wh1.jpg" alt="" style="max-width:100%;"></a></p>
<p>Solving these equations yields a prior's dimensions <code>w</code> and <code>h</code>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/wh2.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/wh2.jpg" alt="" style="max-width:100%;"></a></p>
<p>We're now in a position to draw them on their respective feature maps.</p>
<p>For example, let's try to visualize what the priors will look like at the central tile of the feature map from <code>conv9_2</code>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/priors1.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/priors1.jpg" alt="" style="max-width:100%;"></a></p>
<p>The same priors also exist for each of the other tiles.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/priors2.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/priors2.jpg" alt="" style="max-width:100%;"></a></p>
<h4><a id="user-content-predictions-vis-à-vis-priors" class="anchor" aria-hidden="true" href="#predictions-vis-à-vis-priors"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Predictions vis-à-vis Priors</h4>
<p><a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#multibox">Earlier</a>, we said we would use regression to find the coordinates of an object's bounding box. But then, surely, the priors can't represent our final predicted boxes?</p>
<p>They don't.</p>
<p>Again, I would like to reiterate that the priors represent, <em>approximately</em>, the possibilities for prediction.</p>
<p>This means that <strong>we use each prior as an approximate starting point and then find out how much it needs to be adjusted to obtain a more exact prediction for a bounding box</strong>.</p>
<p>So if each predicted bounding box is a slight deviation from a prior, and our goal is to calculate this deviation, we need a way to measure or quantify it.</p>
<p>Consider a cat, its predicted bounding box, and the prior with which the prediction was made.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/ecs1.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/ecs1.PNG" alt="" style="max-width:100%;"></a></p>
<p>Assume they are represented in center-size coordinates, which we are familiar with.</p>
<p>Then –</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/ecs2.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/ecs2.PNG" alt="" style="max-width:100%;"></a></p>
<p>This answers the question we posed at the <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#a-detour">beginning of this section</a>. Considering that each prior is adjusted to obtain a more precise prediction, <strong>these four offsets <code>(g_c_x, g_c_y, g_w, g_h)</code> are the form in which we will regress bounding boxes' coordinates</strong>.</p>
<p>As you can see, each offset is normalized by the corresponding dimension of the prior. This makes sense because a certain offset would be less significant for a larger prior than it would be for a smaller prior.</p>
<h3><a id="user-content-prediction-convolutions" class="anchor" aria-hidden="true" href="#prediction-convolutions"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Prediction convolutions</h3>
<p>Earlier, we earmarked and defined priors for six feature maps of various scales and granularity, viz. those from <code>conv4_3</code>, <code>conv7</code>, <code>conv8_2</code>, <code>conv9_2</code>, <code>conv10_2</code>, and <code>conv11_2</code>.</p>
<p>Then, <strong>for <em>each</em> prior at <em>each</em> location on <em>each</em> feature map</strong>, we want to predict –</p>
<ul>
<li>
<p>the <strong>offsets <code>(g_c_x, g_c_y, g_w, g_h)</code></strong> for a bounding box.</p>
</li>
<li>
<p>a set of <strong><code>n_classes</code> scores</strong> for the bounding box, where <code>n_classes</code> represents the total number of object types (including a <em>background</em> class).</p>
</li>
</ul>
<p>To do this in the simplest manner possible, <strong>we need two convolutional layers for each feature map</strong> –</p>
<ul>
<li>
<p>a <strong><em>localization</em> prediction</strong> convolutional layer with a <code>3,  3</code> kernel evaluating at each location (i.e. with padding and stride of <code>1</code>) with <code>4</code> filters for <em>each</em> prior present at the location.</p>
<p>The <code>4</code> filters for a prior calculate the four encoded offsets <code>(g_c_x, g_c_y, g_w, g_h)</code> for the bounding box predicted from that prior.</p>
</li>
<li>
<p>a <strong><em>class</em> prediction</strong> convolutional layer with a <code>3,  3</code> kernel evaluating at each location (i.e. with padding and stride of <code>1</code>) with <code>n_classes</code> filters for <em>each</em> prior present at the location.</p>
<p>The <code>n_classes</code> filters for a prior calculate a set of <code>n_classes</code> scores for that prior.</p>
</li>
</ul>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/predconv1.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv1.jpg" alt="" style="max-width:100%;"></a></p>
<p>All our filters are applied with a kernel size of <code>3, 3</code>.</p>
<p>We don't really need kernels (or filters) in the same shapes as the priors because the different filters will <em>learn</em> to make predictions with respect to the different prior shapes.</p>
<p>Let's take a look at the <strong>outputs of these convolutions</strong>. Consider again the feature map from <code>conv9_2</code>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/predconv2.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv2.jpg" alt="" style="max-width:100%;"></a></p>
<p>The outputs of the localization and class prediction layers are shown in blue and yellow respectively. You can see that the cross-section (<code>5, 5</code>) remains unchanged.</p>
<p>What we're really interested in is the <em>third</em> dimension, i.e. the channels. These contain the actual predictions.</p>
<p>If you <strong>choose a tile, <em>any</em> tile, in the localization predictions and expand it</strong>, what will you see?</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/predconv3.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv3.jpg" alt="" style="max-width:100%;"></a></p>
<p>Voilà! The channel values at each position of the localization predictions represent the encoded offsets with respect to the priors at that position.</p>
<p>Now, <strong>do the same with the class predictions.</strong> Assume <code>n_classes = 3</code>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/predconv4.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/predconv4.jpg" alt="" style="max-width:100%;"></a></p>
<p>Similar to before, these channels represent the class scores for the priors at that position.</p>
<p>Now that we understand what the predictions for the feature map from <code>conv9_2</code> look like, we can <strong>reshape them into a more amenable form.</strong></p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/reshaping1.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/reshaping1.jpg" alt="" style="max-width:100%;"></a></p>
<p>We have arranged the <code>150</code> predictions serially. To the human mind, this should appear more intuitive.</p>
<p>But let's not stop here. We could do the same for the predictions for <em>all</em> layers and stack them together.</p>
<p>We calculated earlier that there are a total of 8732 priors defined for our model. Therefore, there will be <strong>8732 predicted boxes in encoded-offset form, and 8732 sets of class scores</strong>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/reshaping2.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/reshaping2.jpg" alt="" style="max-width:100%;"></a></p>
<p><strong>This is the final output of the prediction stage.</strong> A stack of boxes, if you will, and estimates for what's in them.</p>
<p>It's all coming together, isn't it? If this is your first rodeo in object detection, I should think there's now a faint light at the end of the tunnel.</p>
<h3><a id="user-content-multibox-loss" class="anchor" aria-hidden="true" href="#multibox-loss"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Multibox loss</h3>
<p>Based on the nature of our predictions, it's easy to see why we might need a unique loss function. Many of us have calculated losses in regression or classification settings before, but rarely, if ever, <em>together</em>.</p>
<p>Obviously, our total loss must be an <strong>aggregate of losses from both types of predictions</strong> – bounding box localizations and class scores.</p>
<p>Then, there are a few questions to be answered –</p>
<blockquote>
<p><em>What loss function will be used for the regressed bounding boxes?</em></p>
</blockquote>
<blockquote>
<p><em>Will we use multiclass cross-entropy for the class scores?</em></p>
</blockquote>
<blockquote>
<p><em>In what ratio will we combine them?</em></p>
</blockquote>
<blockquote>
<p><em>How do we match predicted boxes to their ground truths?</em></p>
</blockquote>
<blockquote>
<p><em>We have 8732 predictions! Won't most of these contain no object? Do we even consider them?</em></p>
</blockquote>
<p>Phew. Let's get to work.</p>
<h4><a id="user-content-matching-predictions-to-ground-truths" class="anchor" aria-hidden="true" href="#matching-predictions-to-ground-truths"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Matching predictions to ground truths</h4>
<p>Remember, the nub of any supervised learning algorithm is that <strong>we need to be able to match predictions to their ground truths</strong>. This is tricky since object detection is more open-ended than the average learning task.</p>
<p>For the model to learn <em>anything</em>, we'd need to structure the problem in a way that allows for comparisions between our predictions and the objects actually present in the image.</p>
<p>Priors enable us to do exactly this!</p>
<ul>
<li>
<p><strong>Find the Jaccard overlaps</strong> between the 8732 priors and <code>N</code> ground truth objects. This will be a tensor of size <code>8732, N</code>.</p>
</li>
<li>
<p><strong>Match</strong> each of the 8732 priors to the object with which it has the greatest overlap.</p>
</li>
<li>
<p>If a prior is matched with an object with a <strong>Jaccard overlap of less than <code>0.5</code></strong>, then it cannot be said to "contain" the object, and is therefore a <strong><em>negative</em> match</strong>. Considering we have thousands of priors, most priors will test negative for an object.</p>
</li>
<li>
<p>On the other hand, a handful of priors will actually <strong>overlap significantly (greater than <code>0.5</code>)</strong> with an object, and can be said to "contain" that object. These are <strong><em>positive</em> matches</strong>.</p>
</li>
<li>
<p>Now that we have <strong>matched each of the 8732 priors to a ground truth</strong>, we have, in effect, also <strong>matched the corresponding 8732 predictions to a ground truth</strong>.</p>
</li>
</ul>
<p>Let's reproduce this logic with an example.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/matching1.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/matching1.PNG" alt="" style="max-width:100%;"></a></p>
<p>For convenience, we will assume there are just seven priors, shown in red. The ground truths are in yellow – there are three actual objects in this image.</p>
<p>Following the steps outlined earlier will yield the following matches –</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/matching2.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/matching2.jpg" alt="" style="max-width:100%;"></a></p>
<p>Now, <strong>each prior has a match</strong>, positive or negative. By extension, <strong>each prediction has a match</strong>, positive or negative.</p>
<p>Predictions that are positively matched with an object now have ground truth coordinates that will serve as <strong>targets for localization</strong>, i.e. in the <em>regression</em> task. Naturally, there will be no target coordinates for negative matches.</p>
<p>All predictions have a ground truth label, which is either the type of object if it is a positive match or a <em>background</em> class if it is a negative match. These are used as <strong>targets for class prediction</strong>, i.e. the <em>classification</em> task.</p>
<h4><a id="user-content-localization-loss" class="anchor" aria-hidden="true" href="#localization-loss"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Localization loss</h4>
<p>We have <strong>no ground truth coordinates for the negative matches</strong>. This makes perfect sense. Why train the model to draw boxes around empty space?</p>
<p>Therefore, the localization loss is computed only on how accurately we regress positively matched predicted boxes to the corresponding ground truth coordinates.</p>
<p>Since we predicted localization boxes in the form of offsets <code>(g_c_x, g_c_y, g_w, g_h)</code>, we would also need to encode the ground truth coordinates accordingly before we calculate the loss.</p>
<p>The localization loss is the averaged <strong>Smooth L1</strong> loss between the encoded offsets of positively matched localization boxes and their ground truths.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/locloss.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/locloss.jpg" alt="" style="max-width:100%;"></a></p>
<h4><a id="user-content-confidence-loss" class="anchor" aria-hidden="true" href="#confidence-loss"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Confidence loss</h4>
<p>Every prediction, no matter positive or negative, has a ground truth label associated with it. It is important that the model recognizes both objects and a lack of them.</p>
<p>However, considering that there are usually only a handful of objects in an image, <strong>the vast majority of the thousands of predictions we made do not actually contain an object</strong>. As Walter White would say, <em>tread lightly</em>. If the negative matches overwhelm the 
