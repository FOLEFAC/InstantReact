# InstantReact
This repo contains code for the <b>InstantReact</b> project. This project helps people and organizations with <b>CCTV cameras</b> take quick decisions automatically depending on what the Camera sees.
In this project, we shall use <b>Facebook's PyTorch, Wit.ai, ReactJs and Docusaurus</b> to train and save a deep learning model which does Video Captioning, to extract sound from a video footage, do a demo app using <b>ONNX.js</b> and document a library which we shall develop and which will permit other <b> developers</b> easily use their own data and achieve great results!!!

<p>Basic knowledge of PyTorch, convolutional Neural Networks, Recurrent Neural Networks and Long Short Term Memories is assumed.</p>
<p>If you're new to PyTorch, first read <a href="https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html" rel="nofollow">Deep Learning with PyTorch: A 60 Minute Blitz</a> and <a href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html" rel="nofollow">Learning PyTorch with Examples</a>.</p>
<p>Questions, suggestions, or corrections can be posted as issues.</p>
<p>I'm using <code>PyTorch 1.6.0</code> in <code>Python 3.7.4</code>.</p>

<h1><a id="user-content-contents" class="anchor" aria-hidden="true" href="#contents"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Contents</h1>
<p><a href="https://github.com/FOLEFAC/InstantReact#objective"><em><strong>Objective</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#concepts"><em><strong>Concepts</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#overview"><em><strong>Overview</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#implementation"><em><strong>Implementation</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#training"><em><strong>Training</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#evaluation"><em><strong>Evaluation</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#inference"><em><strong>Inference</strong></em></a></p>
<p><a href="https://github.com/FOLEFAC/InstantReact#faqs"><em><strong>Frequently Asked Questions</strong></em></a></p>
<h1><a id="user-content-objective" class="anchor" aria-hidden="true" href="#objective"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Objective</h1>
<p><strong>To build a system, which can extract useful information from a video like actions and sounds.</strong> <br> This will permit security agents be more efficient as the computer can be trained to automatically see what is contained in the video, and alert them so that they can take quick and instant measures. Also we shall see how to build a package and quickly document it so that other developers can make use of it with ease. Then finally we shall see how to deploy these models in production using ONNX and how to do a demo app in ReactJs.
 <br>  Hopefully after going through this tutorial, you shall learn how to use the InstantReact Package and also how to create and document yours :)</p>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="https://github.com/FOLEFAC/InstantReact/blob/main/fighting.gif"><img src="https://github.com/FOLEFAC/InstantReact/blob/main/fighting.gif" style="max-width:100%;"></a>
</p>
<hr>
<h1><a id="user-content-concepts" class="anchor" aria-hidden="true" href="#concepts"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Concepts</h1>
<ul>
<li>
 
 <p>Upon selecting candidates for each <em>non-background</em> class,</p>
<ul>
<li>
<p>Arrange candidates for this class in order of decreasing likelihood.</p>
</li>
<li>
<p>Consider the candidate with the highest score. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than, say, <code>0.5</code> with this candidate.</p>
</li>
<li>
<p>Consider the next highest-scoring candidate still remaining in the pool. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than <code>0.5</code> with this candidate.</p>
</li>
<li>
<p>Repeat until you run through the entire sequence of candidates.</p>
</li>
</ul>
</li>
</ul>
<p>The end result is that you will have just a single box – the very best one – for each object in the image.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/nms4.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/nms4.PNG" alt="" style="max-width:100%;"></a></p>
<p>Non-Maximum Suppression is quite crucial for obtaining quality detections.</p>

</li>
<li>
<p><strong>Single-Shot Detection</strong>. Earlier architectures for object detection consisted of two distinct stages – a region proposal network that performs object localization and a classifier for detecting the types of objects in the proposed regions. Computationally, these can be very expensive and therefore ill-suited for real-world, real-time applications. Single-shot models encapsulate both localization and detection tasks in a single forward sweep of the network, resulting in significantly faster detections while deployable on lighter hardware.</p>
</li>
<li>
<p><strong>Multiscale Feature Maps</strong>. In image classification tasks, we base our predictions on the final convolutional feature map – the smallest but deepest representation of the original image. In object detection, feature maps from intermediate convolutional layers can also be <em>directly</em> useful because they represent the original image at different scales. Therefore, a fixed-size filter operating on different feature maps will be able to detect objects of various sizes.</p>
</li>
<li>
<p><strong>Priors</strong>. These are pre-computed boxes defined at specific positions on specific feature maps, with specific aspect ratios and scales. They are carefully chosen to match the characteristics of objects' bounding boxes (i.e. the ground truths) in the dataset.</p>
</li>
<li>
<p><strong>Multibox</strong>. This is <a href="https://arxiv.org/abs/1312.2249" rel="nofollow">a technique</a> that formulates predicting an object's bounding box as a <em>regression</em> problem, wherein a detected object's coordinates are regressed to its ground truth's coordinates. In addition, for each predicted box, scores are generated for various object types. Priors serve as feasible starting points for predictions because they are modeled on the ground truths. Therefore, there will be as many predicted boxes as there are priors, most of whom will contain no object.</p>
</li>
<li>
<p><strong>Hard Negative Mining</strong>. This refers to explicitly choosing the most egregious false positives predicted by a model and forcing it to learn from these examples. In other words, we are mining only those negatives that the model found <em>hardest</em> to identify correctly. In the context of object detection, where the vast majority of predicted boxes do not contain an object, this also serves to reduce the negative-positive imbalance.</p>
</li>
<li>
<p><strong>Non-Maximum Suppression</strong>. At any given location, multiple priors can overlap significantly. Therefore, predictions arising out of these priors could actually be duplicates of the same object. Non-Maximum Suppression (NMS) is a means to remove redundant predictions by suppressing all but the one with the maximum score.</p>
</li>
</ul>
<h1><a id="user-content-overview" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Overview</h1>
<p>In this section, I will present an overview of this model. If you're already familiar with it, you can skip straight to the <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#implementation">Implementation</a> section or the commented code.</p>
<p>As we proceed, you will notice that there's a fair bit of engineering that's resulted in the SSD's very specific structure and formulation. Don't worry if some aspects of it seem contrived or unspontaneous at first. Remember, it's built upon <em>years</em> of (often empirical) research in this field.</p>
<h3><a id="user-content-some-definitions" class="anchor" aria-hidden="true" href="#some-definitions"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Some definitions</h3>
<p>A box is a box. A <em>bounding</em> box is a box that wraps around an object i.e. represents its bounds.</p>
<p>In this tutorial, we will encounter both types – just boxes and bounding boxes. But all boxes are represented on images and we need to be able to measure their positions, shapes, sizes, and other properties.</p>
<h4><a id="user-content-boundary-coordinates" class="anchor" aria-hidden="true" href="#boundary-coordinates"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Boundary coordinates</h4>
<p>The most obvious way to represent a box is by the pixel coordinates of the <code>x</code> and <code>y</code> lines that constitute its boundaries.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/bc1.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/bc1.PNG" alt="" style="max-width:100%;"></a></p>
<p>The boundary coordinates of a box are simply <strong><code>(x_min, y_min, x_max, y_max)</code></strong>.</p>
<p>But pixel values are next to useless if we don't know the actual dimensions of the image.
A better way would be to represent all coordinates is in their <em>fractional</em> form.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/bc2.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/bc2.PNG" alt="" style="max-width:100%;"></a></p>
<p>Now the coordinates are size-invariant and all boxes across all images are measured on the same scale.</p>
<h4><a id="user-content-center-size-coordinates" class="anchor" aria-hidden="true" href="#center-size-coordinates"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Center-Size coordinates</h4>
<p>This is a more explicit way of representing a box's position and dimensions.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/cs.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/cs.PNG" alt="" style="max-width:100%;"></a></p>
<p>The center-size coordinates of a box are <strong><code>(c_x, c_y, w, h)</code></strong>.</p>
<p>In the code, you will find that we routinely use both coordinate systems depending upon their suitability for the task, and <em>always</em> in their fractional forms.</p>
<h4><a id="user-content-jaccard-index" class="anchor" aria-hidden="true" href="#jaccard-index"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Jaccard Index</h4>
<p>The Jaccard Index or Jaccard Overlap or Intersection-over-Union (IoU) measure the <strong>degree or extent to which two boxes overlap</strong>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/jaccard.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/jaccard.jpg" alt="" style="max-width:100%;"></a></p>
<p>An IoU of <code>1</code> implies they are the <em>same</em> box, while a value of <code>0</code> indicates they're mutually exclusive spaces.</p>
<p>It's a simple metric, but also one that finds many applications in our model.</p>
<h3><a id="user-content-multibox" class="anchor" aria-hidden="true" href="#multibox"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Multibox</h3>
<p>Multibox is a technique for detecting objects where a prediction consists of two components –</p>
<ul>
<li>
<p><strong>Coordinates of a box that may or may not contain an object</strong>. This is a <em>regression</em> task.</p>
</li>
<li>
<p><strong>Scores for various object types for this box</strong>, including a <em>background</em> class which implies there is no object in the box. This is a <em>classification</em> task.</p>
</li>
</ul>
<h3><a id="user-content-single-shot-detector-ssd" class="anchor" aria-hidden="true" href="#single-shot-detector-ssd"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Single Shot Detector (SSD)</h3>
<p>The SSD is a purely convolutional neural network (CNN) that we can organize into three parts –</p>
<ul>
<li>
<p><strong>Base convolutions</strong> derived from an existing image classification architecture that will provide lower-level feature maps.</p>
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
<p>However, considering that there are usually only a handful of objects in an image, <strong>the vast majority of the thousands of predictions we made do not actually contain an object</strong>. As Walter White would say, <em>tread lightly</em>. If the negative matches overwhelm the positive ones, we will end up with a model that is less likely to detect objects because, more often than not, it is taught to detect the <em>background</em> class.</p>
<p>The solution may be obvious – limit the number of negative matches that will be evaluated in the loss function. But how do we choose?</p>
<p>Well, why not use the ones that the model was most <em>wrong</em> about? In other words, only use those predictions where the model found it hardest to recognize that there are no objects. This is called <strong>Hard Negative Mining</strong>.</p>
<p>The number of hard negatives we will use, say <code>N_hn</code>, is usually a fixed multiple of the number of positive matches for this image. In this particular case, the authors have decided to use three times as many hard negatives, i.e. <code>N_hn = 3 * N_p</code>. The hardest negatives are discovered by finding the Cross Entropy loss for each negatively matched prediction and choosing those with top <code>N_hn</code> losses.</p>
<p>Then, the confidence loss is simply the sum of the <strong>Cross Entropy</strong> losses among the positive and hard negative matches.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/confloss.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/confloss.jpg" alt="" style="max-width:100%;"></a></p>
<p>You will notice that it is averaged by the number of positive matches.</p>
<h4><a id="user-content-total-loss" class="anchor" aria-hidden="true" href="#total-loss"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Total loss</h4>
<p>The <strong>Multibox loss is the aggregate of the two losses</strong>, combined in a ratio <code>α</code>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/totalloss.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/totalloss.jpg" alt="" style="max-width:100%;"></a></p>
<p>In general, we needn't decide on a value for <code>α</code>. It could be a learnable parameter.</p>
<p>For the SSD, however, the authors simply use <code>α = 1</code>, i.e. add the two losses. We'll take it!</p>
<h3><a id="user-content-processing-predictions" class="anchor" aria-hidden="true" href="#processing-predictions"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Processing predictions</h3>
<p>After the model is trained, we can apply it to images. However, the predictions are still in their raw form – two tensors containing the offsets and class scores for 8732 priors. These would need to be processed to <strong>obtain final, human-interpretable bounding boxes with labels.</strong></p>
<p>This entails the following –</p>
<ul>
<li>
<p>We have 8732 predicted boxes represented as offsets <code>(g_c_x, g_c_y, g_w, g_h)</code> from their respective priors. Decode them to boundary coordinates, which are actually directly interpretable.</p>
</li>
<li>
<p>Then, for each <em>non-background</em> class,</p>
<ul>
<li>
<p>Extract the scores for this class for each of the 8732 boxes.</p>
</li>
<li>
<p>Eliminate boxes that do not meet a certain threshold for this score.</p>
</li>
<li>
<p>The remaining (uneliminated) boxes are candidates for this particular class of object.</p>
</li>
</ul>
</li>
</ul>
<p>At this point, if you were to draw these candidate boxes on the original image, you'd see <strong>many highly overlapping boxes that are obviously redundant</strong>. This is because it's extremely likely that, from the thousands of priors at our disposal, more than one prediction corresponds to the same object.</p>
<p>For instance, consider the image below.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/nms1.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/nms1.PNG" alt="" style="max-width:100%;"></a></p>
<p>There's clearly only three objects in it – two dogs and a cat. But according to the model, there are <em>three</em> dogs and <em>two</em> cats.</p>
<p>Mind you, this is just a mild example. It could really be much, much worse.</p>
<p>Now, to you, it may be obvious which boxes are referring to the same object. This is because your mind can process that certain boxes coincide significantly with each other and a specific object.</p>
<p>In practice, how would this be done?</p>
<p>First, <strong>line up the candidates for each class in terms of how <em>likely</em> they are</strong>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/nms2.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/nms2.PNG" alt="" style="max-width:100%;"></a></p>
<p>We've sorted them by their scores.</p>
<p>The next step is to find which candidates are redundant. We already have a tool at our disposal to judge how much two boxes have in common with each other – the Jaccard overlap.</p>
<p>So, if we were to <strong>draw up the Jaccard similarities between all the candidates in a given class</strong>, we could evaluate each pair and <strong>if found to overlap significantly, keep only the <em>more likely</em> candidate</strong>.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/nms3.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/nms3.jpg" alt="" style="max-width:100%;"></a></p>
<p>Thus, we've eliminated the rogue candidates – one of each animal.</p>
<p>This process is called <strong>Non-Maximum Suppression (NMS)</strong> because when multiple candidates are found to overlap significantly with each other such that they could be referencing the same object, <strong>we suppress all but the one with the maximum score</strong>.</p>
<p>Algorithmically, it is carried out as follows –</p>
<ul>
<li>
<p>Upon selecting candidates for each <em>non-background</em> class,</p>
<ul>
<li>
<p>Arrange candidates for this class in order of decreasing likelihood.</p>
</li>
<li>
<p>Consider the candidate with the highest score. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than, say, <code>0.5</code> with this candidate.</p>
</li>
<li>
<p>Consider the next highest-scoring candidate still remaining in the pool. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than <code>0.5</code> with this candidate.</p>
</li>
<li>
<p>Repeat until you run through the entire sequence of candidates.</p>
</li>
</ul>
</li>
</ul>
<p>The end result is that you will have just a single box – the very best one – for each object in the image.</p>
<p><a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/nms4.PNG"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/nms4.PNG" alt="" style="max-width:100%;"></a></p>
<p>Non-Maximum Suppression is quite crucial for obtaining quality detections.</p>
<p>Happily, it's also the final step.</p>
<h1><a id="user-content-implementation" class="anchor" aria-hidden="true" href="#implementation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Implementation</h1>
<p>The sections below briefly describe the implementation.</p>
<p>They are meant to provide some context, but <strong>details are best understood directly from the code</strong>, which is quite heavily commented.</p>
<h3><a id="user-content-dataset" class="anchor" aria-hidden="true" href="#dataset"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Dataset</h3>
<p>We will use Pascal Visual Object Classes (VOC) data from the years 2007 and 2012.</p>
<h4><a id="user-content-description" class="anchor" aria-hidden="true" href="#description"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Description</h4>
<p>This data contains images with twenty different types of objects.</p>
<div class="highlight highlight-source-python"><pre>{<span class="pl-s">'aeroplane'</span>, <span class="pl-s">'bicycle'</span>, <span class="pl-s">'bird'</span>, <span class="pl-s">'boat'</span>, <span class="pl-s">'bottle'</span>, <span class="pl-s">'bus'</span>, <span class="pl-s">'car'</span>, <span class="pl-s">'cat'</span>, <span class="pl-s">'chair'</span>, <span class="pl-s">'cow'</span>, <span class="pl-s">'diningtable'</span>, <span class="pl-s">'dog'</span>, <span class="pl-s">'horse'</span>, <span class="pl-s">'motorbike'</span>, <span class="pl-s">'person'</span>, <span class="pl-s">'pottedplant'</span>, <span class="pl-s">'sheep'</span>, <span class="pl-s">'sofa'</span>, <span class="pl-s">'train'</span>, <span class="pl-s">'tvmonitor'</span>}</pre></div>
<p>Each image can contain one or more ground truth objects.</p>
<p>Each object is represented by –</p>
<ul>
<li>
<p>a bounding box in absolute boundary coordinates</p>
</li>
<li>
<p>a label (one of the object types mentioned above)</p>
</li>
<li>
<p>a perceived detection difficulty (either <code>0</code>, meaning <em>not difficult</em>, or <code>1</code>, meaning <em>difficult</em>)</p>
</li>
</ul>
<h4><a id="user-content-download" class="anchor" aria-hidden="true" href="#download"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Download</h4>
<p>Specfically, you will need to download the following VOC datasets –</p>
<ul>
<li>
<p><a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" rel="nofollow">2007 <em>trainval</em></a> (460MB)</p>
</li>
<li>
<p><a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" rel="nofollow">2012 <em>trainval</em></a> (2GB)</p>
</li>
<li>
<p><a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" rel="nofollow">2007 <em>test</em></a> (451MB)</p>
</li>
</ul>
<p>Consistent with the paper, the two <em>trainval</em> datasets are to be used for training, while the VOC 2007 <em>test</em> will serve as our test data.</p>
<p>Make sure you extract both the VOC 2007 <em>trainval</em> and 2007 <em>test</em> data to the same location, i.e. merge them.</p>
<h3><a id="user-content-inputs-to-model" class="anchor" aria-hidden="true" href="#inputs-to-model"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Inputs to model</h3>
<p>We will need three inputs.</p>
<h4><a id="user-content-images" class="anchor" aria-hidden="true" href="#images"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Images</h4>
<p>Since we're using the SSD300 variant, the images would need to be sized at <code>300, 300</code> pixels and in the RGB format.</p>
<p>Remember, we're using a VGG-16 base pretrained on ImageNet that is already available in PyTorch's <code>torchvision</code> module. <a href="https://pytorch.org/docs/master/torchvision/models.html" rel="nofollow">This page</a> details the preprocessing or transformation we would need to perform in order to use this model – pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.</p>
<div class="highlight highlight-source-python"><pre><span class="pl-s1">mean</span> <span class="pl-c1">=</span> [<span class="pl-c1">0.485</span>, <span class="pl-c1">0.456</span>, <span class="pl-c1">0.406</span>]
<span class="pl-s1">std</span> <span class="pl-c1">=</span> [<span class="pl-c1">0.229</span>, <span class="pl-c1">0.224</span>, <span class="pl-c1">0.225</span>]</pre></div>
<p>Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.</p>
<p>Therefore, <strong>images fed to the model must be a <code>Float</code> tensor of dimensions <code>N, 3, 300, 300</code></strong>, and must be normalized by the aforesaid mean and standard deviation. <code>N</code> is the batch size.</p>
<h4><a id="user-content-objects-bounding-boxes" class="anchor" aria-hidden="true" href="#objects-bounding-boxes"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Objects' Bounding Boxes</h4>
<p>We would need to supply, for each image, the bounding boxes of the ground truth objects present in it in fractional boundary coordinates <code>(x_min, y_min, x_max, y_max)</code>.</p>
<p>Since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the bounding boxes for the entire batch of <code>N</code> images.</p>
<p>Therefore, <strong>ground truth bounding boxes fed to the model must be a list of length <code>N</code>, where each element of the list is a <code>Float</code> tensor of dimensions <code>N_o, 4</code></strong>, where <code>N_o</code> is the number of objects present in that particular image.</p>
<h4><a id="user-content-objects-labels" class="anchor" aria-hidden="true" href="#objects-labels"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Objects' Labels</h4>
<p>We would need to supply, for each image, the labels of the ground truth objects present in it.</p>
<p>Each label would need to be encoded as an integer from <code>1</code> to <code>20</code> representing the twenty different object types. In addition, we will add a <em>background</em> class with index <code>0</code>, which indicates the absence of an object in a bounding box. (But naturally, this label will not actually be used for any of the ground truth objects in the dataset.)</p>
<p>Again, since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the labels for the entire batch of <code>N</code> images.</p>
<p>Therefore, <strong>ground truth labels fed to the model must be a list of length <code>N</code>, where each element of the list is a <code>Long</code> tensor of dimensions <code>N_o</code></strong>, where <code>N_o</code> is the number of objects present in that particular image.</p>
<h3><a id="user-content-data-pipeline" class="anchor" aria-hidden="true" href="#data-pipeline"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Data pipeline</h3>
<p>As you know, our data is divided into <em>training</em> and <em>test</em> splits.</p>
<h4><a id="user-content-parse-raw-data" class="anchor" aria-hidden="true" href="#parse-raw-data"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Parse raw data</h4>
<p>See <code>create_data_lists()</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py"><code>utils.py</code></a>.</p>
<p>This parses the data downloaded and saves the following files –</p>
<ul>
<li>
<p>A <strong>JSON file for each split with a list of the absolute filepaths of <code>I</code> images</strong>, where <code>I</code> is the total number of images in the split.</p>
</li>
<li>
<p>A <strong>JSON file for each split with a list of <code>I</code> dictionaries containing ground truth objects, i.e. bounding boxes in absolute boundary coordinates, their encoded labels, and perceived detection difficulties</strong>. The <code>i</code>th dictionary in this list will contain the objects present in the <code>i</code>th image in the previous JSON file.</p>
</li>
<li>
<p>A <strong>JSON file which contains the <code>label_map</code></strong>, the label-to-index dictionary with which the labels are encoded in the previous JSON file. This dictionary is also available in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py"><code>utils.py</code></a> and directly importable.</p>
</li>
</ul>
<h4><a id="user-content-pytorch-dataset" class="anchor" aria-hidden="true" href="#pytorch-dataset"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>PyTorch Dataset</h4>
<p>See <code>PascalVOCDataset</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py"><code>datasets.py</code></a>.</p>
<p>This is a subclass of PyTorch <a href="https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset" rel="nofollow"><code>Dataset</code></a>, used to <strong>define our training and test datasets.</strong> It needs a <code>__len__</code> method defined, which returns the size of the dataset, and a <code>__getitem__</code> method which returns the <code>i</code>th image, bounding boxes of the objects in this image, and labels for the objects in this image, using the JSON files we saved earlier.</p>
<p>You will notice that it also returns the perceived detection difficulties of each of these objects, but these are not actually used in training the model. They are required only in the <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#evaluation">Evaluation</a> stage for computing the Mean Average Precision (mAP) metric. We also have the option of filtering out <em>difficult</em> objects entirely from our data to speed up training at the cost of some accuracy.</p>
<p>Additionally, inside this class, <strong>each image and the objects in them are subject to a slew of transformations</strong> as described in the paper and outlined below.</p>
<h4><a id="user-content-data-transforms" class="anchor" aria-hidden="true" href="#data-transforms"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Data Transforms</h4>
<p>See <code>transform()</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py"><code>utils.py</code></a>.</p>
<p>This function applies the following transformations to the images and the objects in them –</p>
<ul>
<li>
<p>Randomly <strong>adjust brightness, contrast, saturation, and hue</strong>, each with a 50% chance and in random order.</p>
</li>
<li>
<p>With a 50% chance, <strong>perform a <em>zoom out</em> operation</strong> on the image. This helps with learning to detect small objects. The zoomed out image must be between <code>1</code> and <code>4</code> times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.</p>
</li>
<li>
<p>Randomly crop image, i.e. <strong>perform a <em>zoom in</em> operation.</strong> This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between <code>0.3</code> and <code>1</code> times the original dimensions. The aspect ratio is to be between <code>0.5</code> and <code>2</code>. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either <code>0</code>, <code>0.1</code>, <code>0.3</code>, <code>0.5</code>, <code>0.7</code>, or <code>0.9</code>, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.</p>
</li>
<li>
<p>With a 50% chance, <strong>horizontally flip</strong> the image.</p>
</li>
<li>
<p><strong>Resize</strong> the image to <code>300, 300</code> pixels. This is a requirement of the SSD300.</p>
</li>
<li>
<p>Convert all boxes from <strong>absolute to fractional boundary coordinates.</strong> At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.</p>
</li>
<li>
<p><strong>Normalize</strong> the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.</p>
</li>
</ul>
<p>As mentioned in the paper, these transformations play a crucial role in obtaining the stated results.</p>
<h4><a id="user-content-pytorch-dataloader" class="anchor" aria-hidden="true" href="#pytorch-dataloader"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>PyTorch DataLoader</h4>
<p>The <code>Dataset</code> described above, <code>PascalVOCDataset</code>, will be used by a PyTorch <a href="https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader" rel="nofollow"><code>DataLoader</code></a> in <code>train.py</code> to <strong>create and feed batches of data to the model</strong> for training or evaluation.</p>
<p>Since the number of objects vary across different images, their bounding boxes, labels, and difficulties cannot simply be stacked together in the batch. There would be no way of knowing which objects belong to which image.</p>
<p>Instead, we need to <strong>pass a collating function to the <code>collate_fn</code> argument</strong>, which instructs the <code>DataLoader</code> about how it should combine these varying size tensors. The simplest option would be to use Python lists.</p>
<h3><a id="user-content-base-convolutions" class="anchor" aria-hidden="true" href="#base-convolutions"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Base Convolutions</h3>
<p>See <code>VGGBase</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py"><code>model.py</code></a>.</p>
<p>Here, we <strong>create and apply base convolutions.</strong></p>
<p>The layers are initialized with parameters from a pretrained VGG-16 with the <code>load_pretrained_layers()</code> method.</p>
<p>We're especially interested in the lower-level feature maps that result from <code>conv4_3</code> and <code>conv7</code>, which we return for use in subsequent stages.</p>
<h3><a id="user-content-auxiliary-convolutions-1" class="anchor" aria-hidden="true" href="#auxiliary-convolutions-1"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Auxiliary Convolutions</h3>
<p>See <code>AuxiliaryConvolutions</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py"><code>model.py</code></a>.</p>
<p>Here, we <strong>create and apply auxiliary convolutions.</strong></p>
<p>Use a <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_uniform_" rel="nofollow">uniform Xavier initialization</a> for the parameters of these layers.</p>
<p>We're especially interested in the higher-level feature maps that result from <code>conv8_2</code>, <code>conv9_2</code>, <code>conv10_2</code> and <code>conv11_2</code>, which we return for use in subsequent stages.</p>
<h3><a id="user-content-prediction-convolutions-1" class="anchor" aria-hidden="true" href="#prediction-convolutions-1"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Prediction Convolutions</h3>
<p>See <code>PredictionConvolutions</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py"><code>model.py</code></a>.</p>
<p>Here, we <strong>create and apply localization and class prediction convolutions</strong> to the feature maps from <code>conv4_3</code>, <code>conv7</code>, <code>conv8_2</code>, <code>conv9_2</code>, <code>conv10_2</code> and <code>conv11_2</code>.</p>
<p>These layers are initialized in a manner similar to the auxiliary convolutions.</p>
<p>We also <strong>reshape the resulting prediction maps and stack them</strong> as discussed. Note that reshaping in PyTorch is only possible if the original tensor is stored in a <a href="https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous" rel="nofollow">contiguous</a> chunk of memory.</p>
<p>As expected, the stacked localization and class predictions will be of dimensions <code>8732, 4</code> and <code>8732, 21</code> respectively.</p>
<h3><a id="user-content-putting-it-all-together" class="anchor" aria-hidden="true" href="#putting-it-all-together"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Putting it all together</h3>
<p>See <code>SSD300</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py"><code>model.py</code></a>.</p>
<p>Here, the <strong>base, auxiliary, and prediction convolutions are combined</strong> to form the SSD.</p>
<p>There is a small detail here – the lowest level features, i.e. those from <code>conv4_3</code>, are expected to be on a significantly different numerical scale compared to its higher-level counterparts. Therefore, the authors recommend L2-normalizing and then rescaling <em>each</em> of its channels by a learnable value.</p>
<h3><a id="user-content-priors-1" class="anchor" aria-hidden="true" href="#priors-1"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Priors</h3>
<p>See <code>create_prior_boxes()</code> under <code>SSD300</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py"><code>model.py</code></a>.</p>
<p>This function <strong>creates the priors in center-size coordinates</strong> as defined for the feature maps from <code>conv4_3</code>, <code>conv7</code>, <code>conv8_2</code>, <code>conv9_2</code>, <code>conv10_2</code> and <code>conv11_2</code>, <em>in that order</em>. Furthermore, for each feature map, we create the priors at each tile by traversing it row-wise.</p>
<p>This ordering of the 8732 priors thus obtained is very important because it needs to match the order of the stacked predictions.</p>
<h3><a id="user-content-multibox-loss-1" class="anchor" aria-hidden="true" href="#multibox-loss-1"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Multibox Loss</h3>
<p>See <code>MultiBoxLoss</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py"><code>model.py</code></a>.</p>
<p>Two empty tensors are created to store localization and class prediction targets, i.e. <em>ground truths</em>, for the 8732 predicted boxes in each image.</p>
<p>We <strong>find the ground truth object with the maximum Jaccard overlap for each prior</strong>, which is stored in <code>object_for_each_prior</code>.</p>
<p>We want to avoid the rare situation where not all of the ground truth objects have been matched. Therefore, we also <strong>find the prior with the maximum overlap for each ground truth object</strong>, stored in <code>prior_for_each_object</code>. We explicitly add these matches to <code>object_for_each_prior</code> and artificially set their overlaps to a value above the threshold so they are not eliminated.</p>
<p>Based on the matches in <code>object_for_each prior</code>, we set the corresponding labels, i.e. <strong>targets for class prediction</strong>, to each of the 8732 priors. For those priors that don't overlap significantly with their matched objects, the label is set to <em>background</em>.</p>
<p>Also, we encode the coordinates of the 8732 matched objects in <code>object_for_each prior</code> in offset form <code>(g_c_x, g_c_y, g_w, g_h)</code> with respect to these priors, to form the <strong>targets for localization</strong>. Not all of these 8732 localization targets are meaningful. As we discussed earlier, only the predictions arising from the non-background priors will be regressed to their targets.</p>
<p>The <strong>localization loss</strong> is the <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss" rel="nofollow">Smooth L1 loss</a> over the positive matches.</p>
<p>Perform Hard Negative Mining – rank class predictions matched to <em>background</em>, i.e. negative matches, by their individual <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss" rel="nofollow">Cross Entropy losses</a>. The <strong>confidence loss</strong> is the Cross Entropy loss over the positive matches and the hardest negative matches. Nevertheless, it is averaged only by the number of positive matches.</p>
<p>The <strong>Multibox Loss is the aggregate of these two losses</strong>, combined in the ratio <code>α</code>. In our case, they are simply being added because <code>α = 1</code>.</p>
<h1><a id="user-content-training" class="anchor" aria-hidden="true" href="#training"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Training</h1>
<p>Before you begin, make sure to save the required data files for training and evaluation. To do this, run the contents of <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/create_data_lists.py"><code>create_data_lists.py</code></a> after pointing it to the <code>VOC2007</code> and <code>VOC2012</code> folders in your <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#download">downloaded data</a>.</p>
<p>See <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py"><code>train.py</code></a>.</p>
<p>The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you need to.</p>
<p>To <strong>train your model from scratch</strong>, run this file –</p>
<p><code>python train.py</code></p>
<p>To <strong>resume training at a checkpoint</strong>, point to the corresponding file with the <code>checkpoint</code> parameter at the beginning of the code.</p>
<h3><a id="user-content-remarks" class="anchor" aria-hidden="true" href="#remarks"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Remarks</h3>
<p>In the paper, they recommend using <strong>Stochastic Gradient Descent</strong> in batches of <code>32</code> images, with an initial learning rate of <code>1e−3</code>, momentum of <code>0.9</code>, and <code>5e-4</code> weight decay.</p>
<p>I ended up using a batch size of <code>8</code> images for increased stability. If you find that your gradients are exploding, you could reduce the batch size, like I did, or clip gradients.</p>
<p>The authors also doubled the learning rate for bias parameters. As you can see in the code, this is easy do in PyTorch, by passing <a href="https://pytorch.org/docs/stable/optim.html#per-parameter-options" rel="nofollow">separate groups of parameters</a> to the <code>params</code> argument of its <a href="https://pytorch.org/docs/stable/optim.html#torch.optim.SGD" rel="nofollow">SGD optimizer</a>.</p>
<p>The paper recommends training for 80000 iterations at the initial learning rate. Then, it is decayed by 90% (i.e. to a tenth) for an additional 20000 iterations, <em>twice</em>. With the paper's batch size of <code>32</code>, this means that the learning rate is decayed by 90% once after the 154th epoch and once more after the 193th epoch, and training is stopped after 232 epochs. I followed this schedule.</p>
<p>On a TitanX (Pascal), each epoch of training required about 6 minutes.</p>
<h3><a id="user-content-model-checkpoint" class="anchor" aria-hidden="true" href="#model-checkpoint"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Model checkpoint</h3>
<p>You can download this pretrained model <a href="https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe" rel="nofollow">here</a>.</p>
<p>Note that this checkpoint should be <a href="https://pytorch.org/docs/stable/torch.html?#torch.load" rel="nofollow">loaded directly with PyTorch</a> for evaluation or inference – see below.</p>
<h1><a id="user-content-evaluation" class="anchor" aria-hidden="true" href="#evaluation"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Evaluation</h1>
<p>See <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py"><code>eval.py</code></a>.</p>
<p>The data-loading and checkpoint parameters for evaluating the model are at the beginning of the file, so you can easily check or modify them should you wish to.</p>
<p>To begin evaluation, simply run the <code>evaluate()</code> function with the data-loader and model checkpoint. <strong>Raw predictions for each image in the test set are obtained and parsed</strong> with the checkpoint's <code>detect_objects()</code> method, which implements <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#processing-predictions">this process</a>. Evaluation has to be done at a <code>min_score</code> of <code>0.01</code>, an NMS <code>max_overlap</code> of <code>0.45</code>, and <code>top_k</code> of <code>200</code> to allow fair comparision of results with the paper and other implementations.</p>
<p><strong>Parsed predictions are evaluated against the ground truth objects.</strong> The evaluation metric is the <em>Mean Average Precision (mAP)</em>. If you're not familiar with this metric, <a href="https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173" rel="nofollow">here's a great explanation</a>.</p>
<p>We will use <code>calculate_mAP()</code> in <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py"><code>utils.py</code></a> for this purpose. As is the norm, we will ignore <em>difficult</em> detections in the mAP calculation. But nevertheless, it is important to include them from the evaluation dataset because if the model does detect an object that is considered to be <em>difficult</em>, it must not be counted as a false positive.</p>
<p>The model scores <strong>77.2 mAP</strong>, same as the result reported in the paper.</p>
<p>Class-wise average precisions (not scaled to 100) are listed below.</p>
<table>
<thead>
<tr>
<th align="center">Class</th>
<th align="center">Average Precision</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><em>aeroplane</em></td>
<td align="center">0.7887580990791321</td>
</tr>
<tr>
<td align="center"><em>bicycle</em></td>
<td align="center">0.8351995348930359</td>
</tr>
<tr>
<td align="center"><em>bird</em></td>
<td align="center">0.7623348236083984</td>
</tr>
<tr>
<td align="center"><em>boat</em></td>
<td align="center">0.7218425273895264</td>
</tr>
<tr>
<td align="center"><em>bottle</em></td>
<td align="center">0.45978495478630066</td>
</tr>
<tr>
<td align="center"><em>bus</em></td>
<td align="center">0.8705356121063232</td>
</tr>
<tr>
<td align="center"><em>car</em></td>
<td align="center">0.8655831217765808</td>
</tr>
<tr>
<td align="center"><em>cat</em></td>
<td align="center">0.8828985095024109</td>
</tr>
<tr>
<td align="center"><em>chair</em></td>
<td align="center">0.5917483568191528</td>
</tr>
<tr>
<td align="center"><em>cow</em></td>
<td align="center">0.8255912661552429</td>
</tr>
<tr>
<td align="center"><em>diningtable</em></td>
<td align="center">0.756867527961731</td>
</tr>
<tr>
<td align="center"><em>dog</em></td>
<td align="center">0.856262743473053</td>
</tr>
<tr>
<td align="center"><em>horse</em></td>
<td align="center">0.8778411149978638</td>
</tr>
<tr>
<td align="center"><em>motorbike</em></td>
<td align="center">0.8316892385482788</td>
</tr>
<tr>
<td align="center"><em>person</em></td>
<td align="center">0.7884440422058105</td>
</tr>
<tr>
<td align="center"><em>pottedplant</em></td>
<td align="center">0.5071538090705872</td>
</tr>
<tr>
<td align="center"><em>sheep</em></td>
<td align="center">0.7936667799949646</td>
</tr>
<tr>
<td align="center"><em>sofa</em></td>
<td align="center">0.7998116612434387</td>
</tr>
<tr>
<td align="center"><em>train</em></td>
<td align="center">0.8655905723571777</td>
</tr>
<tr>
<td align="center"><em>tvmonitor</em></td>
<td align="center">0.7492395043373108</td>
</tr>
</tbody>
</table>
<p>You can see that some objects, like bottles and potted plants, are considerably harder to detect than others.</p>
<h1><a id="user-content-inference" class="anchor" aria-hidden="true" href="#inference"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Inference</h1>
<p>See <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py"><code>detect.py</code></a>.</p>
<p>Point to the model you want to use for inference with the <code>checkpoint</code> parameter at the beginning of the code.</p>
<p>Then, you can use the <code>detect()</code> function to identify and visualize objects in an RGB image.</p>
<div class="highlight highlight-source-python"><pre><span class="pl-s1">img_path</span> <span class="pl-c1">=</span> <span class="pl-s">'/path/to/ima.ge'</span>
<span class="pl-s1">original_image</span> <span class="pl-c1">=</span> <span class="pl-v">PIL</span>.<span class="pl-v">Image</span>.<span class="pl-en">open</span>(<span class="pl-s1">img_path</span>, <span class="pl-s1">mode</span><span class="pl-c1">=</span><span class="pl-s">'r'</span>)
<span class="pl-s1">original_image</span> <span class="pl-c1">=</span> <span class="pl-s1">original_image</span>.<span class="pl-en">convert</span>(<span class="pl-s">'RGB'</span>)

<span class="pl-en">detect</span>(<span class="pl-s1">original_image</span>, <span class="pl-s1">min_score</span><span class="pl-c1">=</span><span class="pl-c1">0.2</span>, <span class="pl-s1">max_overlap</span><span class="pl-c1">=</span><span class="pl-c1">0.5</span>, <span class="pl-s1">top_k</span><span class="pl-c1">=</span><span class="pl-c1">200</span>).<span class="pl-en">show</span>()</pre></div>
<p>This function first <strong>preprocesses the image by resizing and normalizing its RGB channels</strong> as required by the model. It then <strong>obtains raw predictions from the model, which are parsed</strong> by the <code>detect_objects()</code> method in the model. The parsed results are converted from fractional to absolute boundary coordinates, their labels are decoded with the <code>label_map</code>, and they are <strong>visualized on the image</strong>.</p>
<p>There are no one-size-fits-all values for <code>min_score</code>, <code>max_overlap</code>, and <code>top_k</code>. You may need to experiment a little to find what works best for your target data.</p>
<h3><a id="user-content-some-more-examples" class="anchor" aria-hidden="true" href="#some-more-examples"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>Some more examples</h3>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000029.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000029.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000045.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000045.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000062.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000062.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000075.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000075.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000085.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000085.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000092.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000092.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000100.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000100.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000124.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000124.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000127.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000127.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000128.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000128.jpg" style="max-width:100%;"></a>
</p>
<hr>
<p align="center">
<a target="_blank" rel="noopener noreferrer" href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/img/000145.jpg"><img src="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/raw/master/img/000145.jpg" style="max-width:100%;"></a>
</p>
<hr>
<h1><a id="user-content-faqs" class="anchor" aria-hidden="true" href="#faqs"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.775 3.275a.75.75 0 001.06 1.06l1.25-1.25a2 2 0 112.83 2.83l-2.5 2.5a2 2 0 01-2.83 0 .75.75 0 00-1.06 1.06 3.5 3.5 0 004.95 0l2.5-2.5a3.5 3.5 0 00-4.95-4.95l-1.25 1.25zm-4.69 9.64a2 2 0 010-2.83l2.5-2.5a2 2 0 012.83 0 .75.75 0 001.06-1.06 3.5 3.5 0 00-4.95 0l-2.5 2.5a3.5 3.5 0 004.95 4.95l1.25-1.25a.75.75 0 00-1.06-1.06l-1.25 1.25a2 2 0 01-2.83 0z"></path></svg></a>FAQs</h1>
<p><strong>I noticed that priors often overshoot the <code>3, 3</code> kernel employed in the prediction convolutions. How can the kernel detect a bound (of an object) outside it?</strong></p>
<p>Don't confuse the kernel and its <em>receptive field</em>, which is the area of the original image that is represented in the kernel's field-of-view.</p>
<p>For example, on the <code>38, 38</code> feature map from <code>conv4_3</code>, a <code>3, 3</code> kernel covers an area of <code>0.08, 0.08</code> in fractional coordinates. The priors are <code>0.1, 0.1</code>, <code>0.14, 0.07</code>, <code>0.07, 0.14</code>, and <code>0.14, 0.14</code>.</p>
<p>But its receptive field, which <a href="https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807" rel="nofollow">you can calculate</a>, is a whopping <code>0.36, 0.36</code>! Therefore, all priors (and objects contained therein) are present well inside it.</p>
<p>Keep in mind that the receptive field grows with every successive convolution. For <code>conv_7</code> and the higher-level feature maps, a <code>3, 3</code> kernel's receptive field will cover the <em>entire</em> <code>300, 300</code> image. But, as always, the pixels in the original image that are closer to the center of the kernel have greater representation, so it is still <em>local</em> in a sense.</p>
<hr>
<p><strong>While training, why can't we match predicted boxes directly to their ground truths?</strong></p>
<p>We cannot directly check for overlap or coincidence between predicted boxes and ground truth objects to match them because predicted boxes are not to be considered reliable, <em>especially</em> during the training process. This is the very reason we are trying to evaluate them in the first place!</p>
<p>And this is why priors are especially useful. We can match a predicted box to a ground truth box by means of the prior it is supposed to be approximating. It no longer matters how correct or wildly wrong the prediction is.</p>
<hr>
<p><strong>Why do we even have a <em>background</em> class if we're only checking which <em>non-background</em> classes meet the threshold?</strong></p>
<p>When there is no object in the approximate field of the prior, a high score for <em>background</em> will dilute the scores of the other classes such that they will not meet the detection threshold.</p>
<hr>
<p><strong>Why not simply choose the class with the highest score instead of using a threshold?</strong></p>
<p>I think that's a valid strategy. After all, we implicitly conditioned the model to choose <em>one</em> class when we trained it with the Cross Entropy loss. But you will find that you won't achieve the same performance as you would with a threshold.</p>
<p>I suspect this is because object detection is open-ended enough that there's room for doubt in the trained model as to what's really in the field of the prior. For example, the score for <em>background</em> may be high if there is an appreciable amount of backdrop visible in an object's bounding box. There may even be multiple objects present in the same approximate region. A simple threshold will yield all possibilities for our consideration, and it just works better.</p>
<p>Redundant detections aren't really a problem since we're NMS-ing the hell out of 'em.</p>
<hr>
<p><strong>Sorry, but I gotta ask... <em><a href="https://cnet4.cbsistatic.com/img/cLD5YVGT9pFqx61TuMtcSBtDPyY=/570x0/2017/01/14/6d8103f7-a52d-46de-98d0-56d0e9d79804/se7en.png" rel="nofollow">what's in the boooox?!</a></em></strong></p>
<p>Ha.</p>
</article>
      </div>
  </div>


</div>
    <div class="flex-shrink-0 col-12 col-md-3">
            

      <div class="BorderGrid BorderGrid--spacious" data-pjax>
        <div class="BorderGrid-row hide-sm hide-md">
          <div class="BorderGrid-cell">
            <h2 class="mb-3 h4">About</h2>

    <p class="f4 mt-3">
      SSD: Single Shot MultiBox Detector | a PyTorch Tutorial to Object Detection
    </p>

  <h3 class="sr-only">Topics</h3>
  <div class="mt-3">
      <div class="f6">
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:pytorch" href="/topics/pytorch" title="Topic: pytorch" class="topic-tag topic-tag-link ">
  pytorch
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:pytorch-tutorial" href="/topics/pytorch-tutorial" title="Topic: pytorch-tutorial" class="topic-tag topic-tag-link ">
  pytorch-tutorial
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:object-detection" href="/topics/object-detection" title="Topic: object-detection" class="topic-tag topic-tag-link ">
  object-detection
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:single-shot-multibox-detector" href="/topics/single-shot-multibox-detector" title="Topic: single-shot-multibox-detector" class="topic-tag topic-tag-link ">
  single-shot-multibox-detector
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:single-shot-detection" href="/topics/single-shot-detection" title="Topic: single-shot-detection" class="topic-tag topic-tag-link ">
  single-shot-detection
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:object-recognition" href="/topics/object-recognition" title="Topic: object-recognition" class="topic-tag topic-tag-link ">
  object-recognition
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:ssd" href="/topics/ssd" title="Topic: ssd" class="topic-tag topic-tag-link ">
  ssd
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:tutorial" href="/topics/tutorial" title="Topic: tutorial" class="topic-tag topic-tag-link ">
  tutorial
</a>
      <a data-ga-click="Topic, repository page" data-octo-click="topic_click" data-octo-dimensions="topic:detection" href="/topics/detection" title="Topic: detection" class="topic-tag topic-tag-link ">
  detection
</a>
  </div>

  </div>

  <h3 class="sr-only">Resources</h3>
  <div class="mt-3">
    <a class="muted-link" href="#readme">
      <svg class="octicon octicon-book mr-2" height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm8.755 3a2.25 2.25 0 012.25-2.25H14.5v9h-3.757c-.71 0-1.4.201-1.992.572l.004-7.322zm-1.504 7.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574z"></path></svg>
      Readme
</a>  </div>

  <h3 class="sr-only">License</h3>
  <div class="mt-3">
    <a href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/LICENSE" class="muted-link" >
      <svg class="octicon octicon-law mr-2" height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.75.75a.75.75 0 00-1.5 0V2h-.984c-.305 0-.604.08-.869.23l-1.288.737A.25.25 0 013.984 3H1.75a.75.75 0 000 1.5h.428L.066 9.192a.75.75 0 00.154.838l.53-.53-.53.53v.001l.002.002.002.002.006.006.016.015.045.04a3.514 3.514 0 00.686.45A4.492 4.492 0 003 11c.88 0 1.556-.22 2.023-.454a3.515 3.515 0 00.686-.45l.045-.04.016-.015.006-.006.002-.002.001-.002L5.25 9.5l.53.53a.75.75 0 00.154-.838L3.822 4.5h.162c.305 0 .604-.08.869-.23l1.289-.737a.25.25 0 01.124-.033h.984V13h-2.5a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-2.5V3.5h.984a.25.25 0 01.124.033l1.29.736c.264.152.563.231.868.231h.162l-2.112 4.692a.75.75 0 00.154.838l.53-.53-.53.53v.001l.002.002.002.002.006.006.016.015.045.04a3.517 3.517 0 00.686.45A4.492 4.492 0 0013 11c.88 0 1.556-.22 2.023-.454a3.512 3.512 0 00.686-.45l.045-.04.01-.01.006-.005.006-.006.002-.002.001-.002-.529-.531.53.53a.75.75 0 00.154-.838L13.823 4.5h.427a.75.75 0 000-1.5h-2.234a.25.25 0 01-.124-.033l-1.29-.736A1.75 1.75 0 009.735 2H8.75V.75zM1.695 9.227c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L3 6.327l-1.305 2.9zm10 0c.285.135.718.273 1.305.273s1.02-.138 1.305-.273L13 6.327l-1.305 2.9z"></path></svg>
        MIT License
    </a>
  </div>

          </div>
        </div>
          <div class="BorderGrid-row">
            <div class="BorderGrid-cell">
              <h2 class="h4 mb-3">
  <a href="/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/releases" class="link-gray-dark no-underline ">
    Releases
</a></h2>

    <div class="text-small color-text-secondary">No releases published</div>

            </div>
          </div>
          <div class="BorderGrid-row">
            <div class="BorderGrid-cell">
              <h2 class="h4 mb-3">
  <a href="/users/sgrvinod/packages?repo_name=a-PyTorch-Tutorial-to-Object-Detection" class="link-gray-dark no-underline ">
    Packages <span title="0" hidden="hidden" class="Counter ">0</span>
</a></h2>
