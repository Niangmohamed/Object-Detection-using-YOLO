# Object Detection using YOLO from PyImageSearch

<p align='justify'>By applying object detection, you’ll not only be able to determine what is in an image, but also where a given object resides. We’ll start with a brief discussion of the YOLO object detector, including how the object detector works.</p>

## What is the YOLO object detector?

<p align='justify'>When it comes to deep learning-based object detection, there are three primary object detectors you’ll encounter:</p>

* R-CNN and their variants, including the original R-CNN, Fast R-CNN, and Faster R-CNN;

* Single Shot Detector (SSDs);

* YOLO.

<p align='justify'>R-CNNs are one of the first deep learning-based object detectors and are an example of a two-stage detector. In the first R-CNN publication, <a href="https://arxiv.org/abs/1311.2524">Rich feature hierarchies for accurate object detection and semantic segmentation</a>, (2013), Girshick et al. proposed an object detector that required an algorithm such as <a href="http://www.huppelen.nl/publications/selectiveSearchDraft.pdf">Selective Search for Object Recognition</a> (or equivalent) to propose candidate bounding boxes that could contain objects.</p>

<p align='justify'>These regions were then passed into a CNN for classification, ultimately leading to one of the first deep learning-based object detectors. The problem with the standard R-CNN method was that it was painfully slow and not a complete end-to-end object detector. Girshick et al. published a second paper in 2015, entitled <a href="https://arxiv.org/abs/1504.08083">Fast R-CNN</a>. The Fast R-CNN algorithm made considerable improvements to the original R-CNN, namely increasing accuracy and reducing the time it took to perform a forward pass; however, the model still relied on an external region proposal algorithm.</p>

<p align='justify'>It wasn’t until Girshick et al.’s follow-up 2015 paper, <a href="https://arxiv.org/abs/1506.01497">Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks</a>, that R-CNNs became a true end-to-end deep learning object detector by removing the Selective Search requirement and instead relying on a Region Proposal Network (RPN) that is (1) fully convolutional and (2) can predict the object bounding boxes and objectness scores (i.e., a score quantifying how likely it is a region of an image may contain an image). The outputs of the RPNs are then passed into the R-CNN component for final classification and labeling.</p>

<p align='justify'>While R-CNNs tend to very accurate, the biggest problem with the R-CNN family of networks is their speed — they were incredibly slow, obtaining only 5 FPS on a GPU. To help increase the speed of deep learning-based object detectors, both Single Shot Detectors (SSDs) and YOLO use a one-stage detector strategy. These algorithms treat object detection as a regression problem, taking a given input image and simultaneously learning bounding box coordinates and corresponding class label probabilities.</p>

<p align='justify'>In general, single-stage detectors tend to be less accurate than two-stage detectors but are significantly faster. YOLO is a great example of a single stage detector. First introduced in 2015 by Redmon et al., their paper, <a href="https://arxiv.org/abs/1506.02640">You Only Look Once: Unified, Real-Time Object Detection</a>, details an object detector capable of super real-time object detection, obtaining 45 FPS on a GPU. Note: A smaller variant of their model called Fast YOLO claims to achieve 155 FPS on a GPU.</p>

<p align='justify'>YOLO has gone through a number of different iterations, including <a href="https://arxiv.org/abs/1612.08242">YOLO9000: Better, Faster, Stronger</a> (i.e., YOLOv2), capable of detecting over 9,000 object detectors. Redmon and Farhadi are able to achieve such a large number of object detections by performing joint training for both object detection and classification. Using joint training the authors trained YOLO9000 simultaneously on both the ImageNet classification dataset and COCO detection dataset. The result is a YOLO model, called YOLO9000, that can predict detections for object classes that don’t have labeled detection data.</p>

<p align='justify'>While interesting and novel, YOLOv2’s performance was a bit underwhelming given the title and abstract of the paper. We’ll be using YOLOv3 in this blog post, in particular, YOLO trained on the COCO dataset. The COCO dataset consists of 80 labels. YOLO — You Only Look Once — is an extremely fast multi object detection algorithm which uses convolutional neural network (CNN) to detect and identify objects. The neural network has this network architecture.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/a9d4ee8e2d6b856b0a598141624ac197c86171da/images/yolo-nn-architecture.png"/>
</p>

<p align="center">
  Source: <a href="https://arxiv.org/pdf/1506.02640.pdf">You Only Look Once — Unified, Real-Time Object Detection.</a>
</p>

## How does the YOLO framework works?

<p align='justify'>Now that we have grasp on why YOLO is such a useful framework, let’s jump into how it actually works. In this section, I have mentioned the steps followed by YOLO for detecting objects in a given image.</p>

* YOLO first takes an input image.

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-1.png"/>
</p>

* The framework then divides the input image into grids (say a 3 X 3 grid).

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-2.png"/>
</p>

* <p align="justify">Image classification and localization are applied on each grid. YOLO then predicts the bounding boxes and their corresponding class probabilities for objects (if any are found, of course).</p>

<p align="justify">Pretty straightforward, isn’t it? Let’s break down each step to get a more granular understanding of what we just learned. We need to pass the labelled data to the model in order to train it. Suppose we have divided the image into a grid of size 3 X 3 and there are a total of 3 classes which we want the objects to be classified into. Let’s say the classes are Pedestrian, Car, and Motorcycle respectively. So, for each grid cell, the label y will be an eight dimensional vector.</p>

<div align='center'>
  
|       |   pc  |
| :---: | :---: |
|       |   bx  |
|       |   by  |
|   y   |   bh  |
|       |   bw  |
|       |   c1  |
|       |   c2  |
|       |   c3  |
  
</div>

<p align="justify">Here:</p>

* pc defines whether an object is present in the grid or not (it is the probability);

* bx, by, bh, bw specify the bounding box if there is an object;

* c1, c2, c3 represent the classes. So, if the object is a car, c2 will be 1 and c1 & c3 will be 0, and so on.


<p align="justify">Let’s say we select the first grid from the above example.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-3.png"/>
</p>

<p align="justify">Since there is no object in this grid, pc will be zero and the y label for this grid will be.</p>

<div align='center'>
  
|       |   0   |
| :---: | :---: |
|       |   ?   |
|       |   ?   |
|   y   |   ?   |
|       |   ?   |
|       |   ?   |
|       |   ?   |
|       |   ?   |
  
</div>

<p align="justify">Here, <b>?</b> means that it doesn’t matter what bx, by, bh, bw, c1, c2, and c3 contain as there is no object in the grid. Let’s take another grid in which we have a car (c2 = 1).</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-4.png"/>
</p>

<p align="justify">Before we write the y label for this grid, it’s important to first understand how YOLO decides whether there actually is an object in the grid. In the above image, there are two objects (two cars), so YOLO will take the mid-point of these two objects and these objects will be assigned to the grid which contains the mid-point of these objects. The y label for the centre left grid with the car will be.</p>
  
<div align='center'>
  
|       |   1   |
| :---: | :---: |
|       |   bx  |
|       |   by  |
|   y   |   bh  |
|       |   bw  |
|       |   0   |
|       |   1   |
|       |   0   |
  
</div>
  
<p align="justify">Since there is an object in this grid, pc will be equal to 1. bx, by, bh, bw will be calculated relative to the particular grid cell we are dealing with. Since car is the second class, c2 = 1 and c1 and c3 = 0. So, for each of the 9 grids, we will have an eight dimensional output vector. This output will have a shape of 3 X 3 X 8. So now we have an input image and it’s corresponding target vector. Using the above example (input image – 100 X 100 X 3, output – 3 X 3 X 8), the model will be trained as follows.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-5.png"/>
</p>

<p align="justify">We will run both forward and backward propagation to train the model. During the testing phase, we pass an image to the model and run forward propagation until we get an output y. In order to keep things simple, I have explained this using a 3 X 3 grid here, but generally in real-world scenarios we take larger grids (perhaps 19 X 19).</p>

<p align="justify">Even if an object spans out to more than one grid, it will only be assigned to a single grid in which its mid-point is located. We can reduce the chances of multiple objects appearing in the same grid cell by increasing the more number of grids (19 X 19, for example).</p>

## How to encode bounding boxes?

<p align="justify">As I mentioned earlier, bx, by, bh, and bw are calculated relative to the grid cell we are dealing with. Let’s understand this concept with an example. Consider the center-right grid which contains a car.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-6.png"/>
</p>

<p align="justify">So, bx, by, bh, and bw will be calculated relative to this grid only. The y label for this grid will be.</p>

<div align='center'>
  
|       |   1   |
| :---: | :---: |
|       |   bx  |
|       |   by  |
|   y   |   bh  |
|       |   bw  |
|       |   0   |
|       |   1   |
|       |   0   |
  
</div>

<p align="justify">pc = 1 since there is an object in this grid and since it is a car, c2 = 1. Now, let’s see how to decide bx, by, bh, and bw. In YOLO, the coordinates assigned to all the grids are.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-7.png"/>
</p>

<p align="justify">bx, by are the x and y coordinates of the midpoint of the object with respect to this grid. In this case, it will be (around) bx = 0.4 and by = 0.3.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-8.png"/>
</p>

<p align="justify">bh is the ratio of the height of the bounding box (red box in the above example) to the height of the corresponding grid cell, which in our case is around 0.9. So,  bh = 0.9. bw is the ratio of the width of the bounding box to the width of the grid cell. So, bw = 0.5 (approximately). The y label for this grid will be.</p>

<div align='center'>
  
|       |   1   |
| :---: | :---: |
|       |   0.4 |
|       |   0.3 |
|   y   |   0.9 |
|       |   0.5 |
|       |   0   |
|       |   1   |
|       |   0   |
  
</div>

<p align="justify">Notice here that bx and by will always range between 0 and 1 as the midpoint will always lie within the grid. Whereas bh and bw can be more than 1 in case the dimensions of the bounding box are more than the dimension of the grid. In the next section, we will look at more ideas that can potentially help us in making this algorithm’s performance even better.</p>

## Intersection over Union and Non-Max Suppression

<p align="justify">Here’s some food for thought – how can we decide whether the predicted bounding box is giving us a good outcome (or a bad one)? This is where Intersection over Union comes into the picture. It calculates the intersection over union of the actual bounding box and the predicted bonding box. Consider the actual and predicted bounding boxes for a car as shown below.</p>


<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-9.png"/>
</p>

<p align="justify">Here, the red box is the actual bounding box and the blue box is the predicted one. How can we decide whether it is a good prediction or not? IoU, or Intersection over Union, will calculate the area of the intersection over union of these two boxes. That area will be.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-10.png"/>
</p>

**IoU = Area of the intersection / Area of the union, i.e. IoU = Area of yellow box / Area of green box.**

<p align="justify">If IoU is greater than 0.5, we can say that the prediction is good enough. 0.5 is an arbitrary threshold we have taken here, but it can be changed according to your specific problem. Intuitively, the more you increase the threshold, the better the predictions become. There is one more technique that can improve the output of YOLO significantly – Non-Max Suppression. One of the most common problems with object detection algorithms is that rather than detecting an object just once, they might detect it multiple times. Consider the below image.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-11.png"/>
</p>

<p align="justify">Here, the cars are identified more than once. The Non-Max Suppression technique cleans up this up so that we get only a single detection per object. Let’s see how this approach works.</p>

<p align="justify">It first looks at the probabilities associated with each detection and takes the largest one. In the above image, 0.9 is the highest probability, so the box with 0.9 probability will be selected first.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-12.png"/>
</p>

<p align="justify">Now, it looks at all the other boxes in the image. The boxes which have high IoU with the current box are suppressed. So, the boxes with 0.6 and 0.7 probabilities will be suppressed in our example.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-13.png"/>
</p>

<p align="justify">After the boxes have been suppressed, it selects the next box from all the boxes with the highest probability, which is 0.8 in our case.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-14.png"/>
</p>

<p align="justify">Again it will look at the IoU of this box with the remaining boxes and compress the boxes with a high IoU.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-15.png"/>
</p>


<p align="justify">We repeat these steps until all the boxes have either been selected or compressed and we get the final bounding boxes.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-16.png"/>
</p>

<p align="justify">This is what Non-Max Suppression is all about. We are taking the boxes with maximum probability and suppressing the close-by boxes with non-max probabilities. Let’s quickly summarize the points which we’ve seen in this section about the Non-Max suppression algorithm:</p>

* <p align="justify">Discard all the boxes having probabilities less than or equal to a pre-defined threshold (say, 0.5);</p>

* <p align="justify">For the remaining boxes:</p>

  - <p align="justify">Pick the box with the highest probability and take that as the output prediction;</p>

  - <p align="justify">Discard any other box which has IoU greater than the threshold with the output box from the above step.</p>

* <p align="justify">Repeat step 2 until all the boxes are either taken as the output prediction or discarded.</p>

<p align="justify">There is another method we can use to improve the perform of a YOLO algorithm – let’s check it out!</p>

## Anchor boxes

<p align="justify">We have seen that each grid can only identify one object. But what if there are multiple objects in a single grid? That can so often be the case in reality. And that leads us to the concept of anchor boxes. Consider the following image, divided into a 3 X 3 grid.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-17.png"/>
</p>

<p align="justify">Remember how we assigned an object to a grid? We took the midpoint of the object and based on its location, assigned the object to the corresponding grid. In the above example, the midpoint of both the objects lies in the same grid. This is how the actual bounding boxes for the objects will be.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-18.png"/>
</p>

<p align="justify">We will only be getting one of the two boxes, either for the car or for the person. But if we use anchor boxes, we might be able to output both boxes! How do we go about doing this? First, we pre-define two different shapes called anchor boxes or anchor box shapes. Now, for each grid, instead of having one output, we will have two outputs. We can always increase the number of anchor boxes as well. I have taken two here to make the concept easy to understand.</p>

<p align="center">
  <img src="https://github.com/Niangmohamed/Object-Detection-using-YOLO/blob/353a2be4fd8750b044c4f0c3bcbe3168d200c043/images/image-19.png"/>
</p>

<p align="justify">This is how the y label for YOLO without anchor boxes looks like.</p>


<div align='center'>
  
|       |   pc  |
| :---: | :---: |
|       |   bx  |
|       |   by  |
|   y   |   bh  |
|       |   bw  |
|       |   c1  |
|       |   c2  |
|       |   c3  |
  
</div>

<p align="justify">What do you think the y label will be if we have 2 anchor boxes? I want you to take a moment to ponder this before reading further. Got it? The y label will be.</p>

<div align='center'>

|      | pc   |
| :--- | :--- |
|      | bx   |
|      | by   |
|      | bh   |
|      | bw   |
|      | c1   |
|      | c2   |
|   y  | c3   |
|      | pc   |
|      | bx   |
|      | by   |
|      | bh   |
|      | bw   |
|      | c1   |
|      | c2   |
|      | c3   |

  </div>

<p align="justify">The first 8 rows belong to anchor box 1 and the remaining 8 belongs to anchor box 2. The objects are assigned to the anchor boxes based on the similarity of the bounding boxes and the anchor box shape. Since the shape of anchor box 1 is similar to the bounding box for the person, the latter will be assigned to anchor box 1 and the car will be assigned to anchor box 2. The output in this case, instead of 3 X 3 X 8 (using a 3 X 3 grid and 3 classes), will be 3 X 3 X 16 (since we are using 2 anchors).</p>


<p align="justify">So, for each grid, we can detect two or more objects based on the number of anchors. Let’s combine all the ideas we have covered so far and integrate them into the YOLO framework.</p>
