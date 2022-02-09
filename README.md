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
