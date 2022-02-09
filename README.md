# Object Detection using YOLO

<p align='justify'>By applying object detection, you’ll not only be able to determine what is in an image, but also where a given object resides. We’ll start with a brief discussion of the YOLO object detector, including how the object detector works.</p>

## What is the YOLO object detector?

<p align='justify'>When it comes to deep learning-based object detection, there are three primary object detectors you’ll encounter:</p>

* R-CNN and their variants, including the original R-CNN, Fast R-CNN, and Faster R-CNN;

* Single Shot Detector (SSDs);

* YOLO.

<p align='justify'>R-CNNs are one of the first deep learning-based object detectors and are an example of a two-stage detector. In the first R-CNN publication, <a href="https://arxiv.org/abs/1311.2524">Rich feature hierarchies for accurate object detection and semantic segmentation</a>, (2013), Girshick et al. proposed an object detector that required an algorithm such as <a href="http://www.huppelen.nl/publications/selectiveSearchDraft.pdf">Selective Search for Object Recognition</a> (or equivalent) to propose candidate bounding boxes that could contain objects.</p>

<p align='justify'>These regions were then passed into a CNN for classification, ultimately leading to one of the first deep learning-based object detectors. The problem with the standard R-CNN method was that it was painfully slow and not a complete end-to-end object detector. Girshick et al. published a second paper in 2015, entitled <a href="https://arxiv.org/abs/1504.08083">Fast R-CNN</a>. The Fast R-CNN algorithm made considerable improvements to the original R-CNN, namely increasing accuracy and reducing the time it took to perform a forward pass; however, the model still relied on an external region proposal algorithm.</p>
