# Object Detection using YOLO

<p align='justify'>By applying object detection, you’ll not only be able to determine what is in an image, but also where a given object resides. We’ll start with a brief discussion of the YOLO object detector, including how the object detector works.</p>

## What is the YOLO object detector?

<p align='justify'>When it comes to deep learning-based object detection, there are three primary object detectors you’ll encounter:</p>

* R-CNN and their variants, including the original R-CNN, Fast R-CNN, and Faster R-CNN;

* Single Shot Detector (SSDs);

* YOLO.

<p align='justify'>R-CNNs are one of the first deep learning-based object detectors and are an example of a two-stage detector. In the first R-CNN publication, <a href="https://arxiv.org/abs/1311.2524">Rich feature hierarchies for accurate object detection and semantic segmentation</a>, (2013) Girshick et al, proposed an object detector that required an algorithm such as <a href="http://www.huppelen.nl/publications/selectiveSearchDraft.pdf">Selective Search for Object Recognition</a> (or equivalent) to propose candidate bounding boxes that could contain objects.</p>
