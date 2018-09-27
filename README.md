
This repository contains experiments with sliding windows as detection mechanism for digits in a sequence of digits. 
<br/><br/>
As a simple baseline, an image created by concatenation of MNIST handwritten digits, is used. A window is horizontally moved over the image. At each given point a NN is used to classify the pixels determined by the window (box). No attempt has been made to include scaling etc. to cover more realistic scenarios since the main purpose was to evaluate different Non-Maximum Suppression (NMS) implementations.
<br/>
![Alt text](images/slidingwindow.gif?raw=true "")
<br/>
NMS suppress bounding boxes that overlap significantly with bounding boxes that have a higher detection score, i.e. NN class probability. In addition the algorithm also suppress bounding boxes with a detection scores lower than a given threshold. As output NMS will generate a list of unique boxes with the highest detection probability.
<br/>
![Alt text](images/line.gif?raw=true "")

<br/>
<b>SlidingWindow_NMS_evaluation</b> – the notebook provides an evaluation of four different implementations of the NMS algorithm:
<br/><br/>
- Numpy implementation of NMS adapted from <a href="https://gitlab.informatik.haw-hamburg.de/acf530/ssd-pytorch/blob/bc72ac7d50bf8905e6f8f5650254365e931b97d4/box_matcher.py">here.</a>
<br/><br/>
- GPU implementation of NMS using the code from <a href="https://github.com/jwyang/faster-rcnn.pytorch">faster-rcnn.pytorch </a>
<i>To run the example download the faster-rcnn repository and compile the “lib” directory as described in the readme or the repository. Link the “model” directory in the “lib” folder into the directory in which the notebook is run.</i>
<br/><br/>
- Pytorch implementation of NMS
<br/><br/>
- Python List implementation of NMS 
<br/><br/>
<b>SlidingWindow_NMS_greedy_gpu_based, 
SlidingWindow_NMS_greedy_list_based, 
SlidingWindow_NMS_greedy_numpy_based, SlidingWindow_NMS_greedy_torch_based</b> - are notebooks that allow to experiment with the different implementations of NMS and visualize results.

