# 100 Days Of Code - Log

### Day 0: June 27, Monday

**Today's Progress**: I've gone through many exercises on FreeCodeCamp.

**Thoughts** I've recently started coding, and it's a great feeling when I finally solve an algorithm challenge after a lot of attempts and hours spent.

**Link(s) to work**
1. [Find the Longest Word in a String](https://www.freecodecamp.com/challenges/find-the-longest-word-in-a-string)
2. [Title Case a Sentence](https://www.freecodecamp.com/challenges/title-case-a-sentence)



### Day 0: July 26, 2018 

**Today's Progress**: command notes:
#disable touchpad command
xinput list
xinput disable 13/14

# inference Mask RCNN
python2 tools/infer_simple.py  --cfg configs/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml     --output-dir demo/output/detectron-visualizations  --image-ext jpg --wts weights/model_final.pkl demo

python2 tools/train_net.py  --cfg configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml  OUTPUT_DIR demo/output/weights/detectron-output


disable the touch pad

GPU monitoring:
watch -n 1 nvidia-smi

check tensorflow version
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3
python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2

**Thoughts:** 
1. use stereo matching to identify the outdoor situation. also can try
2. if the object space occupation in image over threshold, give the single label.
3. if more than 3 types of labels, defining and classify as a scene.
    3.1 count the objects, compare the size of box, calcuthe  all in door, class. 
    3.2 define the scene.
    3.3 interface with

4. how to develop different module and integrate them.
5. try autoML and watch tutorial

**Link to work:** [Multi label classification](http://www.example.com)

### Day 1: July27, 2018
**Today's Progress**: 
stuck on the training own data on detectron.
1. convert dataset into /lib/datasets in json formate.
2. change number of categories.
3. finetune
4. TODo : Choose a dataset, identify three questions, and analyze the data to find answers to these questions. Focus on machine learning, data visualization, and communication.
##

steps:

1. Create a COCO-like dataset.
2. Add it to the dataset catalog (lib/datasets)
3. load the pre-trained model and modify the output layers
4. train the model using train_net.py
5. edit yaml for configuration.
   
todolist on firefox:

1. auto anotation paper search.
2. the popular label box: https://github.com/tzutalin/labelImg.git
   seach on youtube tutorial. 
3. annotation tool on youtube:https://www.youtube.com/watch?v=-F4V2AwSSEA
   github: https://github.com/VisionForFuture/box-label-tool
   manual annotation list https://en.wikipedia.org/wiki/List_of_manual_image_annotation_tools
4. Create COCO dataset tool: https://github.com/waspinator/pycococreator/
5. https://github.com/nightrome/cocostuffapi
   http://cocodataset.org/#download
6. search in chiese with detectron 训练
7. yaml study http://www.ruanyifeng.com/blog/2016/07/yaml.html
   


**Link to work:** [object detection](http://www.example.com)


### Day 2: July 30, 2018
**Today's Progress**: 
https://github.com/shiyemin/voc2coco
this can be used for convert annotation xml to coco Json file.
Now start train R-CNN, only 1 image takes 2hours training time.
aws should be applied.

according keras tutorial:
http://hellodfan.com/2017/11/08/Keras%E5%AE%9E%E7%8E%B0Mask-R-CNN/


error for training own module, need to download coco dataset
and then check these links:
https://github.com/caffe2/caffe2/issues/2130
https://github.com/facebookresearch/Detectron/issues/60
http://www.infoq.com/cn/articles/image-object-recognition-mask-rcnn

### Day 3: July 31, 2018
**Today's Progress**: 
best for selective search
http://jermmy.xyz/2017/05/04/2017-5-4-paper-notes-selective-search/


### Day 4: Augest 1, 2018
**Today's Progress**: 
1. Disparity map to find out the sky area. check the distance and segmentation.
2. shif mean segmentation.
3. tried 2 programme,  show_seg and SGBM
4. 
   

### Day 5: Augest 2, 2018
**Today's Progress**: 
1. stixel concept: http://www.shujubang.com/m/info.aspx?InfoId=17289
2. wrong diretion

3. 2.5D datasets:  https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2
4. ToDO  add pythonPATH in ~/.bashrc

### Day 6: Augest 3, 2018
**Today's Progress**: 
1. SIFT understanding: 
https://blog.csdn.net/abcjennifer/article/details/7639681
https://blog.csdn.net/u010807846/article/details/49660095   Very good

2. texture represent: 
3. full con
**Term explaination**: 
1. fine-turning: change the model and then training with small dataset.  https://www.zhihu.com/question/40850491
2. cookbook size:  front-back ground identify. search on zhihu.
3. de-cNN https://github.com/vdumoulin/conv_arithmetic
4. Feature Fusion: a way of adding context information to a fully convolutional architecture for segmentation.
5. Horizontal Height Angle (HHA) :are used for encoding the depth into three channels as follows: horizontal disparity, height above ground
   and the angle between local surface normal and the inferred gravity direction.
6. LSTM Code line by line explaination: https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
7. to run DSRG model, re-compile the caffe.  https://github.com/speedinghzl/DSRG  "Weakly-Supervised Semantic Segmentation Network with
Deep Seeded Region Growing"

7.  search CVPR2018 segmentation in github
8.  search LSTM in zhihu
9.  Context Encoding for Semantic Segmentation (optional)
10. Multi-view  Self-supervised  Deep  Learning  for  6D  Pose  Estimation in  the  Amazon  Picking  Challenge (option)
11. Semantic-Segmentation-Suite :https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
12. SIFT Meets CNN:https://arxiv.org/pdf/1608.01807.pdf
13. A comparative study of texture measures with classification based on featured distributions 
14. Keras recurrent tutorial  https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
15. Demo for segNet  and keras segnet https://github.com/imlab-uiip/keras-segnet
16. keras segmentation: https://github.com/divamgupta/image-segmentation-keras
17. A survey on deep learning-based fine-grained object classification and semantic segmentation
18. !! try first :https://github.com/ankitdhall/imageSegmentation
### try github first
  https://github.com/vsatyakumar/Image-Segmentation-Using-Color-and-Texture-Descriptors-with-Expectation-Maximization
19. https://github.com/sunilvengalil/TextureDL
https://github.com/meetshah1995/EE-702/blob/master/stereo-vision/papers/stereo_ethZ.pdf  (Opetion)

20. https://github.com/TimoSaemann/ENet/tree/master/Tutorial
21. Pyramid Stereo Matching Network (CVPR2018) https://github.com/JiaRenChang/PSMNet
22. https://github.com/MaybeShewill-CV/sky-detector
23. https://github.com/k29/horizon_detection
24. https://github.com/mithi/advanced-lane-detection
25. Disparity image segmentation for free-space detection DOI: 10.1109/ICCP.2016.7737150
26. Vehicles Detection in Stereo Vision Based on Disparity Map Segmentation and Objects Classification (Opetion)
27. https://palaeo-electronica.org/2002_1/light/stereo.html
28. https://github.com/s-gupta/rcnn-depth
29. https://github.com/s-gupta/rgbd
30. https://github.com/robotology/segmentation/tree/master/dispBlobber
    

### Day 7: Augest 4, 2018
**Today's Progress**: 
FCN with Depth
test FCN
RGB-D
https://github.com/SeongjinPark/RDFNet
https://github.com/SSAW14/STD2P
https://github.com/bonlime/keras-deeplab-v3-plus

2D FCN
https://github.com/MarvinTeichmann/KittiSeg  which integrated  https://github.com/MarvinTeichmann/tensorflow-fcn
https://github.com/MarvinTeichmann/MultiNet

understanding cityscapes
http://tuprints.ulb.tu-darmstadt.de/6893/1/20171023_dissertationMariusCordts.pdf
http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf


Note for caffe compile (2 common issues):
https://github.com/NVIDIA/DIGITS/issues/156
https://github.com/BVLC/caffe/issues/489

caffe compiled.

### Day 8: Augest 7, 2018
**Today's Progress**: 
The project depth-map result is just so so,or say is not good.

check keras version cmd:
python -c 'import keras; print(keras.__version__)'
python -c 'import numpy;print numpy.__path__'
rm -rf .git

very very good instruction for deeplab:
https://lijiancheng0614.github.io/2018/03/13/2018_03_13_TensorFlow-DeepLab/

how to prepare your own dataset
http://hellodfan.com/2018/07/06/DeepLabv3-with-own-dataset/   (chinese: https://blog.csdn.net/u011974639/article/details/80948990)

generate some label and then 

从基本功开始，先看segmentation
https://www.scipy-lectures.org/packages/scikit-image/index.html
Texture recognition
https://blog.csdn.net/u011974639/article/details/79887573

https://sthalles.github.io/deep_segmentation_network/
https://www.zhihu.com/question/24529483   (weight decay,  optional)

制作数据流程：
https://blog.csdn.net/u010402786/article/details/72883421

https://blog.csdn.net/zoro_lov3/article/details/74550735
https://blog.csdn.net/m_buddy/article/details/78667813
https://oldpan.me/archives/image-segment-make-voc-datasets

Semantic Segmentation review
https://zhuanlan.zhihu.com/p/37801090

cs231n Stanford 
https://www.bilibili.com/video/av17204303/

The conventional approaches of feature extraction are based on color and texture. The semantic segmentation was widely applied after 2010.

https://zhuanlan.zhihu.com/p/30732385

### Day9 Augest 10 2018
1. mask overlap for texture, color and disparity.
2. prepare the  index file and TF record.
3. add new class, include color, class name and number.
   

http://paduaresearch.cab.unipd.it/10337/1/pagnutti_giampaolo_tesi.pdf

https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef

https://zhuanlan.zhihu.com/p/25013378

https://stackoverflow.com/questions/45247717/how-to-segment-the-connected-area-based-on-depth-color-in-opencv

https://github.com/aleju/papers/blob/master/neural-nets/Fast_Scene_Understanding_for_Autonomous_Driving.md



### Day 10 Augest 12 2018
Note for deeplab
https://github.com/xmojiao/deeplab_v2

https://geekcircle.org/machine-learning-interview-qa/
https://github.com/imhuay/Interview_Notes-Chinese 

http://hangzh.com/PyTorch-Encoding/experiments/segmentation.html

search labelme_json_to_dataset
https://github.com/JakobChen/LabelMe_scripts/blob/master/scripts/labelme_json_to_dataset

tensorflow 数据流图解
https://zhuanlan.zhihu.com/p/27238630


https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html
https://zhuanlan.zhihu.com/p/27794982  (中文))


jupyter notebook : tensorflow note: on pc
note on deeplab v2
https://github.com/xmojiao/deeplab_v2


scene understanding 3 level:
The construction of benchmark should have a multi-level an-
notation, including low-level participants annotation, mid-
level trajectory annotation and high-level relationship anno-
tation.

### Interesing Project
google's robostic arm: Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning


My plan:
To read paper and write the thesis and report. send the basic content to patrice. the paper include <Deep Learning: A Critical Appraisal >


Deep learning 
For   most   problems   where   deep   learning   has   enabled transformationally  better  solutions  (vision,  speech),  we've entered diminishing returns territory in 2016-2017. 
François Chollet, Google, author of Keras 
December 18, 2017 
‘Science   progresses   one   funeral   at   a   time.'   The   future depends  on  some  graduate  student  who  is  deeply  suspicious of everything I have said. 
Geoff Hinton, grandfather of deep learning 
September 15, 2017 

Limits :
Deep learning thus far is data hungry .
Deep learning thus far is shallow and has limited capacity for transfer .
Deep learning thus far has no natural way to deal with hierarchical structure .
Deep learning thus far has struggled with open-ended inference .
Deep learning thus far is not sufficiently transparent .
Deep learning thus far has not been well integrated with prior knowledge .
Deep learning thus far cannot inherently distinguish causation from correlation .
Deep learning presumes a largely stable world, in ways that may be problematic .
Deep learning thus far works well as an approximation, but its answers often cannot be fully trusted .
Deep learning thus far is difficult to engineer with.


Unsupervised learning.
Symbol-manipulation, and the need for hybrid models.
More insight from cognitive and developmental psychology .
Bolder challenges


Notes:
Stixels are specific to street scenes. They exploit prior information and the typical geometric layout to
yield an accurate segmentation.

classic process:
1. computation of bottom-up image segments. eg rely on such superpixels but computed at multiple scales to retrieve a set of overlapping segments.
2. feature descriptors for the bottom-up image regions are computed in order to facilitate their classification into the
semantic classes. The feature extraction step is typically split into two parts.

First, one or multiple pixel-level feature representations
such as textons (Malik et al., 2001), SIFT (Lowe, 2004),

Second, these vectors are pooled over all pixels within the region in order to ob-
tain a fixed length, compact, and discriminative descriptor of the re-
gion.
3. the region descriptors are classified into the semantic classes of interest by leveraging standard classifiers. Eg. SVM, KNN, Adaboost
4. the final pixel-level labeling is inferred.

Deep learning:
dilated convolutions that can circumvent the downscaling in
the network altogether while maintaining its receptive field, which
is important for context knowledge.

The CRF can be simply appended to the network
as an individual processing block


Only a few works have focused on efficient semantic labeling for
real-time autonomous driving applications.


Indoor scenes typically impose less overall constraints, but nevertheless exhibit a common layout that is considered by the models of D. Lin et al., 2013


https://stackoverflow.com/questions/45741254/importerror-no-module-named-utils?rq=1


correct way to install mxnet
https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/mxnet/


    ImportError: No module named parse    
    solution: The app you're trying to run requires Python 3, and you're trying to run it on Python 2.

    mportError: numpy.core.multiarray failed to import
    solution: 1. When I ran into this, it was because I had another version of numpy installed on my machine in a different location -- I had to uninstall the other (older) version.
    1. Therefore:  pip install -U numpy

### Day13 Augest 20, 2018
1. Simple does it weakly suprivised segmentation:  works! but training 9 hrs just finished 16%. so it need 50+ hours totally.
2. github: interactive segmentation RGBD:  PyQt5/QtGui.so: undefined symbol: PySlice_AdjustIndices  好像也是没空间。 (pytorch :())
3. github: Rednet: out of memory
4. BDWSS (MXNet):  out of memory
5. modular_semantic_segmentation: lack of data 


### Day 14
1. sensor fusion，就是用两套传感器组件同时测量一个数值，然后两个测量结果A跟B进行一下加权平均得到最终的结果(Fusion这个词是融合的意思，可以理解为A、B结果的融合)。为什么要使用fusion技术呢？就是为了得到更精确的“结果”。
2. for sudo input in jupyter notebook
   https://stackoverflow.com/questions/44996933/using-sudo-inside-jupyter-notebooks-cell


modular_semantic_segmentation failed
1. missing disparity map file, right_image for cityscapes dataset
2. missing npy file for synthesis RAND dataset

Now try RedNet again

### Day15
it tooks long time to perform the segmentaiton with depth map.
firstly, I tried 2D 
several paper and relative solutions are tried.  
the network based on Fully con


The experiment is for only 1 class: sky. After manually make the segmentation label for 
FC-DenseNet56(2016):          500

MobileUNet(2017):             230
DeepLabV3_plus-Res50(2018):   215

 Encoder-Decoder:       
 RefineNet-Res50
 PSPNet-Res50  (2016)         512
 ApapNet(2018)
If training  with public datasets CamVid with 32 classes, the time will take much longer. there is the results.


for this step, i chose the fastest solution.


secordly, I tried to train the disparity map.
1. I tried to training 
   set 
   
2. depth map fusion require more on hardware. Several solution failed during the training because of the GPU memory.
   so review the backbone with fastest
3. layer should be limited in 50.
4. interactive sematic  segmentation
5. currently beyas / dirichlet fusion are under testing. I will review the result.


### Day 16 
For Deployment:
1. finished get start tutorial part1-5
    1. install docker-machine
    2. insatll virtualbox
    3. stuck with
2. waiting for aws account recovery

### Day 17
Deployment:
1.  Docker get-started
2.  tensorflow serving
   1. I thought serving can be used directly with serve script，then start training. but after building the system, I found it deployed the trained model and return the feedback of the user input. It's different with what I expected. Meanwhile, the interface and libs are imcompatible. still lots of bugs. 很多坑，
   2. 
3.  CloudFoundry + flask + pickle
4.  aws sageMaker
5.  

### Day 18
1. how to restore pre-trained checkpoint 
2. predictions[key] = deeplab.predict(images)
3. change class from CamVid to Cityscapes  (12 to 30)


### Day 19
Real time segmentation
1. https://averdones.github.io/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/
http://www.scikit-video.org/stable/

2. BiSeNet on suit and  https://arxiv.org/pdf/1808.00897.pdf
3. Coda forklifts class training


### Day 20
1. Basic grammar learning for kotlin.
2. snpe c++
3. Coda breaking rule fuctions: add centre retangle and build intersection.


### Day 21
1. Design pattern review. 
2. basic review of slam
3. leetcode finished: reverse intergrate. 


### Day 22
**Today's Progress**: 

**Thoughts** 

**Link(s) to work**
