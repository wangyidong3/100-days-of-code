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
python -c 'import nump;print numpy.__path__'
rm -rf .git

very very good instruction for deeplab:
https://lijiancheng0614.github.io/2018/03/13/2018_03_13_TensorFlow-DeepLab/

how to preprare your own dataset
http://hellodfan.com/2018/07/06/DeepLabv3-with-own-dataset/   (chinese: https://blog.csdn.net/u011974639/article/details/80948990)

generate some label and then 

从基本功开始，先看segmentation
https://www.scipy-lectures.org/packages/scikit-image/index.html
纹理识别
https://blog.csdn.net/u011974639/article/details/79887573

https://sthalles.github.io/deep_segmentation_network/
https://www.zhihu.com/question/24529483   (weight decay,  optional)

制作数据流程：
https://blog.csdn.net/u010402786/article/details/72883421

https://blog.csdn.net/zoro_lov3/article/details/74550735
https://blog.csdn.net/m_buddy/article/details/78667813
https://oldpan.me/archives/image-segment-make-voc-datasets

语义分割综述
https://zhuanlan.zhihu.com/p/37801090

最最最重要的视频
https://www.bilibili.com/video/av17204303/

还是要先看知乎呀。
传统分割方法，都是基于颜色纹理的初级特征。语义分割才是2010年后出来的
https://zhuanlan.zhihu.com/p/30732385

### Day Augest 10 2018
1. mask overlap for texture, color and disparity.
2. prepare the  index file and TF record.
3. add new class, include color, class name and number.
   

http://paduaresearch.cab.unipd.it/10337/1/pagnutti_giampaolo_tesi.pdf

https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef

https://zhuanlan.zhihu.com/p/25013378

https://stackoverflow.com/questions/45247717/how-to-segment-the-connected-area-based-on-depth-color-in-opencv

https://github.com/aleju/papers/blob/master/neural-nets/Fast_Scene_Understanding_for_Autonomous_Driving.md



### Day 12 Augest 12 2018
Note for deeplab
https://github.com/xmojiao/deeplab_v2