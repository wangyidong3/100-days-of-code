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
   