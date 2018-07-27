# 100 Days Of Code - Log

### Day 1: June 27, Monday

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

disable the touch pad


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

### Day 3: July27, 2018
**Today's Progress**: 
stuck on the training own data on detectron.
1. convert dataset into /lib/datasets in json formate.
2. change number of categories.
3. finetune

steps:
1.    Create a COCO-like dataset
2.    Add it to the dataset catalog (lib/datasets)
3.    load the pre-trained model and modify the output layers
4.    train the model using train_net.py
5.    edit yaml for configuration.
   

**Thoughts:** 
**Link to work:** [object detection](http://www.example.com)
