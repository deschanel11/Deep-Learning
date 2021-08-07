# 1. You Only Look Once: Unified, Real-Time Object Detection
  
link of paper : https://arxiv.org/abs/1506.02640
  
 
#### overall indexes : 
- Abstract
- 1. Introduction
- 2. Unified Detection
- 3. Comparison to Other Detection Systems
- 4. Experiments
- 5. Real-Time Detection In The Wild
- 6. Conclusion
- References



### Abstract
  
prior work on object detection : repurposed classifiersto perfrom detection.  
<->  
YOLO :  
frame object detection as a regression problem to  
spatially separated bounding boxes,  
and associated class probabilities.  
  
A single neural network  
-> predicts bounding boxes  
and class probabilities  
directly from full images in one evaluation.  
  
  
It can be optimized end-to-end directly on detection performance,  
cause  the whole detection pipline is a "Single Network".  
  
  
- base YOLO  
> - processes images in real-time at 45 frames per second  
  
  
  
- Fast YOLO  
> - smaller version of the network
> - processes an astounding 155 framges per second, while achieving double the mAP of other real-time detectors. 
  
  
YOLO makes more localization errors compared to state-of-the-art detection systems(state of the art : 최첨단의),  
but it's far less likely to predict false detections where nothing exists.  
  
(* localization error : difference between true position and estimated position)  
  
-> YOLO learns very general representations of objects,  
and outperforms all other detection methods(which is DPM, R-CNN) by a wide margin  
when generalizing from natural images to artwork on both the Picasso Dataset and the People-Art Dataset.  
  
  
### 1. Introduction
  
To detecct an object :  
current detection systems take a classifier for one object and evaluate it at various locations and scales in a test image.
  
- DPM(deformable parts model) -> use a sliding window approach, where the classifier is run at evenly spaced locations over the entire image.  
(classifier가 전체 이미지에서 균일하게 위치되며 동작하는 슬라이딩 윈도우 방식, 해석 : https://blog.daum.net/sotongman/10 블로그 참고)  
  
  
processes like "generating potential bounding boxes",  
"run classifier on these proposed boxes",  
"post-processing to refine the boinding boxes",  
"eliminate duplicate detections",  
"rescore the boxes based on other objects in the scene"..  
  
  
=> this pipelines are slow and hard to optimize,  
because each individual component must be trained separately.  
  
<->  
  
YOLO => reframe object detection as a SINGLE REGRESSION PROBLEM, straight from image pixels to bounding box coordinates(좌표들) and class probabilities.  
  
  
  ##### 1. YOLO is extremely fast.
  
  - base network -> batch processing 없이 Titan X GPU에서 1초당 45프레임을 학습한다.
  - fast version -> 150fps보다 빠름
  - real-time system의 mAP(mean average precision)의 두 배의 mAP를 달성한다.
  

  
  ##### 2. YOLO reasons globally about the image when making predictions.
  
  
  - YOLO와 sliding window, region proposal-based techniques의 차이점
  > - YOLO는 training과 test time 동안 entire image를 봄. 따라서 암묵적으로(implicitly) 
  (class의?)appearance뿐만 아니라 class에 대한 contextual information도 함께 encode함.
  
  
  
  ###### Fast R-CNN은 top detection mehod이지만 객체의 이미지의 background patches에서 실수를 함. 더 큰 context를 볼 수 없기 때문.
  ###### 하지만 YOLO는 background error을 Fast R-CNN의 반보다도 덜 만듬
  
  ##### 3. YOLO learns generalizable representation of objects. (representation : 대표적인 특징.)
  
  하지만 욜로는 빠르기는 하나 최첨단 시스템보다 정확도는 떨어짐. 또한 몇몇 객체들을 정확하게 localize 하는 건 어려워 함. (특히 작은 객체에 대하여)
  이후 이 trade-off에 대해서도 다룰 것.
  
### 2. Unified Detection

