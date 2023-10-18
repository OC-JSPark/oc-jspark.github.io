---
title: "Computer Vision 딥러닝 비전 1"
escerpt: "Computer Vision 딥러닝 비전 1"

categories:
  - Vision
tags:
  - [AI, Vision]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-10-17
last_modified_at: 2023-10-18

comments: true
 

---


# 1.Deep learning

## 1-1. Overview
: Visual Recognition = Image(video) representation + decision making

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d5d4ba0b-07da-4d5f-ae5d-4fe20b4aad6b)

  - image의 intensity를 이용
    : gray scale, 1개의 채널에서만 사용 / 요즘은 rgb, 3개의 채널을 유지시켜서 어떤 함수를 통과시켜서 representation을 만든다.

  - view-point variation : 카메라 뷰포인트가 바뀌는경우
  - scale variation : 이미지가 커지거나 작아지거나
  - Deformation : 이미지에 와핑같은게 있는경우
  - Occlusion : 일부분이 가려져있는 경우
  - Illumination change : 조도가 너무 강하거나 낮아지거나.
  - Intra-class variation : 똑같은 사람이더라도 여성,남성,반바지입은사람 등 다를수 있다.

  - 그래서 수치적표현으로 나타내줄 필요가 있다. histogram, 딥러닝 모델 등..

## 1-2. Before the Deep Learning 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b24c135b-23ce-43e2-9186-02bd93d1ae9a)

  - 2단계가 있었다!! 이미지 represention을 하고, classification function f를 활용하여 디자인하고.

## 1-3. Deep Learning
: 딥러닝에서는 2단계를 통합함.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/204c15c7-54fd-4573-b4ba-02d5a6baca13)

  - image representation + classifier가 하나의 통합된 구조가 됨.
  - image가 convolution을 통과해서 pooling.
  - 최종적으로 하나의 vector representation을 만들고 마지막 mlp(multi layer perceptron)를 통과시켜서 하나의 classification vector를 만든다.
  - 학습시 선이나 패턴을 잡게된다.

  * pooling : CNN 에서 pooling layer는 네트워크의 파라미터 갯수나 연산량을 줄이기 위해 input에서 spatial 하게 downsampling을 진행해 사이즈를 줄이는 역할. 일반적으로 CNN에서는 Convolution layer 다음에 들어감. max pooling 외에도 average pooling, L2-norm pooling 등 다양한 pooling 방법이 있음. 
  * Pooling(downsampling)이 필요한 이유는, featuremap의 weight parameter 갯수를 줄이기 위해서.pooling layer가 없다면, 너무 많은 weight parameter가 생기고, 심각한 overfitting을 유도할 수도 있고, 많은 연산을 필요로 하게됨. 또한 Pooling을 사용하면 연속적인 ConvNet층이 점점 커지는 window를 보도록 만들어 (receptive field를 넓힘) 필터의 공간적인 계층구조를 형성하는데 도움을 줄 수 있음.

## 1-4. Deep Learning for Visual Recognition

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/ae9b497f-e8ed-4854-ac94-e661779aa1b1)

  - 딥러닝기법은 종종실패한다
  - train data가 부족해서.학습 parameter가 많은데 충분한 train data가 필요함.
  - convergence(통합)가 느리다. 예전에는 모델만 확실하면 optimization해서 정답 구할수 있음.
    - 그러나 예전보다 많은 정답을 원하니(classes가 엄청 많아짐) convergence가 느리다.
    - sigmoid functino이 gradient을 중간에 줄여서 optimize를 방해한다.
    - 모델수렴하는데 느리다.

  - 최근성공이유
    - train data가 많아짐.
    - gpu사용해서 parallel하게 작동하기에 o의n제곱에서 O(n)처럼 처리 가능.
    - dropout, batch normalization등의 방법 이용시 overffting도 막는효과있음.
  
  * optimizer : 뉴럴넷의 가중치를 업데이트 하는 알고리즘










Neural Networks and Training
CNNs
Overfitting and Network Initialization
AlexNet, LeNet, VGG
ResNet
DenseNet, SENet, EfficientNet
Efficient CNN: SqueezeNet, ShuffleNet, MobileNet
Vision Transformer 1: Self-attention
Vision Trnasformer 2: Image Processing Transfomer


# Representation learning
# Object detection & segmentation
# Video
# Multiview geometry
# 3D Vision
# Generative models and graphics


---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}