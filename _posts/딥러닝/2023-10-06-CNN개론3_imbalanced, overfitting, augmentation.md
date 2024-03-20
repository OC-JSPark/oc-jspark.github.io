---
title: "CNN개론3 : imbalanced, overfitting, augmentation"
escerpt: "imbalanced, overfitting, augmentation"

categories:
  - DeepLearning
tags:
  - [AI, DeepLearning, imbalanced, overfitting, augmentation]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-10-06
last_modified_at: 2023-10-06

comments: true
 

---

# 10. 데이터 불균형(imbalance)

- 모델입장에서 epoch를 돌때마다 train dataset에서 예측을 하고 정답과 비교를 통한 값을 역전파시키고 업데이트 되는 과정이다.

- dataset의 양이 적다면 다양성이 적다. 그러므로 dataset양을 늘리면서 다양성도 늘려서 비중을 적게 가진 class를 확장시켜 비율을 비슷하게 맞춘 후 학습을 시키는 방법이 필요하다

## 10-1. anormaly detection
- 1) augmentation 
: 작은비중의 class 증강

- 2) normal 만 사용.
: semi-supervised learning이다. 
  - 비정상,정상이 있다면 정상data만 있는 boundary를 찾을수 있다.그러면 그외께 들어오면 boundary 밖이므로 비정상임을 바로 알수있다.
 
## 10-2. class imbalance 해결법
- 1) oversampling 
: 단순히 data 중복 복제하는기법
  - **데이터 양만 증가, 다양성은 x.**
  - 만약 다양성이 충분히 확보된 dataset이라면 oversampling하여도 효과를 볼수 있다.

- 2) Data augmentation
: **data의 다양성을 증가시키기 위해!**, 성능은 늘어날 확률이 적다.


- image를 pre-process 할때 다양한 transformation(변화)을 진행한다. 
- 다만 data augmentation할때는 부족한 class만 진행하면 문제가 생긴다.
- 왜냐하면 tranform했던 feature가 다른 class에는 없는 feature이기 때문에 무조건 해당 feature가 있는 tranform한 class로 classification되기 때문이다. 
- 그래서 학습하는 모든 class를 함께 augmentation해줘야 한다. 이렇게 되면 예측 accuracy는 기존과 결국 변화가 크지 않을수 있다! 다만 dataset의 다양성만 증가된다. 

### 10-2-1. transformation
: **data augmentation시 transformation을 적용시키는데 이를 많이 해줄수록 epoch도 증가시켜줘서 학습횟수를 늘려줘야 한다!**


- p=확률 : transformation을 적용할 확률 
- 이를 p=0.5로 둔다면 이를 통과한 image는 transformation 된거와 안된거 2개가 나오는것이 아니라 둘중에 하나의 결과만 나온다.  
- 그러므로 우리는 **transformation 을 많이 해줄수록 epoch를 증가시켜줘야 한다.** 
- 즉, dataset을 한바퀴 다 돌리는 횟수를 증가시켜 줘야 한다. 
- data augmentation의 목적은 데이터의 다양성을 늘리는게 목적이다. 하지만 다양성만 늘어나고 epoch가 늘어나지 않는다면 ...
- 즉 epoch가 1이라면 augmentation을 여러개 넣어서 데이터가 다양한 transformation이 되었을때 우리는 해당 dataset을 한번에 학습하기가 어렵다. 이럴때는 epoch를 늘려줘야 동일한 dataset을 돌지만 일정한 확률로 학습이 진행된다. 

## 10-3. Focal loss
: 적은비중을 가진 class에 Focus를 맞추겠다는 의미이다.

- 참고자료
  - [Focal Loss for Dense Object Dection 논문](https://arxiv.org/abs/1708.02002)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/45c614fa-ad00-4908-826c-01d237df633e)

  - Pt: 등장확률
  - dataset에서 많은  %를 가지고 있는 class에 대해서는 작은 focal loss value를 가지게 한다.

  - class imbalaced에 대한 근본적인 원인은 무엇인가?   
    - 어떤 dataset의 비율이 class별로 차이가 있기 때문에 model이 비율이 큰 class를 훨씬더 예측을 할때 치우친 예측을 하는것이 문제였다.

  - 예시)
    - car는 loss가 적다. truck은 loss가 크다. 이것의 의미는?

  - 만약 우리가 가끔 등장하는 class를 예측이 틀렸을때 loss를 10배로 크게 주었다면 즉...**우리가 loss를 구하고 loss를 역전파 시키면서 가중치를 업데이트시키는데** loss를 크게 하면 그것에 대해 10배더 학습을 많이하게 된다. 
  - 즉, 작은비율을 가진 class를 틀리면 훨씬더 학습을 많이하고, 많은 비율 가진 class를 틀리면 조금덜 학습을 하게 하는것이다. 그렇게 학습에 대한 가중치를 부여. 그래서 우리는 **비율이 적은 class에 대해 조금더 민감하게 학습을 진행할수 있게 된다.**


# 11. 오버피팅(overfitting, 과적합)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/fb929328-c786-42b2-ad43-d452d1633233)


  - 학습의목적 : 일반화(generalization)성능 높이기

  - 과적합의 의미
    - 1) dataset중에 model이 학습하는 train dataset에 너무 fit이 되어, val과 test에 대해 예측시 성능이 떨어지는 경우
    
    - 2) 전체dataset에서는 좋은 예측값을 가졋지만 real dataset에서는 나쁜 예측값을 주는경우

  - real world에 존재하는 dataset의 수많은 **다양성**을 충분히 포함하는 dataset을 구축하는것이 근본적인 해결방법이다. 

## 11-1. overfitting 감지하는 방법

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/54cf7d0f-c08b-4dc9-9306-17f7db9b6b1b)

  - case 1
  : 학습이 잘된 경우(일반화 성능이 좋아지고 있다고 표현)
    - 훈련손실 = loss : loss가 감소할수록 예측과 정답의 차이가 감소하며, 학습이 잘된다고 표현한다.

  - case 2 
  : train에서는 학습이 잘되지만 검증에서 어느순간부터 validaion loss가 늘어난다. 
    - 이 의미는 train dataset에 대해 너무 over fitting이 되어 있는것이다. 즉, validation dataset에 대한 다양성은 현재 model이 담아내지 못하고 있단 의미이다. 

## 11-2. overfitting 방지하는 방법

### 11-2-1. 일반화(Regularization)항 추가

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f71a8c75-2b4d-406a-bd14-a2e9150f8004)

  - 일반적으로 연산이 진행될때 loss가 구해진다. 이 loss의 총합이 cost이다. 
    - 람다 : 상수로 생각.
    - w² = weight² 을 의미한다. 절대값을 사용해도 된다.
      - 절대값 : L1 normalization
      - 제곱 : L2 normalization라고 표현한다.
    
  - **"loss = 예측-정답 차이이다." 이것을 낮추는것이 학습의 목적이다!!**

  - weight를 loss에 포함시킨다. 왜?
    - loss+weight를 추가한다면 낮추는게 목적에 어긋나는거 아닌가?
    - weight값도 함께 낮추고 싶은것이다. 정확히 말하면 weight값이 너무 커지지 않도록 견제하는 것이다.   
  - 만약 어느 weight하나만 극단적으로 크다면 다른 weight들과의 연결고리는 무시되고 해당 weight간의 연결고리만 신경써서 overfit이 발생할수 있다.**(weight decay)** 

### 11-2-2. Ensemble
: dataset을 이용하여 hyper-parameter를 다양하게 하여 다양한 model_1,2,3 **집합**을 만든다.이를 이용하여 inference를 진행할 때 **voting(투표)를 통해서** 예측하는방법이다. 

- soft voting을 하게 되면 class마다에 대한 확률값이 나오는데 그것을 모두 각 class마다 sum하여 가장 높은걸 채택하면 되는 방식도 있다.

- kaggle에서는 dataset이 고정되어 있기 떄문에 model 갯수를 늘려서 voting을 하게 되면 성능이 높아질 확률이 존재한다. 하지만 계산비용과 시간이 많이드는게 단점이다.

### 11-2-3. Dropout
:특정층에서 몇개의 node를 누락시키는것. node가 누락되면 연결되어 있는 weight들이 업데이트가 안된다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e826efa1-6d8d-4cf0-92d0-03d3e8308b5c)


  - 이것이 왜 일반화의 개념이 되는가?
    - 해당 모델이 너무 특정 가중치와 특정 특성에 학습된거면 문제가 될 수 있다. 이러한부분을 일부러 누락시킨다면 너무 특정 feature에 의존하며 판단하는 오버피팅기능을 없앨수 있으며, 전반적으로 다양한 특성을 활용하면서 일반화된 성능을 낼수 있는 모델이 만들어 질 수 있다.
    - 학습을 너무 가속화하는게 아니라 학습에 방해되는 요소들을 조금씩 추가하는 방식으로 학습이 우리 dataset 뿐만아니라 real data를 포함할수 있는 일반화된 model이 될수 있다.


# 12. 이미지증강(augmentation)

- data imbalanced issue를 해결하기 위해서 transformation(변환)을 통해서 original image를 일정한 확률로 발생시킨 다음에 model의 input으로 넣어준다.
- 이러한 변화를 통해서 dataset의 다양성한계를 극복하고, real data를 더 포함시키는 방향으로 가기 위함이다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/be2c7dbc-185d-43b2-956b-c31f8147a2c3)


  - flip : 좌우상하,rotate(회전) 
  - blur : 흐리게
  - HueSaturationValue : 휴(색조,색상) + 채도:색의 선명도 를 random하게!
  - GaussNoise : noise를 추가한것. ex)film효과
  - CoarseDropout(=erasing = cutout) : 특정영역을 잘라낸것. 
  - Gray : 3채널에서 1채널로 변환하기
    -  original image 뿐만아니라 gray 도 넣기 떄문에 data다양성측면에서 훨씬 좋다.
    - gary는 r,g,b 채널의 값들을 일정한 연산을 통해서 하나의 채널로 변환한것. 그러므로 rgb 채널보다 적은 정보를 갖고있다라고 판단하지 말것! 그러므로 정보손실이 아닌 정보의 다양성측면에서 봐야한다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/3c067144-af3f-4ad8-bf65-defa58474f65)

  - 참고자료 [논문 : YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

  - MixUp : 두사진에 일정한 값을 곱해서 하나의 사진으로 만든것
  - CutMix : Cut 하고 Mix한것
  - Mosaic : (모자익) 4장의 사진을 crop해서 하나로 가져온것.
  - Blur : 흐리게!


- 이런 image augmentation한것은 실제로 input에 들어갈떄는 일정한 확률로 곱해지는 것이다. 즉, transformation 된게 다 들어가는게 아니라 random으로 input에 들어가는경우, 안들어가는경우 등이 생긴다.

- **즉, 우리는 데이터셋을 먼저 파악후, task가 무엇인지 알아야 한다. 그리고 inference에서 문제될 소지가 무엇이 있는지, 우리가 수집한 데이터의 한계, 우리가 원하는 일반화 model의 level 등을 고민하고 augmentation을 진행해야 한다.**

## 12-1. GAN(Generative Adversarial network, 생성적적대신경망)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b228b32e-158e-4da5-810b-d128ebdf4ccb)

- 참고자료[논문 :Data Augmentation Generative Adversarial Networks](https://arxiv.org/abs/1711.04340)

- 생성적 적대 신경망

- random한 noise image를 input을 받아서(Generator model) fake와 real 데이터에 대해 차이값(Discrimator model)을 개선해나가는 모델. 
  - 모델(fake인지 real인지 분류하는 모델)의 output이 real image를 맞추었다면, generator model은 해당 real image와 fake image의 차이를 loss로 계산해서 학습하게 된다. 즉, 조금더 real image에 가까워 지도록 생성하는것. 
  - output이 real이라면 fake 가중치를 real로 맞춰나간다. 즉 G모델은 D모델을 속이고 싶어한다.

- GAN으로 만든 생성이미지 dataset을 기존 dataset에 추가한다면 dataset이 증가되기 때문에 augmentation이라 한다.

- GAN으로 data augmentation진행한 data는 project의 domain에 따라서 전문가의 검수역할이 중요하다.

---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}