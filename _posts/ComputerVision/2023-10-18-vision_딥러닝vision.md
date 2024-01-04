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










# 2.Neural Networks and Training

## 2-1. perceptron
: single-layer neural network

- image를 input으로 삼는 perceptron 이해

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d7a5269d-af29-4b9d-a3e9-f77014e309ee)

  - input의 전기signal이 숫자 vector or tensor가 된다. 이미지의 pixel value들이 0~255값들을 input으로 받아서 modulate해준다. 그것을 weight와 bias를 통해서 summation을 통해 하나의 signal로 output을 뽑아낸다. 그리고 activation function을 통하게 하여 해당값을 죽일것인지 살릴것인지 해서 다음 neural로 내보낸다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/707441c9-aa46-4f16-b074-04eda6c675d1)

  - matcthematical form
    - input : x는 d:dimention의 벡터이다.
    - output : y라는 스칼라값을 가진다.
    - model : w라는 벡터를 말한다. w는 input size와 같아야 한다.거기에 bias scalar 벡터인 b도있다.

  - activation function : non-linear form을 줄수있기에 중요하다.
  - 이걸 안주게 되면 linear form으로 만들어져서 한번 layer쌓는거나, 여러개 layer쌓는거나 똑같게 된다.

## 2-2.training perceptron의 의미

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/73ef56e2-f1ed-4757-a185-16a9c8ab080f)

  - w parameter를 estimate한다는것.
  - 어떻게 학습하냐면 loss(error)값을 minimize하도록 학습하는것.
  - 즉, 학습은 w가 학습되는것이다.
  - 학습된 w로 모델을 만들어서 loss를 구하는것이다. 
  - loss는 주어진 데이터의 parameter를 업데이트하는 function이라고 할수 있다. 
  - gradient descent 구할때 문제점은?
    - 많은 train data필요
    - 계산하는데 오래걸림
    - 많은 epoch필요함
 * L2 distance란? 차리를 제곱한것.

## 2-3. MLP(Multi-Layer Perceptron)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/aec9496f-6c44-4ff8-84f3-6dfeec5fa0c2)


- single perceptron은 desicion boundary를 정확히 그리기 어렵지만 MLP를 사용시에는 잘 구분하에 그릴수 있다.
- MLP에서의 에러를 구하는 backpropagation
  - 중간에 chain rule로 적용시 gradient vanishing현상이 발생할수도 있다.그럴때는 drop out,scaling 등을 통해서 gradient vanishing을 막을수도 있다. 
  - loss function은  L2 distance를 사용말고 다른것도 사용가능. 왜? prediction하는게 vector form으로 prediction한다. 예를들어 classification을 한다고 했을때 classification label이 0,1,2,~10까지 있는데 그게 distribution인데 정답은 one-hot vector로 표현이 된다. distribution할때는 cross entropy loss를 이용하는게 가장 좋다.
  - H(P) : 확률분포 P가 어느정도 entropy값을 가지는지에 대한 수식이다.
  - 즉, 대칭적이지 않아서 P와Q를 바꿔주면 안된다.

## 2-4. 다양하게 쓰이는 loss function

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/6c8a06f1-4ae7-4528-8c41-a048a42ac52e)


  - cross entropy loss(for classfication in general)
    * cross entropy의 특징은? KLD(Kullback-Leibler divergence, 쿨백-라이블러 발산)은 두 확률분포의 차이를 계산하는데 사용하는 함수로서, 어떤 이상적인 분포에 대해 그 분포를 근사하는 다른 분포를 사용해 샘플링을 한다면 발생할 수 있는 정보 엔트로피 차리를 계산한것.

  - multi-label soft-margin loss
    - 이미지보면 다양한 class가 있을수 있다. cross entropy는 확률,분포 가정을 따라야하기 때문에 이런경우 표현하기가 어렵다.(binary classification loss에 대한 표현 못함) 그래서 각 class에 대해 multi-label soft-margin loss를 사용하게 된다.
    - c는 클래스 갯수를 말하며 d는 mini batch sample수를 말한다.
  - Mean squared error( for regression)
    - regression task는 classification과 다르게 one-hot coding을 쓰는게 아니기 때문에 하나의 value가 나온다.

## 2-5. SGD(Stochastic Gradient Descent)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/6f584b9a-99d0-4c27-a052-2397fd19cc34)

- 학습이 loss값을 구하는 loss function을 구했다면 이제 gradient decent를 통해서 global minimal을 향해서 optimization을 해야한다. 각샘플,,즉 이미지하나를 보고 update를 하면 시간이 엄청 오래걸린다. 그리고 image가 잘못 label 되어있다면 문제가 된다. 

- 각각 샘플에 대해 weight구할때 : n번째 샘플에 대해서 loss를 구하고 gradient를 사용해서 update를 진행한다. t:time 번째 weight를 t+1번쨰 weight에 넣어주면 된다. 장점은 빠르게 업데이트가능. 단점은 만약 label이 잘못되어있다면,,,noise에 sensitive하다는것. 그러면 잘못된곳으로 optimize할수 있다.
- 그래서 사용한 technic이 minibatch SGD(Stochastic Gradient Descent)이다. 이건, 전체 샘플이 있다면 일부만 샘플링하여 그 weight를 업데이트 하는 방식이다.  수식을 보면 n번째 샘플을 쓰는게 아닌, batch갯수만큼의 샘플들을 다 붙여서 batch수만큼 업데이트 해주는것이다. 그래서 하나의 샘플을 사용해서 loss를 계산하는게 아닌 minibatch개의 샘플을 사용해서 loss를 update하여 준다.장점은 매우 빠르고 noise에 robust하다.
- minibatch가 아니라 한번에 모든 샘플을 다 업데이트할수도 있다. 그러나 이미지가 결국 224x224 개수 x 3channel만해도 이것만해도 엄청 큰 벡터 demention이 들어온다. 이렇게 모든 샘플을 다 보고 할수 없으니 minibatch를 사용하는편이다.
- 보통의 optimization techiniques 딥러닝은 SGD기반으로 만들어져있다.
  ex.ADAM, ADAM-W,  

## 2-6. Momentum

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/6b479f9a-ce29-41a1-9614-199d22b85392)


  - 최적화 테크닉에는 momentum이란것도 있다.SGD할때 이전 direction을 기억하는것이다. (즉, 움직임을 기억한다는뜻.) 장점은 빠르게 수렴한다. 샘플 update할때 value weight를 추가로 줘서 update한다 

## 2-7. MLP학습시 이슈는?

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b572d954-f397-4d23-9e9a-0264409c3dd4)

  - overfitting 
    : model parameter, 즉 w가 많았다. 이런게 generalize 될려면 train dataset에 너무 fit하면 안되는데 너무 fit된것. 해결법은 더많은 train data를 사용하는것 or regularization 기술(ex.dropout)을 사용하면 된다.

  - 학습시간이 너무긴것
    : parameter수가 input개수만큼 엄청 많다.결국 계산시 너무 오래걸린다. 그래서 gpu사용한다. sigmoid activation function사용시 vanishing gradient 문제가 뜨면, 즉 sigmoid함수를 보면 값이 커질수록 gradient값이 엄청 작아진다. 이럴때 gradient값이 너무 작아져서 backperception(역전파)이 어느순간 끊어진다. 그렇게 되면 낮은 layer, 즉, input과 가까이 있는 layer는 값이 너무 작아져서 업데이트가 잘 안된다. 이것이 MLP의 학습시간을 더 오래걸리게 만든다.ReLU나 sigmoid대신에 다른 variant를 사용한다.


# 3.CNNs

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/90f53b63-b5f4-4271-a1c4-db8773a5dd1c)

  - MLP를 visual data에 사용하기엔 부적절. 왜? dimention이 너무 크다.비디오는 time이라는것도 같고있다(w,h,c 이외에 time) MLP는 fully-connected layer이기때문에 부적합! 해결법은? convolution kernel사용.

- convolution for visual data
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/2fea9b0c-93ef-43dc-8bbe-54f4ec4622fc)

  - 이것도 선형연산이다. non linear한 효과를 주기 위해 activation function을 통과시키는것이다. (y = WX + B)가 convolution 연산으로 대체된것이라고 생각하면 된다.

## 3-1. convolution neuron vs perceptron

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/a378fd86-2444-41c4-9952-0ef8c03a6c87)

  - image와 kernel size가 같다면 fully-connected layer이다. 이런경우는 비효율적이다.

## 3-2. convolution neuron network 구조

- convolution neuron network는 3단계로 구성되어있다.
: convolution layers + pooling operations + MLP(fully-connected layers)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/43e8f227-a3ef-4b24-b31e-db1a60c0d99f)

  - 마지막이 3인이유는 input 채널과 같이해주기 위해서이다. 
  - weight수는 3x3x3x64 만큼이 나온다. 그것을 ReLU function을 먹여서 non-linear한 feature map을 얻게된다. local feature는 keypoint를 얻었는데 그런것과 대응되는 거라 생각하면 된다.

  - feature의 한 값이 image에 더많은부분을 보도록 하거나,줄이기도 한다. 기본적으로 max pooling만써도 잘 작동한다.
  - pooling사용이유?
  : spatial invariance를 보장하기위해.위치가 변경되어도 max pool이라는 operation은 변하지 않는다. 이미지의 정보를 abstract하기 위해. 이미지의 더 많은 부분을 보거나, pixel인데 더 많은 부분을 볼수있으니깐.원래 16개 봐야되는데 max pooling하면 4개의 parameter만 볼수있으니 memory를 효율적으로 쓸수있다. parameter 줄인다는건 결국 memory절약이 된다는 의미이다.
 

## 3-3. typical cnn architecture

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/9ffda554-3d34-487a-893b-81588870f611)

  - 64kernel 쓰면 64개 channel이 나온다. 채널수는 계속 늘리면서 resolution을 줄이는 형태로 네트워크가 구성된다. 왜냐하면 cnn이 처음나온게 이미지자체를 하나의 벡터로 인코딩해서 classification을 해결하기 위해 디자인되었다. 그래서 spatial size는 줄여서 estruction시키고 채널은 키워서 표현할수있는 범위를 키워주는 형태로 디자인되었다. 3짜리 input을 64로 키우고 128채널로 키우고 1024채널로 키운다음에 global maxpooling 해서 하나의 채널로 바꾸고 그것을 fc layer 몇개를 통과시킨후 1000개의 prediction을 하기위해서 1000개를 남긴후에 남긴 1000개를 softmax를 통과시켜서 확률분포의 가정을 맞추고 cross entropy를통과시키는 형태


  - 낮은layer에서는 edge,blob을 캐치하고있다. 높은수준의 layer에서는 best prediction 된 feature를 캡쳐하고 있다. 그래서 texture나 shape등을 알고있다.

## 3-4. Learning CNN

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/c7e434bf-76c3-44b6-a9e4-f4da6d1fc486)

  - MLP학습하는것과 같다. SGD의 backpropagation을 이용해서 학습하게 된다.
  - conv를 통과할떄마다 spartial size는 작아지고 channel은 점점커진다. 그리고 fully connected layer를 통과한 후 loss를 계산하게 된다. 그리고 one-hot vector를 열어서 계산한다. backward할때는 error에 대해서 8번쨰 layer의 weight의 미분값을 계산하고  7번쨰 layer의 weight의 미분값을 계산하고 이것을 최소로 하는 0을 만드는 어떠한 값을 찾아서 w값을 업데이트한다.

  - MLP와 전반적으로 같지만 image가 input으로 들어오면서 convolution neural network가 나옴. 그렇지 않은 데이터는 MLP를 써서 학습시켜도 충분하다. convolution이 locality 보장, 지역적인 pattern을 캡쳐할수있다는 지역성도 보장, pooling operation을 통해서 high level 도 인코딩할수있다는 장점이 있다. 


# 4. Overfitting and Network Initialization

## 4-1. overfitting

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d2d9b8a2-3985-4a20-9c92-d4685fedaec9)

  - overfitting : 일반화(generalization)가 잘안된것. 
  - train data는 parameter수보다 많아야 한다.
  - 그래서 train data를 더이상 안느리고 overfitting 피하는걸 연구함.

## 4-2. tricks to avoid overfitting

### 4-2-1. Dropout
: 랜덤하게 neural을 꺼주는것.요즘은 dropout 잘안씀.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/2e923fc4-2c67-4d35-8222-3da7c65d78e3)
  
  - dropout은 보통 fc layer에서 사용한다.

* 앙상블 : prediction을 여러개해서 중첩후 더해서 여러개 모델의 효과를 같이누리는 data처리기법이다.

### 4-2-2. weight decay
 
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/c54289b4-cb37-49d0-9a79-ba6a2d34d605)

  - weight값넣어줄때 어느정도 이상 커지지 않게 막아주고 weight을 지정하는것.
  - weight decay를 하면 generalization이 더 잘된다.


### 4-2-3. Early stopping

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b55418ff-f58c-4a74-b2e5-3885fee46e13)

  - train과 val dataset으로 나누게 된다.testset과 겹치면안된다.
  - 간단하지만 overfitting을 피하는 가장쉬운방법

## 4-3. Network Weight Initialization

### 4-3-1. Learing from scratch

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/cf854ed4-96c9-4697-9940-536b5b41d950)


  - scratch: 초기상태
  - pre-train된 모델이 없을때, 즉 scratch상태부터 학습을하고자 할때 작은 random number로부터 모델을 학습시켜야 한다.shallow network에서는 zero mean과 0.01의 standard deviation값으로 모델 initialization하면 잘작동한다. 그러나 deep network로 가면 문제점이, activation이 0으로가서 gradient vanishing같은경우가 발생할 수 있다. 예를들면 10개의 neural network를 가진 activation의 distribution의 탄젠트h를 보면 뒤로갈수록 0값의 activation 값이 나오는걸 볼수있다.

  - 그래서 나온 기법이 Xavier initialization이다.
    - weight를 zero mean과 variance를 distribution으로 부터 Initialize하는데 input,output값을 hyper parameter값으로 사용해서 weight를 initialization해주는것이다.
    이렇게 하면 그문제를 해결할수있다.
    - tanh에 대해서는 잘분포가 되도록 할수있는데 ReLU를 쓰면 문제가발생한다.  
  - random initialization을 해도 충분히 target task에 대해 잘 동작하는 편이다.(보통, tensorflow나 pytorch에서는 크게 위문제를 신경안써도 되더라.)

### 4-3-2. Finetuning an exiting network
: 기존 network에서 parameter를 가져와서 finetuning하는것도 모델 network의 initialization의 좋은기법으로 쓰이고 있다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/2780d178-9f3b-47f2-ba08-a54caaf8690f)


  - 이미 학습된 data를 가져와서 finetuning하는경우.(=transfer learning = finetuning)

  - target task에 대해서 finetuning하는게 일반적이다
  - representation learning, self-supervised learning(자기주도학습) for representation 이란 이름으로도 많이 불리는데, 예를들어 language model같은경우 language의 sentense에 어떤 값으로 비워놓고 그걸 representation하게 하여 그모델을 pre-train을 하고 그 pre-training된 모델을 다음 task에서 finetuning하여 쓰는경우(BERT에서 쓰는구조)
  - 그래서 ViT라해서 vision쪽에서도 pre-train하고 finetuning하는기법이 많이쓰인다. pre-train model weight를 initialize parameter로 우리가 하고싶은 모델에서 쓰면 좋은효과를 보는경우도 있다. 특히 train data가 충분하지 않을때 pre-train model을 많이 사용하게 된다.

  - future learning, one shot learning, meta learning 같은 적은수의 dataset가지고 하는 모델학습기법도 있다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/34719864-8cb3-4648-b095-1c881362bf58)

  - dataset이 적으면 앞부분 layer는 freeze하고 뒷부분 layer만 학습할수도 있다.

- semantic segmentation

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/74d53bd9-07ee-4cb5-9dcb-0ba188254ae6)
  
  - clasification 뿐만 아니라 segmentation task에서도 finetuning이 적용될수 있다(ex.semantic semgentation)
  - imagenet 은 classification task로 학습된 weight를 이용해서 최종적으로 semantic segmentation label을 통해서 network를 새로 구성할수도 있음.

### 4-3-3. Optimization Techniques of NN
: NN을 optimization하는 기술은? 주로 SGD를 사용한다. 이때 Learning rate이 필요하다

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/efec3908-5716-4f1e-be26-6962d4d90798)

  * Learning rate: jump size같은것이다. 이게 너무 크면 너무 크게 뛰고 너무 작으면 threshold를 넘지 못할수 있다. 일반적으로 크게 줬다가 적게주는게 일반적이다. pytorch,tensorflow에 다양한 learning rate scheduler가 구현되어 있다.

- learning rate이 크거나 작은걸 막기위한 솔루션

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/86a70da9-88ed-4ab1-85a7-acee53074d63)

  1. carefully하게 learning rate을 디자인하는것.
  2. SGD variation을 써서 learning rate을 맞춰가는 방법도 있다.

  - 어떤하나가 너무좋다라는건 없다. task에 따라 맞춰서 써야된다.



# 5. AlexNet, LeNet, VGG

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/657ba745-a2cb-48ad-9c6e-0e440a2ba7cb)

  - 이전까지는 ImageNet Large Scale Classification Challenge에서 시작한것. shallo layer처럼 고전적인 layer를 사용. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/159318ea-ac78-4037-b791-7eca0c320123)

  - classification CNN은 단순히 classification task뿐만아니라 다양한 task의 backbone network로 사용가능하다.이미지를 통과시켜서 feature map을 만들고 쭈욱 펼친다음에 MLP를 통과시켜서 어떤 distribution을 만든다면 이것은 image level classification이 된다. person 이나 bike의 확률이 높으면 해당 이미지에 대해 person 이나 bike로 classify한거로 예측한것이다. 여기에서 해당 feature map을 그대로  위치에 대해서  classification 후 bbox를 찾는 regression을 한다. 그래서 사람을 찾아주는 object detection task도 가능. 또는 feature map을 max pool 이나 avg pool을 이용하면 feature map이 network 지나면서 줄어든다.dense pixel level classification을 수행해서 semantic segmentation task도 풀수 있다. pixel 수준에서 모든 확률을 구하는것이다. 조금더 하면 instance segmentation, 번호까지 붙여주는 panotic segmentation도 있다.

## 5-1. CNN Architecture flow

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/363820d3-14b1-456d-9d6d-854259a521d4)

  - ILSVRC : image Large Scale net Visual Recognition Challenge
  - ResNet : residual connection 즉, 잔차연결방법을 통해서  stick connect 기법을 통해서 gradient의 흐름이 계속 끝까지 prove되게 된것이 핵심.
  - DenseNet : stick connect 뿐만아니라 다른 connection도 연결해서 더 깊게 만든것.
  - SENet : 모듈처럼 다른 network에 포함될수 있게됨.

## 5-2. LeNet-5
: CNN 중 시초격이다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/5708ea40-de58-4419-8b32-f2d4f4c754b0)

  - input이미지가 있다면 이걸 convolution을 통과시키고 subsampling(=max pooling or avg pooling 같은것)을 통해서 이미지 사이즈를 줄이고  conv 돌리고 pooling해서  fully connected를 돌려서 output을 만드는 현대의 network의 전반적인 구조를 만듬.
  - 문제점: 이떄는 gpu도 없었다.세부적인 batch norm같은 technique가 없어서 느렸다.

## 5-3. AlexNet
:2012년에 나옴

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/4398b5a0-f1dc-4f63-88c1-1cde7959acc5)

  - 7개의 hidden layer가 있고 650,000개의 뉴런이 있고 6000만개 정도의 parameter가 있음.
  - ImageNet의 Large dataset으로 train 됨.
  - activation function(ReLU)와 regularization technique(dropout)을 사용함.ReLU function이 조금더 gradient signal을 보존하고 gradient vanishing을 막는데 효과가 크고, dropout을 사용해서 model의 flexiablity도 높이고, 일반화의 가능성도 강화시킴.
  - LRN(Local Response Normalization) : batch normalization의 시초격이다.

## 5-4. VGG

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/fe95c1ce-2603-4084-8ef2-a45e294af82a)

  - 조금더깊게 16개와 19개 layer를 쌓음(Alexnet을 7개 layer였음)
  - 특징으로 간단한 architecture를 사용
    - LRN을 없앰.
    - 3x3 filter로만 구성되고, 2x2 max pooling과 fc layer로 구성(alexnet은 처음 layer에 11x11 filter를 통과시켜서 한번에 resolution도 줄이고 인코딩도했다. 결국은 conv layer에 뒤에 ReLU가 붙고 non linear activation이 생기고 또 거기에 conv layer 2개를 통과시켜서 receptive field가 더 커지니깐 이러한 점을 활용해서 3x3을 vgg에서 filter로 넣음.) 다만 깊어질수록 채널수가 늘어난다. 1000개의 class를 classification한다. 그리고 softmax를 통해서 가장 확률이 높은 confidence score를 높여주는 역할을 한다. 
  - 3x3 연산이 gpu 최적화하는데 좋은 연산이라고 한다.
  - ICLR2014까지는 c++로 구성(이땐 tensorflow, pytorch가 없었음)
  - finetuning없이도 다른 task에 generalization이 잘됨.

  - vgg는 input으로 224x224 RGB image를 받는다.
  - 3x3 conv layer를 많이 쌓는게 좋다라는 최초의 연구.(즉 적을걸 많이하는게 좋다. 큰거하나보다!)

  - VGG 학습방법은?
    - mini-batch(256) gradient descent와 momentum 0.9로 줌
    - 2개의 앞에 FC layer에 대해서 dropout하고 weight를 losstum에 넣어줌.
    - learning rate=0.01로 initialization하고 3epoch마다 learning rate decay를 해줘서 optimization을 촘촘하게 줄여나갈수 있게 해줌
    - 370,000 iteration과 74번의 train epoch를 수행.
    - model initialization이 중요한데 11개의 layer에 shallow network 하고 (이건 가우시안 distribution으로 initialization된거) vgg16, vgg19를 학습할떄는 4개의 conv과 마짐가 3 fc layer를 11 layer로 학습을 하고 나머지 중간에 있는거는 random gaussian으로 진행. 즉. two stage training을 진행한것이다.11개짜리의 layer를 먼저 학습하고 parameter를 넣은다음에 가운데 layer는 random gaussian으로한것.


## 5-5. Explainable CNN (Layer-wise Network Visualization)

- 참고자료
  - [Explainable CNN github](https://github.com/ashutosh1919/explainable-cnn)






# 6. ResNet

## 6-1. revolution of depth(깊이의 혁명)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/5ce6403f-a702-4282-b1fa-7c4754a8eb07)

  - depth는 중요한 요소이다
  - depth가 깊어질수록 receptive fiels(수용자)가 커진다. semantic 한 concept를 큰 context에서 알아낼수 있다.
  - 또한, 더 깊어질수록 non-linear function이 더 중첩되기 때문에 더 큰 non-linearities를 얻을수 있다. 그래서 더 richer하고 complicated한 feature space를 모델링할수 있는 기법을 제공해준다

  - alexnet은 8 layer.
  - vgg는 19 layer
  - ResNet 152 layer까지 쌓을수 있는 방식이 생김

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/8ae49d3b-f9cb-4372-b9b3-af6a5faeda74)

  - 네트워크를 깊게쌓는것의 문제점은 backward gradient, forward response가 너무 쉽게 사라지거나 폭발한다는점.
  - 이걸 더 쉽게 쌓는걸 가능하게 한점은?
    1. ReLU activation : sigmoid activation은 끝으로 갈수록 gradient가 작아진다
    2. 적절한 initialize 테크닉.
    3. batch normalization : batch norm은 딥네트워크의 train을 강조한논문이다. 

  * activation(활성화) : 입력신호의 총합을 출력신호로 변환하는 함수

- Initialization Under ReLU Activation
: 30개의 layer를 쌓으면 converage할수도있지만 안할수도있다.그래서 이런문제가있으니 batch normalization이 나왔다.

## 6-2. batch Normalization

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/1f2ca772-ae54-4e47-aebf-99f70f0ea26b)

  - 이것의 효과? Internal covariate shift해줘서 잘됨
  - Internal covariate shift ?  각레이어들이 training되는 동안에 parameter가 뒷부분에 따라서 바뀐다. 그것이 지장이 없게 layer마다 계속 normalizaion을 해줘서 똑같은 signal들이 연장될수 있게!
  - forward propagation에서는 hidden layer 1 에서 output이 나왔으면 hidden layer 2에서는 그 output가지고 일을함. 그래서 normalization을 다시 해줘서 update를 진행하면서 forward propagation을 진행한다.
  - 그러나,다시업데이트하면 output이 바뀐다. 그럼 기존에 있던 값과 달라지니깐 헷갈릴수 있잖아? 그래서 학습을 느리게하고 어렵게 만듬. 그래서 batch norm이 아나온것임!!

  - backward propagation을 하면은 한번에 업데이트한다.backward가 한번에 계산이 되니깐. 그래서 이전 layer의 변화에 따라서 값이 바뀔수 있다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/67f81275-02be-4c49-935d-d9efd67be58e)


  -batch normalization은 normalize하면서 internal covariate를 막아주고, varient가 바뀌는걸 막아주고,zero mean과 univarient gaussian형태의 값을 가지도록 input을 normalization해주는 역할을 한다.

  - 그래서 conv/FC를 통과 후 output x에 대해서 learning mean과 learning varience를 이용해서 normalization하고 scaling&shifting은 학습되는 parameter이고, normalization은 input에 들어온값(mini batch의 mean과 standard deviation을 이용) 이걸 한 이후에 non-linearity function인 ReLU통과시켜주고 다시 다음 layer로 넘어간다. 

- 왜 batch normalization이 잘되는이유는?

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f3d1d281-7fd6-4b9b-8347-f2824873cfb5)

  - 사실 Internal covariate shift떄문에 잘되는게 아니었다! 단지 batch normalization은 optimization landscape를 smooth하게 만드는것뿐이었다. 즉, covarient shift가 아니라 optimization을 하는 field가 있다면, parameter에 따라서 loss가 최저로 가는 field가 있다.거기에서 loss gradient를 조금더 smooth하게 만들고 stable하게 만드는 효과가 있는거다. learning rate이 바뀌어도 조금더 flat한 reason에 대해서 local minimar나 global minimal을 잘 찾아갈수 있게 만들어주는 역할을 한다. 그것때문에 landscape가 smooth해졌으니깐 faster하게 되고 hyper-parameter도 덜 민감하게(원래는 조금만 바뀌어도 결과가 확 바뀜)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f454325b-1c84-40b0-8237-8cb0a8ff81ed)

  - batch normalization은 각 layer를 mini batch에 대해서 normalizing해준다. training을 가속화해서 수렴할수있게 도와주고, 첫 network의 initialization에 대해서도 덜민감하게 만들어주고, 모델이 genelar(일반화)잘되게 regularization 효과도 줌.

  - 그러면 network initialization을 잘하고 batch normalization을 함께하면 network가 잘쌓으면 학습이 잘될까? no!!즉, simply stacking을 더 많이 쌓으면 performance가 degrades되었다.그래프 보면 더 깊이 쌓으니 train,test error 둘다 늘어남. overfitting이 되면 train error는 낮은데 test error가 높은경우다.이떈 모델이 너무 complex해서 overfitting된거다. 근데 그런게 아닌경우네!! 그래서 아래처럼 생각함.

### 6-2-1. degradation(저하,악화) problem

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/9c77fba6-9787-40cc-9004-43b01fc25439)


  - 모델이 깊어지면 optimize하는게 어렵구나!!
  - 그래서 경험적으로 verification을 한다.18개의 shallower layer에서 extra layer를 추가한다.extra layer들이 x가 들어오면 x가 나오게 identity mapping을 대략적으로 할수 있으면,깊은 layer들의 training error가 높지 않았다. 왜냐하면 shallow 는 이미 여기서 나온 결과들을 가지고 추가적인 결과를 더해주는 셈이다.만약에 멀티플한 non-liniear로 identity maapping을 approximate하면 난이도가 더 어려워짐. 그래서 output 자체에 identity mapping을 아예 넣어주고면 더 좋을걸 경험적으로 알수있었다.

- A Solution to The Degradation Problem

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f2eabf68-5c72-460f-9649-5859c3466a55)

  - 그래서 resnet에서 제안하는 방식이 residual connection을 추가함.(skip connection)
  - plain net, 즉 일반네트워크에서는 목표하는 함수 H(x)를 찾는거를 목표로 해서 network를 계산한다. 그러나 Residual net은 identity를 포함시켜서 학습을 한다. Residual Net은 함수 F(x)를 찾는게 목표가 된다. F(x)가 residual일때 조금더 H(x)를 estimate한것보다 더 쉽게 계산된다. 만야r identity가 정답이게 되면 그냥 weight값을 0으로 주면된다.optimal이 identity에 가까울수록 fluctuation도 줄어든다.

- Basic Architectrure Design
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b7f4c160-9302-490d-8598-0d06cabaa38d)

  - 심플하지만 깊은 network 구성이된다.
  - VGG스타일을 따름. 다만 가운데 skip connection이 들어감.
  - 모든 layer마다 학습을 안정화 시키기 위해서 batch normalization이 conv layer뒤에 들어간다. 그리고 주기적으로 filter size를 stride=2 conv net을 통과시켜서 size를 줄여준다.stride=2로 주면 feature map이 반절로 줄어든다.
  - 또한 fc layer가 없다. fc layer없이도 global average pooling만 가지고도 충분히 feature가 학습이 잘되었기 떄문에 fc layer없음
  - fc layer가 없으니 drop out도 없음.

- Goind Deeper

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/983773f5-80d3-44f1-8841-f16248637e84)

  - resnet가지고 다양한 블록의 구조를 search함. 
  - complexity를 가지고자 layer를 2개에서 3개로 바꿈. 즉, 4번째 layer에 대해서 residual block을 추가시켜주고, 그래서 총 152개의 layer를 만들어줌.model complexity는 vgg-16나 vgg-19보다 작고, parameter수는 적은데 더 좋은 결과를 내게 된다.

- deeper is Better

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e339a634-3249-4931-8715-6b73dcbd4078)

  - 모델size도 줄이고 inference time도 줄이면서 더 성능은 높아진 발견이다.

- How Does ResNet Work?
: 왜 resnet이 잘 되나?

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/14fde1a1-c329-4bac-880a-84cd01edc86a)

  - identity skip connection의 장점은? degradation problem 즉, layer 1개만 통과시켰을때 예측하지 못했던 문제를 매우 깊은네트워크에서 residual connection을 통해 해결함. gradient propagation이 skip connection model을 타고 통과함. 그래서 vanishing gradient 문제도 막아줄수 있다. 또한 shallow network에서 Ensembles하는 효과도 있다.  
  - 즉, gradient가 사라지지 않을 가능성이 높기 때문이라고 논문에서 주장함.







# 7. DenseNet, SENet, EfficientNet

|layer|특징|
|---|---|
|DenseNet|Add Dense Connection|
|SENet|Squeeze and Excitation module|
|EfficientNet|Network Depth, Network Width, Input Resolution search|
 
## 7-1. DenseNet

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/1e8ae9a4-529d-4519-bfed-56decaadf854)

  -ResNet은 skip-connect를 이용해서 Element-wise addition을 했다면, DenseNet은 더많은 connection을 통해서 더하고, Channel-wise concatenation을 이용함. 더한게 아니라 concatenation하면, 정보자체가 아예 보존이 되니깐 효율적으로 training이 가능하다.
  - 강한 gradient flow를갖는다 :  뒤로갈떄 모든 connection이 다 연결되어 있으니깐 하나의 큰 network 앙상블로 볼수있다.
  - 채널이 중첩되면 parameter가 많아지지않는다.
  - concat이 붙인다는 의미이다. 그래서 유지한채 끝까지 끌고가니깐 정보들을 계속 유지할수 있다.
  - resnet보다 densenet이 더 깊게 쌓았음에도 성능이 더 좋음을 확인할수 있다.

## 7-2. SENet

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/ea70e2c6-e9c0-41b7-9b80-89e06709ce9b)

  - squeeze and excitation module을 통해서 channel attention을 수행하는 부분이다. 
  - squuze : global average pooling을 통해서 channel-wise-responses의 분포를 캡처하는것.
  - Excitation : global average pooling으로 하나로 만든 attention vector를 gating을 통해서 어떤게 더 중요한지를 학습하는 것.

  - 특징
    - SE(Squuze & Excitation) modelue 이어서 어떤 network에도 적용가능.
    - computation cost를 약간만 증가시키더라도 performance를 크게 향상시킬수 있다.

  - 즉, chaanel로 만드는 FC layer를 하나 통과시켜서 weight를 주어서 channel에 gating해주는것이다. 그래서 channel에 scaling을 해서 더중요한 channel에 attended되게 학습을하는것이다.

  - 예를들면  inception module뒤에 global pooling-FC-ReLU-FC-Sigmoid-Scale해줘서 더해주면 SE-Inception module이 된다. Residual modelu에도 붙일수 있다. 그래서 chaanel atteded를 해주고 그다음 residual sum을 해주는것이 SE-ResNet Module도 있다.

  - 성능을 봄녀 CIFA_10,100의 Classification error가 있으면, 기존모듈에서 SE module을 더해주었을때 성능향상이 됨을 볼수 있다.
  - 또한 places365의 Single crop error rates에서도 성능향상확인.
  - objecte detection에도 성능확인.coco minimal validation set에 적용시 AP가 증가함을 확인. 더 bbox를 잘찾은걸 확인가능.

## 7-3 EfficientNet

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/17d19a3a-9f48-4649-9441-44d1173a864a)

  - baseline의 scling up을 하는것이 핵심이다.
  - 알고리즘 search가 아니라 grid search를 하는방식이다. 실험계획법! 이라고 하는데 많은 요소들을 다 search하는게 아니라 depth,width,resolution이 있다면 compound scaling이라는 실험계획법을 가지고 model search를 진행한다. 알파,베타,감마라는 값이 있고 그것에 특정 제곱을 한값이 depth, width, resolution이 되는데 constrate를 주고 그 모델을 찾음. 
  - 그래서 baseline이 있다면 baseline을 통과시켜서 resolution 이 있고 i번째 layer가 있고 channels수를 만들어주는 구조가 있다.
  - width scaling은 baseline구조에서 network의 width를 키운것이다(채널수를 더 많이 만들도록 하기 위해서) 그래서 더 긴 channel이 나오게 된다.
  - depth scaling은 더 깊게 만드는것.  baseline에서 더 중첩을 시켜서 깊게만드는것.
  - resolution caling은  이미지의 higher resolution자체를 키우는것이다.  예를들면 224x224 이미지일때, 448x448 이미지를 넣어보는것.그래서 얼마까지 커질수있는지 체크하는것.
  - compound scaling은  width,depth,resolution 모두 키워서 어떤 조합으로 키웠을때 가장 성능이 좋게나오는지 optimal한 수준을 찾는것.  

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/1e536d85-4c8e-45ba-80fb-9de757f5da49)
  - B7은 model parameter를 엄청 증가시킨것.
  - B0의 baseline network설정도 중요하기에 layer, channel, resolution에 대해 search한것.
    - MBConv라는 특별한 module을 사용해서 resolution과 channel도 잘줄이는 efficientNet의 B0을 정의한것을 가지고 compound scaling을 통해 키워나간것.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/18dbf7e1-01f9-47f6-a771-2c97b7ff84a0)

  - compound scling하기 전에 각각 모델의 width,depth,resolution을 따로따로 search함. 그때 FLOPS가 증가하면서 accuracy가 어떻게 증가하는지를 보여주는것.
  - 모델이 커질수록 성능이 증가하는데 어느 수준이 되면 수렴함. depth가 깊어지더라도 계속 증가하는게 아닌 d=6만 줘도 성능이 수렴하게 된다.



# 8. Efficient CNN: SqueezeNet, ShuffleNet, MobileNet
: 스몰모델을 만들기위한것.
* 스몰모델 : 더 적은 파라미터와, 더 적은 계산량으로 충분한 성능을 내게하는것.

## 8-1. Efficient CNN이란?
: 더 적은 weight은 on-chip integration을 가능하게 한다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/9dd16187-3f18-4769-80ff-0a920fbf17fc)

  - 아무리 cnn모델을 만들어도 실제 상용화 위해서는 범용성있는 gpu에서 실행할수도 있으며, 작은 chip에서도 실행시키기 위해서는 model도 작아져야 한다. 그래서 cnn model weight를 줄여야 한다.
  - 그러기 위해선 몇가지의 tecnique를 통해 모델사이즈 줄일수 있다.
    1. Quantization(양자화) : 실수형 변수를 정수형 변수로 변환하는 과정
    2. Pruning(가지치기) : 학습 후에 불필요한 부분을 제거하는 방식

  - 기술적인게 아니라 아키텍처에서 조절하는 방법에 대해서 보자.

  - bestCNN은 가장 정확한 cnn이 아니다.
  - 즉, 좋은 퀄리티의 아웃을 내면서도 충분한 feature map을 만들수있어야한다.일반적인 app은 100mb 넘지 않는다.
  - snapdragon 865 gpu ~ 1450 Gflops에 들어가야 좋다.
  - 배터리문제도 고려해야한다. 
  - ex. mobilenet이 가장 대표적인 모델이다.


## 8-2.small model advantage
1. 에너지 효율 줄일수 있다.
2. CNN model같은거에서 data gathring도 더 적게하여 상용화 가능하도록 함.
3. 즉, 작은 chip에 cnn model을 이식하고 sensor에다가 붙여서 실생활에 사용가능하도록 조금더 용도에 특화된것.
4. realtime으로 자동차가 움직일때 자체 online training을 하고싶을때,새로운 이미지나 상황이 발견되었을때 계속 재학습을 할수 있고 또 그모델을 업데이트 할수 있는 장점이 있다. 데이터가 취득되면 클라우드나 서버쪽으로 전송이 되어서 업데이트되어 돌아올수도 있다. small model쓰면 이렇나 latency가 줄어들게 된다.

## 8-3. efficient하게 만드는 방법은 뭐가 있나?

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/108867c9-a4e5-4b8c-ad62-6d37ce86ad45)
  - depth에 따로따로 웨이트를 주어서 웨이트수나 연산을 줄이기
  - channel에서의 spartial sifting을 통해서 모델내부에서의 feature 간의 inter corealation을 고려하는것.
  - low-precision computation : 지금은 float32를 쓴다면, 점점줄여서 unsigned int, 더줄여서 2bit, 3bit 등의 network를 만드는것.
  - pruning을 통해서 network parameter를 잘라서 없앨수도있다.

## 8-4. SqueezeNet
* SENET은 squeeze and excitation module을 통해서 channel attention을 수행하는 부분이다. squeeze는 global average pooling을 통해서 channel attention을 만들어내는 방식이었다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/a425a6a2-b827-4f89-b9de-551b55441e23)

  - strategy1:여기서의 squeeze는 원래있던 network에 3x3 filter를 1x1 filter로 바꾼다. 3x3 filter이나, 1x1 filter나 input,output의 featrue resolution, channel 은 같다. 그래서 9개 더 적은 parameter수를 줄임. 

  - strategy 2 : 들어오는 input channel수를 줄임  
  - strategy 3 : conv의 activation maps을 뒤로 보냄. 앞에서는 줄어든 파라미터만큼 손실이 있었을껀데 이를 feature resolution을 키우면서 feed forwarding을 하면서 더 좋은 표현력을 얻게된다. 그래서 delayed downsampling을 하면 activation map이 커지고 classification accuracy에 조금더 기여할수 있게 된다.(이렇게 하면 모델 parameter수는 줄지만, 실제로 동등하게 size를 줄이면 연산수는 줄어들게된다.즉, 이건 연산수는 늘어날수있다.)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/71c0f394-c6ae-428b-9312-f239a30ac633)

  - squeeze module이 1x1 conv을 통과시키고 expand module에서 1x1과 3x3 conv filter를 중첩시켜서 expand후 concat진행한다. 이 2개합친것을 fire module이라고 한다.

- Architecture
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b2882850-43fb-4a9f-8efe-4176e150cd5d)
  - image가 들어가면 3x3 conv를 통과하고 maxpool한번하고 fire 진행한다. 그리고 maxpool하고 fire하고 gap(global avgpool)을 통과시킨다.
  - 이중에 skip connection을 추가해서 학습을 용이하게 만듬.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/a3c883e6-6850-4787-be2b-73b02d64ec06)

  - squeezeNet은 1.2M parameter를 가지고있다.
  - 96개의 feature map이 들어오면 squeeze를 하고 expand를 하여 input feature map과 convolution ouput 그리고 pooling 단계로 구성된다.
  - 해당표는 parametert와 activation size와 compression정보를 나타낸것.
  - pruning 전과 후의 parameter수를 기재함.
  - 여기서의 핵심은 이렇게 줄여서 학습을 해도 성능이 나온다는것이 point이다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/1e90d09e-3e6a-4ff5-85b8-28d96cd0b1a0)
  - compressed했을떄의 결과를 보여줌.
  - 기존 compression 기법인 SVD, Network Pruning, Deep compression같은 기법과 비교해도 squeezenet은 bit수가 줄어들떄도 모델사이즈가 충분히 작아져도 accuracy가 보장됨을 알수있다.

## 8-5. Shift operation
: spatial convolution이 necessary하지 않다는것이 point
* spatial convolution : 3x3 filter가 한채널과 spartial resolution을 주욱 긁어서 읽고 summation을 한다. 그래서 channel을 shiting을 하면서 읽어도 결국 summation한 정보라면, 그위치에 대해 유지만 하면 되지 않느냐라는 의미이다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d43d5d6f-1925-4685-8c0d-bf01c64c04cc)

  - basic idea: shift operation을 통해 spatial convolution을 replace하자. 그렇게 하면 1x1 convolution을 사용해도 feature extraction때 충분하다. 지엽적인 정보도 인코딩되면서도, 성능을 낼수있다는게 핵심적인 아이디어이다.
  - (a) spatial convolution의 예시
  : k x k kernel size를 가지는 N개의 필터를 가지고 input을 M으로 받는 convolution을 통과시키면 쌓이게된다. 노란색필터를 통과하면 노란색 channel을 만들고 주황색필터를 통과하면 다음의 주확생 channel을 만듬을 확인할수 있다. 
  - (b) depth-wise convolution
  : 각 channel별로 따로따로 kernel race를 구성하는것. 이렇게 하면 parameter수가 절약이 되고, 채널별로 따로따로 convolution이 수행되기 때문에 (각 channel이 독립적으로 operation이 된다.). parameter수가 절약되는 이유는 channel 전체를 sum하지 않으니깐!
  - (c) shift
  : feature map의 채널들을 각각 전후좌우 방향으로 shift를 시킨것.shift되면 읽히는 위치가 주변의것들이 옮기면서 읽혀진다. 그러면 1x1 convolution으로도 주변의 특정채널과 값들을 읽어낼수 있다.

- 참고논문
[Shift: A Zero FLOP, zero Parameter Alternative to Spatial Convolutions]()

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/4e369b33-c4c0-4c0c-a6fc-d2678fa8f8bd)

  - shift는 Efficient하다. 왜냐하면 operation을 추가하지 않고도 적용가능하기 떄문이다. 이렇게 하는 이유는 hardware operation들이 shift operation에 최적화되어 있기 떄문이다. vector나 matrix operation에 대해서 SIMD pipeline/ systolic array 등의 방식들이 index변경에 대해서 굉장히 최적화가 쉽게 되어있다. 왜냐하면 SIMD pipeline은 point 위치만 바꾸면 되니깐 쉽다.
  - high-dimension하게 convolution할때 deplication 및 replace도 해줘야 되서 내부적으로 hardware측면에서 봤을때 데이터를 옮기고 하는건 메모리를 상당히 소모하는 이유가 된다.
  - shift operation을 통해서 spatial operation이 필요가 없어질수 있다. 예를들어 shift를 통해서 이미 옆에있는것의 다른 channel이긴 하지만 coreleation을 1x1 convolution만을 가지고 읽어낼수 있으니깐! 이를통해 higt-order tensor의 핸들링을 줄일수있다.  shifting을 통해서 hardware  circuit을 구성하면 되니깐. 더이상 multiple한 neighborhood를 보고 할필요 없이 채널내부에서 shifting을 통해서 그 값들을 인코딩할수 있기때문이다.
  - network architecture
  : inpu data가 들어오면 하나의 모듈(shift,kernel size, Dilation rate) 을 통과하고 batchnorm+ReLU 하고 1x1 conv 통과시킨다. 그리고 다시 operation을 바꿔주고 BN+ReLU하고 1x1 conv통과시키고 avg pooling을 더해주고 concat해주면 output을 얻는다. 이러면 shift가 추가된 구성을 얻게된다.
  - 여기서 말하는 shift를 요약하자면, feature map의 하나의 channel에서 그것의 전후좌우로 이동시키는 shift를 통해서 지역적 정보도 한 feature point에서 읽어주고 그걸 인코딩해주는 기법이다.

  - 참고논문
  [Constructing Fast Network through Deconstruction of Convolution, Jeon et al, NIPS 2018]()

- Active shift

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/49967e6f-c802-40cf-957e-13237645026d)

  - 이것을 한단계 개선을 해서 shift 기법을 active shift로 개선한것
  - shift : memory acess를 줄이기 위해, spatial convolution을 deconstruction함. 즉, spatial convolution을 없애고 channelize shifting을 통해서 했었따.
  - active shift : grouped shift를 relaxation 하기 위한 기법이다. 기존의 shift는 전후좌우로만 움직였다면, active shift는 모든방향으로 움직일수 있고, 그리고 정수값이 아닌 shift도 있을수 있다. 즉 1) 추가된 wise는 depthwise shift를 하고 2) parameter가 알파,베타로 추가된다. 3) 그리고 non-integer shift를 진행한다. 그래서 수식은 image input에 대해서 c,n,n(channel, h, w)에 대해서 바뀌는 (알파,베타는 방향을 의미한다.) 수식이다.
  - active shift의 문제점 : bilinear interpolation때문에 구현자체가 realistic하지 않다. 왜냐하면, integer로 shift하면 자연스럽게 된다. pixel이 이동하면 되니깐. 그러나 active shift를 하게 되면 실수값으로 움직이기 때문에 5라는 값이 대각선으로 0.15위치로 움직인다? 그럼 어떠한 값을 넣어야할지 어렵다. 결국 interpolation(보간법)을 해야한다. 그러나 interpolation의 연산량이 크다. 그러나 idea는 validation이 되어서 accept 된 논문이다.
  - grouped shift : 위아래좌우로 움직임
  - ASL(active shift layer) : 알파,베타값을 hyper-parameter로 가져서 학습이 된다.
  - 그림은 각 stage와 shift마다 어떻게 움직였는지를 보여주는 diagram이다.
  - ASL을 요약해보면, 1. depthwise shift를 사용 2. 채널별로 다른 shift paramter(알파,베타를 학습함)를 사용 3. interpolation을 통해서 non-integer shift를 만듬. 그래서 shift value가 differentiable할수 있게! 이렇게 구성해서 shift network자체적으로 좋은것을 training할수 있게. 단점은 over-fitting될수 있다.

- active shift + Quantization
  ![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/4242c54e-3946-441a-a31a-c31718f31cdb)

    - 참고논문
      - [All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification(Chen et al, CVPR 2019)]()
    - shift를 조금더 개선한 논문.
    - shift parameter를 round approximation을 해서 interpolation이 필요없도록 (interpolation 떄문에 연산량이 늘어나니깐.) 한건.(round연산을 통해서 깎았다고 생각하면 된다)
    - straight-through estimation으로 gradient approximation을 해서 학습시간을 줄임.(sign함수를 통해서 부호만 보고 연산을 할수 있게 한다던지, 등등 )
    - L1 regularization을 통해서 aparsity를 보존함. interpolation을 안하고 approximation을 하면서 정수로 만듬. 그것을 학습하고 싶다면 L2가 아닌 L1 regularization을 많이 쓴다. 
    - initialization feature point들이 있을때 sparse learning을 하면은, output space가 구성이 되는 diagram이 있다.  

- Results on IMageNet
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/4699f6f8-ad5d-4149-a139-c19bdfff152f)
  - shiftNet의 Top-1 을 보고 Active Shift는 interpolation때문에 MAdds(Million Multiply Add Operation, AI및 딥러닝 분야에서 연산 비용을 나타내는 일반적인 지표) 가늘어났지만 accuracy(TOP-1)은 증가함. Quantization을 포함한 FE-Net은 72.9%의 성능으로 냄.
  - 모델사이즈가 큰경우 성능이 증가함을 아래 블럭에서 확인가능.
  - 마지막 FE-Net은 연산량도 많이줄가 paramter수도 큰증가없이 accuracy가 증가함을 확인가능하다.



## 8-6. Efficient CNN - MobileNet-v1
: CNN network모델을 경량화 시키기 위해서 고안된 구조.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/6359bdc0-878e-4aa2-a2d1-960fa273e27e)

- 참고논문
  - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, Howrad,2017]()
  - idea : Depthwise-separable convolution
    - (a) standard convolution 
    : convoltuion filter가  N개의 채널에 대해서 커버하는 convolution filter가 slide windows로 돌아가서 하나의 값을 만들게 된다.
    - (b) depthwise convolution
    : 각각 서로다른 convolution이 있는것이다.
    - (c) pointwise convolution
    : 1x1 convolution을 pointwiser convolution이라고 말한다.
  - depthwise-separable convolution = deptiwise convolution + pointwise convolution을 조합한것을 말한다.
  - 기존의 fully convolution은 receptive의 모든 channel을 cover한다. motivation은 어떤 feature map이 redundant & correlated이 될수도 있다는것이다. 그래서 우리는 모든 channel이 필요하지 않다라는 아이디어해서 착안. 그래서 channel-wise feature extractor와 spatial feature extractor를 decoupling하는것이다. 그래서 channel에 대한 convolution과 지역적인 convolution이 분리되는 구조로 되어있는게 deptiwise-separabel convolution이라고 한다.

  - computation cost 
  : 채널input, 채널output,hw는 output resolution size에다가 kernel size을 의미한다.
    - depthwise-separable convolution은 output convolution되는 부분이 decoupling되므로 이부분이 뒤로 빠질수 있고 computation이 약해질수 있다.
    - computation gain을 계산해보면, computation의 gain을 얻을걸 계산할수 있다.

- mobilenet structure
: 3x3 depthwise conv를 해서 채널별로 따로따로 convolution을 해주고, 마지막에 pointwise convolution을 통해서 inter channel relationship을 인코딩을 다시 해준다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/fc3c4b1c-f1d8-488f-ae66-28a95be3cd5d)

  - 각 layer type에 대한 resource를 고려해보면, multi-adds와 paramters수에 대해 나온 결과 표.
  - mobilenet의 body구조를 나타냄.
  - 이것도 최종적으로 clasiification 문제를 풀기때문에 최종적으로 avgpool하고 fc layer를 통과시켜서 1000개의 classifier의 softmax로 구성해주게 된다.

- results

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/53893c8f-05df-48e4-8e2f-c8818af0b430)

  - 기존의 popular models과 비교했을때 mobilenet이 비슷하거나 높은 accuracy를 내면서도 곱셈과덧셈의 기본적인 연산이 더 적게들고 parameter 수도 훨씬줄어듬을 알수있다.
  - 그리고 기존의 비슷한 parameter수를 가진 모델(squeezenet과 alexnet과 비교)과 비교했을때의 결과도 accuracy가 높으면서 더 적은 parameter를 가짐을 알수있다.
  - 그래프는 연산량이 늘어날때 이미지의 accuracy가 얼마나 증가하는지에 대한 그래프이다.
  - width vs resolution 
    - width을 얼마나 주었을때 성능이 얼마나오는지 표로 보여줌.
    - 점점 width를 증가시켜주었을때 성능이 증가함을 알수있다.
    - image resolution을 224x224, 192x192, 160x160,128x128 로 하면서 실험했을때 ,즉 image resolution이 커질수록 accuracy가 증가하면서 곱셈,덧셈연삼이더 많이 듬을 알수있다.
    - 오른쪽표는 parameter와 resolution에 따른 accuracy결과값이다.모델 parameter수가 많아질수록 accuracy가 증가하고, image resolution이 커지면서도 classification 결과가 증가함을 알수있다.


- shufflenet
: 1x1 convolution이 여전히 expensive하다고 생각한것도 있음. 즉, 1x1 conv는 아직도 덧셈,곱셈연산이 많이들어가고 parameter수도 많이 차지하고있다. 그래서 이러한 overhead를 더 줄이는 방법을 고안.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/2134eeb6-6ed3-421e-a26a-f420c58db7ba)

- 참고논문
  - [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices(Zhang et al, CVPR2018)]()



- shufflenet은 group convolution을 사용한다.
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/67d0c51f-337b-4a33-9680-bd73b9e0bb4f)

  - group convolution?
  : G라는 separable convolution을 사용하는것. depthwise convolution을 생각해보면, 채널별로 서로다른 convolution을 적용함. 그러나 Group단위로 서로다른 convolution 을 취해주는것.
    - network는 alexnet같은 구조를 사용.
    - depthwise-separabel convolution의 단점은? 채널간의 정보교환이 없다. 왜냐하면 채널별로 서로다른 convolution을 적용하기 떄문에 서로 덧셈,곱셈같은 연산이 없다. 그래서 shufflenet의 아이디어는 다음과 같다.

- shufflenet idea
  ![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d51548a1-6e12-4c42-935b-153ca3f2df47)

    - (a) : group convolution을 각각 서로다른 channel에 대해서 취해준다.
    - (b) : 채널을 random하게 섞이게 함. 이떈 연산량이 컸음.
    - (c) : 채널을 일정하게, 첫번째 그룹에 대한건 첫번쨰에 두고, 두번째 채널에 대한건 두번째 그룹에 두고..이런식으로 채널을 shuffle해준다. 이렇게 되면 크게 연산량을 늘리지 않으면서도 채널간의 정보교환이 가능하게 된다.
    - channel_shuffle의 코드.(tensorflow로 되어있음)
    : input으로 feature의 size를 받아서, 채널의 특정위치들을 섞어준다.(transposed)

- shufflenet design
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/08456b74-a996-46aa-b777-8747c2a58f10)

  - 1x1 convolution이 group conv로 대체가 된다.즉, group conv와 shuffle이 포함되어 있는 구조가 된다.
  - ReLU는 depthwise conv다음에 적용되는게 아니라 다른곳에 적용된다.
  - element-wise addition을 채널을 키워서 concat하는 구조로 변경한다.
  - 최종적으로 resnet과 비교했을때 bottlenet구조라고 한다면, shufflenet같은경우는 1x1 group conv를 적용하고 shuffle하고 3x3 depth conv, 1x1 group conv 후에 더해져서 연산량이 줄어드는 결과가 나온다.
  - 모델서치한 구조를 비교해보자.
    - (a) : element-wise로 더한 구조
    - (c) : concat한 구조

  - classification한 결과를 비교시 mobilenet보다 shufflenet이 일관성있게 성능이 향상됨을 알수있고, classification error가 점점 줄어든다. 비슷한 정도의 연산량일때 더 줄어듬을 알수있다.

## 8-7. Efficient CNN - MobileNet-v2

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/55208ea9-574f-4322-a887-0dc8c58ea20e)

  - 참고논문
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks, Sandler et al, CVPR 2018]()
  - 기존의 mobilenet은 Depthwise-separable convolution으로 연산량을 크게 절약하면서도 classification 성능유지
  - 그러나 Depthwise-separable convolution 와 residual block을 함께했을때의 문제점은 intermediate activation이 너무 작아진다는것이다. 그래서 intermediate activation의 채널size를 유지할수는 없을까라는 motivation이 생김.
  - feature extractor의 spatial capacity가 충분치 못하다.
  - 그래서 mobilenet-v2에서는 inverted residual structure를 사용해서 구현하는것.
  - 왼쪽은 뚱뚱한 feature가 들어가서 안에를 좁힌다음에 좁아진 feature에 대해서 feed forward를 하고 다시 늘려주는 형태
  - 오른쪽인 inverted residual구조는 feature를 다시 크게 키운다음에 거기에서 feed forwarding을 하고 그다음 다시 줄여주는 형태


![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/7db3aa9b-8c9a-4bc7-850f-c1ce78b8e2dc)
  - information lost없이 dimension reduce를 하면 ReLU와 같은 non-linear activation function을 썼을때 어떤 정보를 많이 잃어버린다.그래서 ReLU를 사용하지 않고 update한 activation function을 사용하기로 함.(ReLU6라는걸 사용)


![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/156fa7a7-7a25-4a55-8913-15268bcc0f2d)
  - pytorch에서 nn.ReLU6 라고해서 initialize가능하다.
  - mobilenet-v2에서 제안한 invertResidual 블록도 구성같이한다.
  - 나머진 다 똑같은구조인데  중간에 있는 feature map의 channel size가 더 커진다는 차이점이 있다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b2738134-2c9f-42cf-ba2d-4c9e3447b8ac)
  - mobilenet-v2를 사용시 parameter수를 비슷하게 했을때 Top1 accuracy를 나타냄.
  - linear bottlenect 보다 ReLU6를 쓴게 더 좋은 결과를 냈음.
  - mobilenetV1보다 전반적으로 mobilenetV2가 좋은 성능을 나타냄. 
  - 컴퓨터의 기존연산수(Multiply-Adds)에 따른 성능이 얼마나 증가하는지를 resolution size에 따라서 표현함
  - 거의 hardware적인 분석을 진행한것임. 



## 8-8. Efficient CNN - MobileNet-v3

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/cd29284d-2c9d-46cf-9a07-cb7bb3b677d6)
  - 참고논문
    - [Searching for MobileNetV3, Howard et al, ICCV 2019]()

  - 3 key component가 있다
    1. h-swish라는 non-linear activation을 사용
    2. squeeze-excitation module을 사용
    3. Automated architecture tuning을 사용해서 성능을 최대로 끌어냄.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/abe45cfa-c868-4715-b853-73e350afc500)
  - H-Swish Function을 ReLU와 비교했을때 ReLU는 0이하의 값을 죽여버린다.swish라는 function은 자신의 identity function에서 sigmoid값을 곱한 형태이다.즉 swish를 approximately하는게 h-swish라고 생각하면 된다.그러면 0인부분에서 gradient가 확 사라지지 않고 어느정도 유지가 되는 형태로 구성이 된다.
  - activation을 계산했을때 전체적인 분포가 histogram처럼 나온다.
  - 이러한 observation을 통해서 swish의 장점은, 네트워크가 깊어질수록 장점이 많아진다는 것이다. 그래서 뒷부분에서는 h-swish function을 모델의 반절쯤에서 사용을 한다.


- Recap : Squeeze-Excitation Module
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/a51b08fa-de78-4ca8-bc98-99c11cd7bb33)

  - squeeze-excitation module을 사용한것이다
  * squeeze-excitation module : input tensor를 넣어서 그것을 global avg pooling을 한다음에 채널 attention값을 계산해서 곱해주는것이었다.
  - 이러한 채널사이즈 sacaling을 통해서 업데이트해주게 된다.
  - mobilenet V3 block을 보면 뚱뚱해졌다가 얇아지는 구조에서 중간에 squeeze-excitation module이 들어가고 swish function이 non-linear activation function으로 사용되게 된다.


- AUtomated Architecture Tuning
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f78d8f55-c831-4b20-8fdf-f01f26f9975c)

  - 블록에 어떠한 optimal channel이 될지에 대해서 의문을 가짐! 
  - 그래서 2가지 approach를 택함.
  1. block-wise search를 위해 platform-aware Nas를 진행.(platform 구조를 알고있는 상태에서 block-wise search를 진행하는것이다.)
    - best global structure를 찾기위해 lock-level optimization을 한다.
    - target latency에 대해서 목표로 하는 연산량에 대해서 가장 최고의 모델을 찾는다.
  2. NetAdapt라는 알고리즘을 사용해서 layer-wise search를 진행한다.
    - layer-wise fine-tuning진행



![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/68b92d60-a91b-4aeb-a7de-7efca16b9a54)
  - 하드웨어 연산량, model parameter수, 다양한구조와 비교했을때에도 mobilenetV2보다 성능이 좋아짐을 확인할수 있다.
  - 오른쪽은 mobilenetV3 architecture 구조이다.
  - exp size나 output channel수, squeeze-excitation module(SE), NL(non linear activation)을 H-swish쓰는지 ReLU를 쓰는지 , s는 feature resolution을 얼마나 줄이는지에 대한 값이다.  
  - mnasNet에서 SE모듈추가하면 성능이 증가함을 알수있다.여기에 h-swish추가하면 latency는 증가하지만 성능이 증가함을 알수있다.
  - model size가 커질때마다 mobilenetV3가 점점더 높은 성능을 냄을 알수있다.
 
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/703c35be-8dfa-4da5-9fa4-30fef4e1fb03)
  - Additional Material
  - shuffleNet 을 업데이트한, mobilenet구조를 차용해서 만든논문
  - condenseNet : denseNet에서 group convolution을 사용해서 update한 논문
  - shift구조를 사용해서 채널간의 corealation도 임베딩하면서 얼마나 더 효과적인 neural network를 만드는지에 대한것.
  - 또한, neural network를 더 효과적으로 하기 위해서 hardware-level에서 optimization하는것도 있다.



# 9. Vision Transformer 1: Self-attention
: vision transformer는 convolution neural network를 대체하는 구조로서 많이 사용됨. CNN은 convolution이라는 기본 연산을 통해서 non-linear activation을 통해서 더 flexiable한 함수를 만들고 batch normalization을 통해서 더 좋은 모델을 학습하는 구조였다. 하지만 vision transformer는 attention이라는 블록을 사용해서 특히 multi-head attention을 사용해서 여러개의 채널들을 쌓고 convolution보다 더 넓은 영역을 볼수있는 즉, convolution의 지역성인 inducted bias를 제거하고 어떤 모델을 구성하는 구조이다.

특히 attention은 NLP(natural language procesing, 자연어처리)쪽에서 발전해오던거였지만, 이것이 vision에 쓰이면서 image classification에서 좋은성능을 내고있다.

## 9-1. attention
: 어떤 위치를 주목한다는 의미.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/0e7a04c2-d701-4c49-8bed-6311d833de9a)


- 참고논문
  - [Neural Machine Translation by jointly Learning to Align and Translate(Bahdanau et al, ICLR 2015)](https://arxiv.org/pdf/1409.0473.pdf)
    - 확률모델같은거로 모델 디자인을 한다.
    - RNN의 인코더,디코더 구조를 가진다.feature는 현재 state의 단어와 이전 state의 단어의 결과를 받아서 이전 state의 t를 만드는것.
    - c는 hidden state로부터 만들어진 것이다.
    - learning to align and translate를 보면 main contribution의 핵심은 디코더이다. 알파는 attention weight이다. 
    - 얻어진 weight들에 대해서 softmax가 취해진, 어디에 강조를 할지 weight가 결정이 된다.
    - diagonal matrix로 나오면 잘되는것인데 실제로 하니 잘나옴을 알수있다.
    - 잘맞추었는지에 대한 bluescore도 있다.
    - related work를 보면 align을 통해서 learning하는 방법이나, 등등이 적혀있다.  
  - [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention(Xu et al, NIPS 2015)](https://arxiv.org/pdf/1502.03044.pdf)
    - 이미지 캡셔닝위주로 쓴 논문이다.
    - 여기선 멀티모달, 즉, 이미지와 단어 간의 cross attention weight을 계산하는 task이다. 어떤 워드가 캡션으로 붙어야할지 찾아주는 network이다.
    - 여기선 RNN이 아닌, LSTM구조를 사용한다.
    - 이미지에 대해선 a라는 feature를, 단어에 대해선 b라는 word 임베딩 feature를 구한다. 이 디코더를 LSTM에 넣어서 만들어준다.
    - 이미지 dataset으로 flaskr8k, flickr30k,coco를 사용해서 evaluation진행함.blue score는 이미지 캡셔닝의 잘된정도를 측정하는 evaluation metric이다.  meteor score도 더 잘됨을 알수있다.
    - appendix가면 soft,hard attention이 얼마나 잘 잡히는지에 대한 정성적 결과들이 있다.기린이 있으면 각 워드에 대해서 어디가 어떻게 attend되는지 예시결과가 있다. 


  - RNN구조에서 Alignment network에 각 위치(단어)들에 대한 attention weight들을 계산해준다.그래서 time T번째까지 attention을 계산해준다.여기에 각각 어떤 단어에 집중할지를 weight로 줘서 다음단계로 넘어가주는 재귀적인 구조를 가지는 것이다. attention weight는 자기자신의 sub-network를 통해서 계산된다고 생각하면 쉽다. 자기자신의 online상황에서 어떤 network에 넣고 그 weight를 계산해주는 .그래서 최종적으로 벡터상에서 softmax를 통해서 weight를 계산해준다. 원래 attention은 **번역** 분야에서 사용되었다. 그다음에는 image captioning이라는 분야에서 사용되기 시작함.
  - figure1을 보면 언어와 사진의 alignment를 통해서 개발이된다. input image가 있고 VGG를 써서 convolutional feature extraction을 한다. 그래서 14x14 feature map을 만들고 RNN구조인 LSTM을 사용해서 어디에 attention을 할지 활성화해주고 최종적으로 나온 word by word generation해서 word token들이 클래스처럼 들어가서 probablity를 multi label classification문제처럼 풀어내는것이다. 즉, corealation(코릴레이션) matrix를 만드는것이다. 이미지 임베딩과 워드임베딩에 대해서 코릴레이션 matrix를 만들어서 word to word generation을 하는것이다.
  - figure2는 워드 하나에 대해서 어디를 attention하고있는지 코릴레이션 score를 정한것.워드임베딩은 원핫벡터라 이미지벡터의 코릴레이션 matrix를 만들어서 score가 높은것을visualization을 한것이다.  위는 soft attention, 아래는 hard attention이다.
  - A는 날개부분을 soft attention한것이고 bird는 잘잡고있고 flying은 새가 날아가는 부분을 잡고있고 over는 물 위같은부분을 잡고있는것.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/3f82c062-b5c3-4e5e-80b0-1f9108200ba1)

  - hard attention : 0 or 1로 attention을 하는것.
  - soft attention : weight를 주는것이라고 생각하면 된다.
  - fresbee와 woman쪽에 attention이 된걸 확인할수 있다.
  - 위는 일반적인 attention이고, 아래는 soft attention 수식이다
  - 0,1사이에 soft-weights으로 attend를 한다. 알파라는 값이 0~1사이의 continues한 value이다. 이것이 feature에 먹여지는것이다. 
  - stochastic hard attention은 0 or 1로 attention.
  - 모든 위치 다하면 hardware resource 많이 잡아먹으니깐 샘플링한다.

- self-attention
: 중첩된구조이다.
이미지넣자!!!!!!!!!!!!
  - 참고논문
    - [Image courtesy: Self-Attention Generative Adversarial Networks(Zhang et al, ICML 2019)]()
  - 자기 자신을 사용해서 어디가 attend되어야 할지를 계산하는.
  - 아까는 워드와 이미지를 사용했다는 여기선 이미지 하나, 자기자신을 복제해서 attend를 주는 형태이다.
  - convolution feature maps이 있으면 서로다른 1x1 conv를 통과해서 f(x), g(x), h(x)를 각각 만든다. 각각 key, query, value를 의미한다.vlaue는 말그대로 값이고, key,query는 둘이 계산이 되서 attention map을 계산하게 된다. key는 transpose되고 query는 그대로 통과해서 코릴레이션 matrix를 만들어준다. 그것을 softmax를 통해서 attention map을 만들어주고 이 attention map이 value의 1x1 conv를 통과한 image feature에 대해서 matrix multiplication으로 계산이 된다. 그래서 input과 이미지 size가 같은 1x1 conv를 다시 통과시켜서 최종 self-attention feature map을 만든다.
  - i, study, at, school이라는 벡터들이 있다면 이게 각각 x들을 만들어서 1x1 conv를 통해서 weight값들을 만든다.그래서 key,query,value가 만들어지고 이건 각각 어디에 강조되어있는지 feature map을 가르킨다.결국 학습되는 matrix는  1x1conv이 있다.  

- multi-head self-attention
이미지!!!!
  - 참고논문
    - [Slide courtesy : Visual Guide to Transformer Neural Networks-(Episode2) Multi-Head & self-attention](https://www.youtube.com/watch?v=mMa2PmYJlCo)
  - Visual is transfomer에서는 multi-head self-attention을 사용한다.
  - multi-head self-attention?
  : attention filter가 이미지를 통과하면서 어디를 강조할지 attention이 되어있는것이다. 첫번째 filter는 사람을 attend하고 있고, 2번째 필터는 하늘을 attend하고 있고, 3번쨰는 뒤에 산을 attend하고 있다. 결국 이 filter들의 weight이 있다. 더 중요한 부분에 대해서 더 높게 알아서 학습이 된다는 것이다. 결국 convolution filter도 중요한 부분에 대해서 attend가 되는 형태로 구성이 되는데 attention의 장점은 학습이 되면서 떨어져있는 지역도 비슷한 관계성이 있을떄 학습이 된다는것이다. 그래서 inducted bias가 더 적다고 말할수 있고, 핵심적인 tranformer의 장점이라고 말할수 있다.

- cross attnetion
이미지!!!
  - 참고논문
    - [Cross Attention Network for Few-show Classification(Hou et al, NIPS 2019)]()
  - self-attention은 key,query, value가 한개에서 나와서 MLP를 통과해서 1x1 conv를 통과해서 만들어진다.
  - cross attention은 하나는 key,query에 input A와 input B가 들어가고 value에 attend하고 싶은 값이 들어가서 2개의 fair 값으로 attend를 계산하는것이다.
  - correlation layer를 통과해서 이미지 p,q에 대해서 어떤 값을 만들고 그것을 fusion해서 각각 사용해서 attention weight를 만들어서 각각을 먹여주게 된다. 그러면 중요한 영역에 대해서 attend가 된다.
  - 적은수의 dataset으로 학습을 하려고 할때 cross attention 기법을 사용하면, 조금더 이미지간의 correlation까지도 활용해서, 더 많은 정보를 활용하여 학습을 할수 있다.

- Stand-alone self-attention
이미지!!!
  - 참고논문
    - [Stand-alone self-attention in vision models, Ramachandran et al, NIPS 2019](https://arxiv.org/pdf/1906.05909.pdf)
  - local self-attention이다.
  - attention이라는것을 이미지 전체와 word 전체에 대해서 계산을 하였었는데 이미지는 지역성이 존재한다.예를들면 이미지를 grid로 나눌수도 있고, 소나 사람이 어디있음을 알고 지역적인 부분만 patch를 때서 계산할수도 있다. 그래서 이러한 지역적인 부분만 때서 local self-attention을 진행한다는 의미이다.
  - 이럴때 장점은 모든이미지에 대해서 계산할필요가 없으며, 모든 이미지에 대해서 하나의 나머지 이미지위치에 대해서 모두 계산하지 않아도 되므로 computation cost도 줄어들고, 필요한 부분만 계산을 해서 조금더 좋은 결과를 내고 classification이 아닌  지역을봐야하는 task에서도 자연스럽게 동작하는 특징이 있다.
  - 수식을 보면, q가 쿼리 k 가 key, v는 value이다. query값을 neightborhood patch k의 i,j에서 샘플링해서 a,b를 주는 그래서 인덱스를 줘서 attention을 주는 local self attention을 만들어준다.
  - 논문을 보면 image net classification에서 parameter와 clop수도 많이 줄이면서 좋은성능 냈다. COCO object detection에서 evaluation을 했다.
  - 상대적인 좌표값을 이용해서 주변값들의 위치정보를 넣어준다.


## 9-3. transformer
이미지!!!
  - 참고논문
    - [Vaswani et al, Attention is All You Need, NeurlPS 2017]()
  - transformer는 위 논문에서 먼저 제안이 되었다.
  - self-attention mechanism에 대해서 잠깐 복습해보자.
    - global dependency를 그려보면 오른쪽 그림처럼 된다.번역하는 task가 있다고 했을때 output의 it이라는 단어는 어디서 왔느냐 하면은, 여러 attention weight를 통해 softmax를 이용하여 어디를 더 강조하느냐를 통해 벡터로부터 값이 오게된다.
    - single sequence의 다른 position에 대해서 값이 얼마나 중요한지에 대해서 자기 스스로 weight를 계산해서 할수있는게 self-attention이다.
  - transformer란? self-attention 구조를 많이 중첩시켜서 아키텍처를 만드는 시퀀스 모델링이라 할수 있다.그래서 sequence-aligned recurrent neural network가 필요가 없다. 왜냐하면 전체에 대해서 weight를 계산해서 중요한것에 하나에 몰아주니깐.그래서 NLP sequence model에서 개발이 됬다. supervision에서 가운데 빵구를 뚫어서 없는걸 예측하는 language model에도 pretrained model에도 많이 사용한다.

  이미지!!
  - transformer의 전체적인 아키텍처
: 인코더를 먼저 통과시켜서 이것을 이제 어떤 워드 임베딩으로 인코더 벡터들을 쭉 만든다. 그 다음에 이 디코에다가 각각 쭉 넣어서 이제 그 디코더로 어떤 아웃풋이 나올지 타겟이 되는 도메인에 대해서 아웃풋을 만들어 내는것. 이때 인코더/디코더의 구조구성은 transformer구조이다.
input이 들어가면 ouput이 어떤 output 임베딩한다. 그래서 input이 들어가면 먼저 position으로 인코딩하고 multi-head attention layer를 통과하고 그 attention을 더해주고 normalize하고 feedforward를 하고 이쪽으로 와서 output임베딩에 대해서 masked multi-head attention을 한다. masked multi-head attention은 그냥 multi-head attention에 그냥 masking을 했다는 것이다. 그래서 input value을 key, qeury로 사용해서 이제 attention을 계산해주고 이건 value로 사용해서 이제 value에 Add & Normalize를 해준다. 그리고 피드포워드를 해주고 linear softmax를 해서 output probablity. 그 확률분포를 계산해주는것이다.이것이 전체적인 transformer구조이다.

이미지!!
- self-attention mechanism
이떄 사용되는게 multi-head attention인데 이것이 핵심이다. self-attention mechanism을 보면 key,query,value가 들어가서 top project를 계산한다. 그러니깐 key, query가 이렇게 matrix 멀티플리케이션 스케일 그리고 마스킹 여기 마스크도 셀프헤드 멀티헤드 어텐션에서 마스킹을 해주고 소프트맥스를 통과한다. 그다음 벨류에 대해서 매트릭스 멀티플리케이션으로 원래 어텐션은 그 엘리먼트 오지으 멀티플리케이션이었다. 그게 weight를 해줬는데 MatMul도 결국 같다. 그게 size가 달라서 어떤 weight가 들어가는게 결국에는 그 위치에 대해서 matrix의 어떤 값들이 곱해지는 거다. 벡터가 벡터끼리 . 그래서 이렇게 attention이 스케일드 닷 프로덕트 어텐션 그냥 어텐션이라고 안하고 스케일드 닷 프로덕트 어텐션이라고 한게 이렇게 닷프로덕트로 결국 어텐션을 계산한다는 의미인것. 그래서 value, key, query에 대해서 이렇게 attention을 계산한다는것이다. attention을 계싼해서 이제 concat하고 그다음 linear layer를 통과해서 최종적인 output을 만든다. 그래서 이 하나의 블록처럼 멀티헤드 어텐션이 쓰이고 넘어갈때 이제 그 원래 인풋과 residual connection이다. 이게 이제 normalize해주고 이런 중첩된 구조가 n개가 더 있따. 그래서 n곱하기가 써있다.
  
이미지!!
- 조금더 자세하게 예시를 보자. 각 input token의 linear layer을 통과하기에 앞서 query, key,value를 우선 계산한다. 즉, k,p,v가 어떤 input embedding에 들어가면은 k,k,v가 세개로 줄어든다. 이게 mlp를 4를 3으로 바꿔주는것이다. 이걸 각각 서로 다른 걸 통과해서 이렇게 vector를 만든다. self-attention이니깐. 그래서 score를 먼저 계산한다. 그래서 q,k를 계산해서 score value가 나오고 드아므 어떤 normalize한 값을 해주고 softmax를 해서 그 softmax의 value랑 이제 곱해주고 원래 input과 sum해준다. 결국 z vector를 만들어주는 것이다. 그래서 key와 query는 attention weight를 draw하는데 만들어내고 이제 그것은 v single vector를 aggregate를 하는데 사용이 된다. 그래서 예시로서 input x가들어왔을때 weight q,k,v가 각각 곱해져서 임베딩을 만들고 그값을 nomalize 나눠주고 그다음 value랑 곱해져서 z output을 만드는게 self-attention block에 가장 잘 요약된 예시이다. 여기의 point는 weight, w부분만 학습이 된다는 점이다. 나머지 부분은 미리 정해져 있는 matix multiplication 은 정해진 연산이니깐. 이런거로 이루어져있다는것.이렇게 tranformer가 잘될수 있었던 이유는 gpu가 행렬곱에 굉장히 최적화된 연산들 이기때문에, 빠르게 수행될수 있었다.

이미지!!
- positional encoding
  - transformer에서 또 중요한 부분이다. 이놈은 말그대로 위치 인코딩이다. attention을 잘 생각해보면 어떤 특정한 위치의 그냥 강조한 영역들에 대해서 강조를 해주는데  그게 어떤 위치값인지 나타내주진 못한다. 예를들면 convolution은 1x1, 2x2, 3x3 이렇게 어떤 위치 정보들이 있어서 그게 sum 되니깐 위치정보가 있는데 이거는 이제 transformer는 그런게 없이, 값들을 전해주기 위해서 positional encoding이 있는것이다. 그래서 self-attention mechnism은 input sequence의 order를 모른다는 것이다. 즉, a dog is on the tree라고 한다면,  a가 첫번쨰, dog가 두번쨰라는 위치정보를 알려줘야 되는데 attention weight만주면 그 값을 모른다는것이다. 그래서 transformer는 positional encoding을 input되는 token embeding에 이제 추가를 해준다. 즉, a에는 1을주고 dog에는 숫자 2를주고 이런식으로 준다는뜻. 근데 이런식으론 안주고 싸인,코사인 인코딩을 한다. 그래서 짝수번째에는 sin x 만분의 position k값에 대해서 d로 나눠준 값을 넣고 cos에 대해서는 만분의 2값을 넣는다. 그래서 뒤에  concat을 해준다. feature vector가 있으면 positional encoding값을 뒤에 그냥 붙여주는 어떤 index 임베딩이라 한다.  random 임베딩을 할수 있지만 랜덤한 아무값이나 넣어도 포지션끼리 서로 다른것을 알려줄수 있기때문에 연속된 어떤 sin곡선이랑 cos곡선이 연속된값이니깐 그 연속된 값들을 넣어주기 위해 sin,cos을 사용하는것이다.  그래서 p가 어떤 포지션이고 i가 디멘젼 인덱스일때 이제 d는 total dimention of size와 position 위치 p, 그리고 그 어떤 포지션 그리고 d, dimention 임베딩 사이즈 인덱스 이렇게 해서, i가 그 디멘젼 인덱스이고, p가 어떤 포지션 이고, d가 인베딩 사지으이다. 이렇게 연속된 함수로 나타낸다고 할수 있다! 

  이미지!!
  - 왜 positional encoding에 sinusoidal하느냐?
    - position encoding을 1,2,3,4고 한다면 a1은 각각 word에 해당하는 위치이다. 이미지에서는 어떤 patch에 해당하는 위치 또는 영역에 해당하는 어떤 인베딩이다. 근데 이렇게 하면 좋지만 문제가 7이면 가중치가 여기가 높아져 버린다.  그래서 gradient explode같은걸 발생시킬수 있다. 그래서 개수로 normalize해서 사이즈가 8이니깐 8로 나눠주면저렇게 되는데 이것도 먼가 cheating한 우려가있따. 뒤로갈수록 점점 값이 커지니깐 뒷부분에 대해서만 더 중요하게 학습이 될수있다. 
    - 그래서 또 다른 방식으로 just count using binary instead of decimal즉, 어떤 십진법이 아니라 이진수를 활용해서  d는 인코딩 사이즈이다. 이값을 이렇게 붙여서 활용하면 된다.a랑 000을 붙인 벡터로 나중에 input을 넣는것이다. 이 positional encoding이라는게 근데 그러면, binary vector들이 어떤 scalar값을 하기 때문에 이거는 어떤값이 커지는 우려도 크게 없고 물론 0,1로 받겠지만...좀더 나은방법일수 있다.
    - 근데 이거를 좀더 확장해서 연속되게 보여지게 continues binary vector를 사용하는것이다.  즉 sin, cos사용하는것. 이건 값들이 아주 연속적이니깐! 이게 음향조절 다이어리라고 했을때 이게 이진법으로 되는것이다. 2배키운 diary고 4배키운 diary라고 했을때 즉, 2배짜리가 2번돌아가는게 4배짜리 1번돌아가는것과 같다.이게 각각의 sequence length고 각각이 word이다. 그 위치에 대해서 이런식으로 positional encoding을 준다는것이다. column이 어떤 positional embedding vector들이고 이제 row가 그 세팅들의 각 위치들을 나타내는것이다. 그럼 이러한 dial은 matrix elemnt로 나타낼수 있다. 이건 연속된 숫자값으로 나타내기 때문에 dial들이 positional encoding matrix가 되고 여기에 어떤 이 sequence length가 말그대로 어떤 word size 그러니깐 sentence size라고 말할수 있다. 그래서 dimension이 커질때 이제 dial도 sensetive해지니깐 개수도 늘려야겠따. 그래서 이 예시는 연속적이면서 미세하게 움직이는 다이얼로 비유되는 continuous binary vector를 사용한 positional encoding 예시이다. 이거를 이제 가져와서 이제 sin 함수에 어떤 matrix로 나타낸다. 이 Mij를 어떤 평면상에 표현해보면 sin곡선이 빙글빙글 돌아가는 값을 이렇게 알수있다. 이렇게 weight를 줄수있는데 이게 이렇게 되면 중복되는 값이 생길수 있따. sin곡선을 이렇게 순환사이클이므로 주기가 있다. 그 주기성을 막기위해서 w값을 아주 주기를 길게한다. 즉, w0를 만분의 1로 주면 숫자가 만까지 갈때까지도 쭉 올라갔다가 제자리로 돌아오지 않을것이다. positional encoding이란건 서로 다른 위치에는 다른 값이 들어가야 되는데 어디가 똑같이 들어가게 되면 그게 혼동될 우려가 있기 때문에 그런걸 막아주기 위해서 이제 w0 개수에 아주 큰 값을 넣어줘서 sine positional encoding을 주는것이라고 생각하면된다. 그래서 이 positional encoding을 visualize해보면 어떤 dimension, 이건 그 encoding dimention이다. 이걸 position에 따라서 계속 서로 다른 값이 들어가게 되는 이런 결과들을 쭉 볼수있따. 정확하게 안보일수 있지만 서로 다른 값이 계속 들어가는 것이다. 그래서 input embedding과 output embedding이 이건 머신 트랜슬레이션 task인데 거기단어들이 embedding이 들어갈때 포지션을 뒤에 다가 넣어준다는게 이제 어떤 트랜스포머에서 짚고 넘어가야할 핵심중 하나이다.



## 9-4. Vit(Vision Transformer)

이미지!
- NLP에서 쓰이던 transformer는 vision에서 vision Transformer로 transfer되어서 해당논문에서 출판됨.
- NLP에서 쓰이던 transformer와 다른점은? 워드임베딩을 이제 16x16 patch를 word로 replace했다는것이 다른점이다.

이미지!
- 구조
  - 원래 tranformer구조는 input embedding이 word가 들어갔다. 그러나 이젠 patch들을 linear production 시켜서 즉, linear projection MLP이다. 언바이언 컨볼루션이라고 할수도 있다. 이런걸 통과시켜서 어떤 패치 임베딩을 만들고 그걸 트랜스포머 인코딩에 통과시키는게 핵심적인 차이이다.
  - transformer encoder는 multi-head self-attention layer들로 구성이 된다.  그래서 이전의 transformer와 동일한 layer이다.
  - 이미지패치에 linear embedding을 input으로 받아서 학습가능한 1D position embedding을 token으로 넣는다. 즉, positional embedding을 원래는 sin값으로 넣었었다. 근데 여기서는 learnable한 어떤 weight로 넣어서 positional embedding을 준다는점이다. 
  - 이모델의 특징이 큰 스케일 dataset에서 학습되었을때 잘된다는 점이다. 요즘은 트랜스포머를 작은 스케일의 데이터셋에서 학습을 하면 오히려 CNN보다 안좋다는 논문이 많다. 그래서 데이터셋 수가 많아질수록 거기서 배울수 있는 어떤 양이 이제 구조적으로 많다는것이 이제 트랜스포머에서 얘기가 나오는 부분이다.
  - learnable embedding은 local image information과 어떤 위치를 같이 이제 인코딩하고 있는  그래서 최종적으로 어떤 classified로서 사용될수 있는 부분이다. 그래서 패치임베딩들이 추가가 되고 최종적으로 MLP head를 통과해서 class probility를 예측하는것이다.

이미지!
- ViT구조는 base, large, huge모델 3가지로 구성되어있다. layer수 차이와 hidden size dimension, MLP size, multi-head가 몇개인지, paramete수로 model variant가 구성될수 있다.
- 오른쪽사진은 attention이 좀 잘뽑힌 예시들.
- self-attention, 즉, 전형 어텐션을 어디할지를 주지 않고 셀프트레이닝으로 셀프 슈퍼바이전 트레이닝으로 학습을 했는데 이제 이렇게 어텐션이 이렇게 visualize가 됬다는 부분이다.
- VTAB-1k performance across task : 다양한 task에서 evaluation하는 어떤 benchmark이다. VTAB-1k라는게 Evaluation평가를 해봤을때 전체적으로 성능이 좋다는것이다. 표에는 base 성능이 안나온상태라 비교가 힘들지만 ViT-Huge model, large model을 봤을때 (JFT는 학습 데이터셋이다,I21K는 iNaturalist 21K dataset) Caltech101에서 90넘는 성능을 보이고 CIFAR-100에선 84, DTD, 등등에서 높은성능을 내는게 확인이 된다. 그리고 Mean으로 평균값을 보여준다.
- 논문을 보자.
  - Google에서 publish한 논문이다. Self-Attention Based Architecture가 이제 Transformer 기준으로 많이 NLP에서 선택받아왔따. 근데 Motivation으로 시작한다. 근데 Vision에서는 이제 CNN Architecture가 dominent한데 이걸 Transformer를 써서 이제 실험을 해보자고 하는게 이 논문의 핵심이다. 
  - Related Work로는 Transformer 계열의 모델들과 Pre-Training등등이 있다.
  - 전체적인 구조는 Transformer Encoder와 동일하다. 그거랑 기존의 Transformer와 ViT가 동일하다 Trnasformer Encoder가 그래서 patch 임베딩 패치가 들어가고 노말라이즈를 하고 Multi-Add Attention 그리고 Sub-Skip Connection으로 더해지고 노멀라이즈하고 NLP를 통과해서 output이 나오는 그걸 LGA에 대해서 반복을 하는 그런 구조.'
  - pre-training을 어떤 데이터셋으로 했을때 Top-1 Accuracy가 어느정도 나오는지에 대한 ImageNet Transfer성능과 JFT Pre-Training Sample에 대한 어떠s Top-1 Accuracy 결과가 있다.
  - 그리고 모델 스케일링 study 즉, huge model,large model, base model에 대해서 scaling 즉 크게 했을때 달라지는 점. 이건 EfficientNet으로 올라오던 계열에서 항상 하는 스터디이다. 모델 사이즈가 커질떄 얼마나 더 커지는지 그래서 이 연구는 이제 아주 리소스를 많이 가지고 있는 대기업에서 하기에 좋은 연구이다.  왜냐면 GPU resource가 아주 많이 요구되니깐! 
  - initial Linear Embedding에 대해서 RGB value를 필터를 print했을때 어떤  convolution과 마찬가지로 독특한 패턴들을 잡아내고 있다. 그래서 First 28개의 principal conferences를 visualize한것이고 가운데는 그 positional encoding에 embedding을 visualize한건데 이제 이 matrix가 자기 이 첫번째 1x1 위치에서 correlation matrix이다. 그떄 1x1과 가장 correlation가 높고 그 주변 위치들에 대해서 이렇게 position이 similarity가 높은걸 볼수있다. 이것도 자기 자신 위치에서 가장 correlation이 높고 이 patch하나를 자세히 보면 이렇게 점점 옆으로 갈수록 correlation이 높아지는 그러니깐 이게 1D로 펼쳤으니까 이제 이런 어떤 지역성을 어느 정도 학습을 한다는걸 보여주는 내용이다. 그옆에는 network depth와 head의 size가 어떻게 바뀌는지에 대한 변화이다.
  - table3을 보면 ResNet + ViT로 했을때 epoch을 얼마나 돌리고 learning rate와 learning rate decay을 항상쓴다(learning rate를 크게 잡은 다음에 점점 줄여나가는 그런 decay방식). weight decay(어떤 Regularization Term의 weight decay있다.), dropout을 어떻게 썼는지에 대한 값이 있다. 그래서 string resolution은 224를 넣었다.
  - appendix를 보면 그래서 multihead self-attention내용을 간단하게 다시한번요약해준다.여기도 key,query,value가 있고 query,key가 곱해지고 이제 더한것을 softmax에서 attention을 구한다. 그래서 query, query value는 MxM짜리 attention이고, A와 V를 matrix multiplication으로 구해서 그 결과에 self-attention Z-tensor를 넣은 self-attention(SA)이 A곱하기V로 나타나질수 있다. 그걸 Multihead로 해서 multihead self-attention(MSA)을 SA1,SA2,SA3해가지고 concatenation을 한게 multihead self-attention인거다. 
  - 그래서 experimetn detail에서 이제 트레이닝 어떻게 했는지 resolution 224로 고정했다. 여기서 fair하게 학습을 하기 위해서 한거고, fine-tuning을 각 task에 맞게 어떻게 했는지 이제 ViT 모델을 free training을 한다. 그다음에 거기에서 이제 fine-tuning을 각 dataset에 대해서 어떻게 주었는지 training sample을 몇개나 썼는지 이런 디테일들에 대한 설명이 있다. 그래서 step size를 얼마나 했는지, running rate을 얼마나 했는지 이런 부분들을 reproduce할일은 아마 없을것이다. 이 모델을 갖다 쓰기만 할껀데 대체로 이제 갖다쓰기만할때 이렇게 학습하면 되드라. 나중에 실생활에서 적용할떄 이제 약간 힌트를 얻을수 있다. base running rate(Base LR)를 대략 이정도로 잡으면 이런 task가 풀리는구나. 아니면 이정도 scale data set에서 step size는 이정도 줘야 되는구나. 이런거를 힌트를 얻으라고 하는것. 
  - self-supervision으로 학습을 했다. marked patch prediction이라는 task를 수행한것이다. 이것이 머냐면, transformer가 language model이라는걸 학습을 한다.만약에 구조이미지를 보면 language면은 거기에 빵구를 뚫어가지고 그 빈칸을 예측을 하는 어떤 task를 수행한다. 마찬가지로 이것도 patch에 대해서 여기를 가리고 이 patch 결과를 예측을 하는 어떤 task를 수행을 한다. 그걸 masked patch prediction이라는 task이다. 이런 task로 어떤 pretraining을 수행한다. NLP task에서! 예를들면 이렇게 빈칸으로 주는거다. 그리고 그걸 prediction하게 하는 self-supervised running방식을 동일하게 채택을 한다. 
  - table5를 보면 각 모델들에 대한 pretraining dataset을 ImageNet, ImageNet-21K, JFT-300M 등을 썼을때 JFT-300M이라는 엄청 큰 데이터셋을 썼을때 가장 결과가 잘 나오더라. 그래서 pretraining dataset의 수도 상당히 중요했고, fine-tuning dataset size의 resolution size도 중요했다란 메시지를 준다.
  - additional analyses는 optimizer를 SGD를 쓰냐, Adam을 쓰냐 이런 additional실험도 있다. 그리고 head type과 class token을 어떻게 쓰는지에 대한 실험도 하고 transformer shape에 대한 실험. positional embedding을 왜 이렇게 줬는지 어떻게 줬는지 positional embedding을 안준경우, 그리고 one-dimensional positional encoding, two, relative 등 각각에 대해서 실험을 해본게 나온다. 그때 one-D positional encoding을 써도 충분했기 떄문에 여기서 one-D를 쓴거 같고 아마 2D로 할거면 sign-cosign 을 따로 줘가지고 할수도 있다. 그래서 Grid sampling으로 positional encoding을 줄수 있을거 같다.
  - 그 imperical한 computational cost에 대해서 이제 여기서 기재를 한다. TPU-3 accelerator를 사용. TPU는 tensor processing Unit이다. 
  - Axial attention은 model variation을 사용해서 이제 실험을 한 결과가 이제 figure13에 있는것이다.

- ViT가 인기가 있어진 이유는?
  - CNN predecessor 즉, CNN이 앞에 온다. 그게 이제 앞에 와서 인코딩을 하는데 그게 speed와 accuracy가 더 나아진다. 그 CNN predecessor보다 그러니까, 앞에서 일단 patch encoding을 하고 지금 MLP로 하지만 사실 그 CNN으로 넣을 수도 있다. conv net을 ViT 앞에다. 그렇게 할수도 있지만 어쩃든 CNN보다 이제 더 나은 speed와 더나은 accuracy를 준다. 다만 매우 큰 dataset에서 학습이 되어야 하는게 조건이다. vision transformer를 처음부터 학습시키는 거는 물론 task에 따라 다르겠짐나 classification에는 좋은생각은 아니다. pre-training large scale dataset에서 학습된걸 쓰는게 좋다. 그러므로 자원도 많이 드니깐 인터넷에 pre-training된거 많으니 그거 사용해라.
  - Image specific inductive bias가 CNN보다 적다. 예를 들어 transformer가 이제 CNN보다 inductive bias가 적은데 예를들면 locality가 있다.2D neightborhood structure를 CNN은 반드시 이렇게 읽으면서 패턴을 파악을 하는데 그거에 대해서 Translation Equivariance가 보장이 된다. 그러니깐 CNN은 사람이 이제 왼쪽에 있거나 오른쪽에 있거나 그거를 CNN을 돌리고 나면 똑같은 feature map이 같은 위치에 대해서 뽑힌다. 즉, object가 translation하더라고 똑같은 위치에 그대로 나오게 되는 equivalence가 보장이 되는데 ViT에는 그런게 없다는것. 그래서 어떤 locality에 대한 측면이 없고 그냥 어떤 패치를 넣어서 결과가 나오기 때문에 그런 constraint가 더 2D neightborhood consistraint가 없어져서 오히려 더 어떤 학습을 통해 알수 있게 보장해준다. 
  - 이제 패치에 대해서 이제 special relationship을 처음부터 학습한다는점이다. conv net은 지역적인 어떤 패턴을 파악하는 거라면, scratch부터 어떤 패치의 어떤 relation들을 학습을 이미 하는거여서 더 flexible한 이제 캡처가 가능하고 이제 lomulation interaction을 entity간에 포함을 학습을 할수가 있다. 예를들면 맨위에 있는 왼쪽 상단에 있는 패치와 오른쪽 하단에 있는 패치는 CNN에서는 receptive filed가 아주 커져야 그게 학습이 될수있는데  ViT상에서는 애초에 그거의 correlation을 구해가지고 attention map을 구하기 때문에 그 correlation interaction을 자동으로 캡처할수 있다는게 장점이다.



## 10. Vision Trnasformer 2: Image Processing Transfomer

# Swin Transformer
이미지!!!!
- Swin = Shifted Windows의 줄임말로 Shifted Windows를 사용한 Transformer라고 생각하면 된다.

- motivation 1: visual entities의 다양한 scale을 커버해야한다는것.
  - 이미지를 보면 작은물체, 큰물체가 등이 있을수 있따. 그 물체에 따라서 다양한 스케일의 input을 받아줘야 한다는점이다. 그걸 커버하는 architecture 구조가 있어야 되는데 기존의 transformer들은 그냥 16x16 patch를 잘라서 input으로 넣어줬따. 그걸 visual word로 썼는데 그렇게 하지 말자는것.
- motivation 2 : large resolution의 이미지 input에 대해서 patch words를 잘 구성해야 한다는점.  
  - 원래 기존의 이미지 classification 푸는 문제들은 다 224x224라든지 384x384 등의 비교적 요즘으로 치면 작은 이미지나 정방형의 이미지를 가정했다. 그게 아니라 더 큰 large resolution의 image input에 대해서 patch words를 잘 구성해야 한다는것이다.

- solution : hierarchical transformer(하이라이티컬 트랜스포커) 사용하고 shifted windows 기법통해 문제를 해결.

- application으로 image classification 뿐만아니라 object detection, semantic segmentation에서 좋은 성능냄. 

- 오른쪽이 swin transformer의 티저이미지인데 A가 transformer, B가 baseline인 BiT 비전 트랜스포머이다. 기존의 ViT를 예로 들면 어떤 싱글로우 resolution image에서 row resolution feature 벨트를 뽑는다. 그래서 16x16 patch로 자르면 이렇게 이미지가 64x64 이미지라고했을때 4x4의 어떤 visual words로 분해가 된다. 그다음 positional coding이 달리고 이게 쭉 들어가서 classification결과가 나온다. 그래서 이것의 특징은 global한 attention을 뽑는다. 이미지 전체에서 self-attention을 뽑을때 전체를 고려해서 뽑아가지고 어떤 이미지 사이즈가 커질떄마다 크 쿼드라틱하게 제곱으로 컴플렉시티가 늘어난다. image size가 커지면 반면에 Swin Transformer는 hierarchical한 feature map을 이제 쓰는데 이제 이미지 패치들을 merging한다. 무슨말이냐면 빨간색 네모가 각각 attention을 뽑는 기준이로 회색 네모가 패치이다. 위에서 16x16으로 자른다고 치면 밑에는 이제 그 전체 이미지를 4등분하고 이 빨간색 네모 안에서 8x8 patch를 하나의 비주얼 워드로 사용하는것. 또 이걸 16개로 잘라가지고 이제 여기 하나를 이제 4x4 짜리 patch를 잘라서 그걸 visual words로 써서 또 하나의 transformer를 먹이는것이다. 그래서 이렇게 hierarchical한 feature map을 구성한다는점. 이렇게 하면 장점은 local windows로 이제 local self-attention을 적용할때 ViT와 Swin Transformer의 큰 차이가, global attention을 적용한다는게 ViT 인데 Swin Transformer는 local attention을 적용한것.
  - 이제 complexity가 local self-attention을 쓰기 떄문에 잘린 grid의 전체 개수만큼 개수로 잘라져가지고 image size가 늘어날떄마다 쿼드라틱하게 complexity가 늘어나는게 아니라 linear하게 complexity가 늘어난다. 그래서 linear computation complexity이다.  이제 이미지 사이즈가 커져도 . 즉 image의 큰 reolustion을 local window기반의 self-attention으로 cover를 해주고 이제 그 visual entity의 다양한 스케일을 커버하기 위해 이제 또 패치를 여러개를 잘라가지고 따로따로 attention을 구한 다음에 합쳐주는 일을 함으로써 좀더 좋은 결과를 냈던 논문이다!! 

이미지!!!!
- figure 2. feet window 방식을 보여준것이다.
  - layer L에서는 regular gird를 잘라서 각 window에 대해서 빨간색 windows에 대해서 self-attention을 계산한다. 하지만 다음 window에서는 똑같이 window를 자르는게 아니라 조금 더 다른 영역들을 보기 위해서 window를 이렇게 좀 다른 형태로 자른다. 그래서 그 윈도우 파티셔닝이 이렇게 shift되서 window가 shift된다는것이다.즉, 논문제목처럼 shifted window가 이렇게 window가 옆으로 이동한다. 윈도우가 다른 방향으로 어떤 gird를 나누는 방식이 이동한다는것이다!
  - 이렇게 자르면 문제점이 boarder line쪽에서 어떤 집이 이렇게 잘려버리니깐 이게 형성관계를 학습할수가 없다. 그래서 옆에서는 다른 윈도우를 shifting함으로써 해당 visual window내에 있는 local window안에 있는 visual관계들도 학습을 할수 있다는점이 장점이다! 그래서 바운더리 이펙트를 줄임으로써 좀더 다양한 어떤 이미지의 로컬관계들을 학습할수 있는 구조를 제안한것. 그래서 사이에 커넥션들이 한번에 다 학습이 된다. 로컬 셀프 어텐션을 쓰면!!
  오른쪽의 그림(Layer L+1)의 회색으로 나누어진 부분들이 patch를 나타내는 영역들이다. 그러니깐 이렇게 윈도우가 잘리고 각각 패치 샘플링이 되서 이걸 각각 visual word로 써서 transformer에 input으로 넣어주는것이다.

이미지!!!
- swin transformer의 전체적인 구조
  - (a) image가 있고 partition을 해서 patch를 먼저 나눈다. 그다음 linear embedding으로 swin transformer block을 넣고 또 stage2에서 swin transformer stage3, stage4 해서 resolution을 점점 줄여나가고 channel은 그만큼 키운다. 
  - (b) swin transformer block 곱하기라고 써있다.연속된 swin transformer block구조인데 이제 먼저 input이 들어오면 layer normalization을 하고 W-MSA(window multi-head self attention), 즉, 윈도우라는 것은 그냥 이제 local self attention이다. 윈도우로 잘린.. 그래서 figure b의 왼쪽부분에서는 local self attention을 그대로 적용해준다. 그다음 단계로 넘어가서 local로 하면 boundary effect가 발생하기 때문에 그 잘리는 영역에 대해서는 계산을 못한다.왜냐하면 관계를 잃어버렸으니깐. 그걸 없애기 위해서 SW-MSA(Shifted window multihead self-attention)을 계산해 준다. 그래서 이 블럭이 있고 여기서도 layer normal 그리고 residual conntection 그리고 mlp를 통과해서 최종적인 output을 내는 2단계의 연속적인 swin transformer block예시가 있다. 
  - figure 4.  swin transformer에서 efficient할떄 batch computation을 위해서 window partitioning할때 padding을 보이는 방식처럼 한다. 윈도우 partitioning이 이렇게 잘리면 여기 abc영역이 밖으로 나가버리게 되는데 이걸 붙여서 cyclic하게 붙여서 masking를 한다는것이다. 또는 이제 reverse cyclic  shift 이런 방식으로 bach computation을 좀더 효율적으로 한다는점이 여기 mask msa multihead self attention이 주는 효과이다. 즉, batch computation을 효율적으로 하는방법이다.

이미지!!
- results of image classification
  - 2가지 pretraining dataset을 가지고 학습(regular imageNet-1K & ImageNet-22K)
  - 해당 논문은 msra(microsoft research asia) 즉, 마이크로소프트에서 나온 논문. 그래서 다양한 베이스라인에 대한 실험과 비교를 할수 있었음.
  - 1K부터 결과를 보자.
    - 이전의 SOTA였던 regnet, EffNEt 그리고 ViT 그리고 Deit랑 비교했을때 같은 resolution일떄 이미지를 넣었을때에도 다른 모델들보다 성능이 더 높게 나옴.
    - image size를 384까지 키웠을때에도 parameter수는 좀 많지만 SOTA성능을 찍음(가장 높은성능!) 그래서 각 모델사이즈에 대해서 성능향상있었음. 
    
  - 22K를 보면 (더 많은 데이터셋을 사용헀을떄) 성능개인이 더 컸음. 그래서 파라미터수도 약간 늘고 FLOPs도 늘었지만 regnet 101보다는 paramter수는 적었고 FLOPs도 작았다.  그리고 throughput은 이미지는 더 많이 들어갈수 있었다. 

  - 똑같은 이미지를 봤는데도 regNet에 비해서 파라미터도 적고 FLOPS도 적고 이미지를 더 많이 feed forwarding을 할수있으니 더 좋은 모델임을 알수있다. 그래서 이러한 image accuracy를 냈고 v100 gpu로 이제진행했음을 보여준다. 이모델은 대규모 모델 라지 스케일 모델을 이제 사용한것이라 큰 gpu가 또 요구된다.

이미지!!
- results of object detection
  - imageNet-1K에서 보면 같은 resolution일때 같은 image size를 넣었을때도 Swin이 높은성능보임. 384까지 키웠을때도 parameter수는 많지만 SOTA성능을 찍음.
  - imageNet-22K보면 throughput는 image가 더 많이 들어감을 의미한다.
  - 똑같은 이미지를 봤는데도 regnet에 비해 파라미터수도 적고 FLOPS도 적고 이미지를 더 많이 피드포워딩을 할수있음이 증명됨.
  - SOTA를 찍는 backbone이나 architecture들은 항상 ImageNet classification에서만 평가를 했었지만, 이 논문은 object detection, semantic segmentation에도 얼마나 down stream task 성능이 잘나오는지 평가를 해준다.
  - (a) various frameworks 를 보면 backbone을 cascade, mask R-cnn 등 에 대해서 regnet 50 backbone을 쓴거랑 swin-T 쓴거랑 비교함. parameter 수, FLOPs, FPS(1초에 프레임 퍼 세컨드, 1초에 프레임 몇개 보는지 비교). 거의 compareable하게 efficiency 나오면서 결과는 더 좋게 나옴을 확인할수 있다. a는 ms coco dataset에서 평가한것.
  - (b) 는 백본끼리의 비교.cascade mask r-cnn썼을때 deiT와 Regnet50 그리고 swin 등을 보면 비슷한 수준의 efficiency로도 더 좋은 효과를 보임을 알수있다. 
  - (c) '#' 은 멀티스케일트레인 태스킹한것. 즉, 인퍼런스 타임때 이미지를 여러개 이미지 피라미드를 만들어서 하면 늘 개인이 어떠 task는 개인이있는것은 좀 알려진 사실임. scale의 roburst를 보장해주니깐. 이게 구조적으로 어떤 다양한 스케일 패치를 여러개로 잘라 가지고 shifted window를 써서 local self attention기법이라고 하더라고 내부적으로 스케일을 고려한다고 하더라도 이미지 피라미드를 쌓아서 멀티 스케일 테스팅을 하는 *를 보면은 효과가 있다는 점을  암시한다. 

이미지!!
- result of semantic segmentation
  - ADE20K 라는 dataset 에서 사용.
  - Uper-Net을 이제 method로 했을때 backbone을 swim-T로 바꾸면 성능이 좋아짐을 알수있다. 물론 large model을 parameter나 FLOPS가 많은데 표만 보면 아주 좋아졌다라기보다는 잘 구실이 뭔가 맞춰서 테스트를 했다는것이고, Large scale data에서 프리 트레이닝을 하고 파인튜닝을 한 결과에서 이제 이렇게 결과가 잘 나온다는 것.
    - DLab v3라던지 기존에 좋았던 모델보다 성능을 향상시켰다는게 핵심임.
    - 여기선 backbone부분의 차이를 보면 좋을거 같다.
  

이미지!!!
- ablation study
  - shifted windows 방식이랑 position embedding 방식을 바꿔서 했을때 어떻게 성능이 달라지는지 . 즉 이 논문에서 제안하는 기법이 얼마나 효과가 있는지 보여주는것.

  - shifting 없이 했을때 image net의 classification이정도 나오고 coco에서 object detection task에서 이렇게 나오고 AP는 average precision으로 box average precision과 mask average precision이다. 그래서 ADE20k는 semantic segmentation이고, mIOU는 mean inter section on union. 얼마나 세그멘테이션 마스크를 잘 맞추는지에 대한 결과로서 shifted 안쓴것보다 쓴게 일관성있게 성능이 올라감을 알수있다. 
  - 밑에 table은 positional 인코딩을 어떻게 주느냐에 대한 차이.
  - no pos : 포지셔널 인코딩 안준것.
  - rel pos : releation positional encoding 한것.
  - w/o app : 이전에 썼었던 테크닉안썼을때의 결과
  - 앱솔루터 포지셔널 인코딩이 어떤 위치를 해킹하게 해서 약간 잘되게 할수도 있으니깐 여기서 제안하는 릴레이셔널 포지셔널 인코딩은 상대적인 그 좌표값을 주는것이다. 왜냐면 이미지를 잘랐으니까 전체적인 값을 알수없잖아. 그래서 이런 방식이 효과가 있었음을 나타냄.

  - Swin ,t,s,b,l 구조로 커지는것이다. 
  - throughput, 이미지를 1초에 몇개나 이제 볼수있는지랑 top-1 accuracy 결과를 나타내줌. 그래서 모델들에 대해서 시간과 어떤 인풋 사이즈, 인풋 이미지 resolution에 대해서 얼마나 성능이 증가하였는지를 보여줌. 즉. 모델사이즈가 커질떄마다 걸리는 시간은 좀더 오래걸리지만 성능이 증가하고  input size가 커질때마다 성능이 증가하지만 시간이 좀더 오래걸리는 특징이 있다.

- 코드보기
  - [swin-transformer github](https://github.com/microsoft/Swin-Transformer)
  - object detection COCO에서 rank15위중.
  - instance segmentation에서rank 7위중.
  - AED20k dataset이용한 semantic segmentation 
  - kinetics-400 dataset 이용한 action classification 
    - 비디오 액션 레커그니션이다. 

  - 각각의 task에 대해 이용하고 싶다면  각 task에서 이런성능을 낸다는게 나온다.
  - object detection은 호크에서 레퍼지토리를 파가지고 이렇게 mask r-cnn 등 백본바꾸어서 swin-T 결과를 내는 실험도 세팅함.
  - Video Swin Transformer를 보면 3D 토큰들을 만들고 레이어 1에 대해서 swin을..그러니까 window local attention 자르는걸 temperal 방향 즉 시간축방향으로 잘라서 결과를 만든다. 
    - something-something v2이나 kinetics 600에서도 평가함.
    - 비디오는 워낙에 heavy한 리소스를 요구하는 task라서 실험이 어려움이 있다. 어쨋든 비디오 액션 레커그니션에서 쏟아내는 멀티뷰 트랜스포머 스포얼 비디오 레커그니션이라는 트랜스포머 기반의 방식을 사용한다.(action classification on kinetics-400을 보면 현재 ) - https://paperswithcode.com/sota/action-classification-on-kinetics-400 여기사이트 참고.. 그러면 SOTA는 InterVideo-T임을 알수있다.
  - 즉 task를 하는 repository가 잘되어있고 swin transformer도 pytorch로 구현되어있다.
  - third party Usage and experiments를 보면 face recognition 등 다양한 task에서 이제 백본을 활용하고 있음을 알수있다.


- 모델구조를 보자
https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

- torch를 임포트하고 MLP를 먼저 만들어준다. MLP처음에 코딩하는 윈도우 파티셔닝을 어떻게 자를지 이미지가 쭉있으면 이미지를 자르는 윈도우 파티션 함수가 있고 윈도우 리버스 함수가 있다. 아까 그 배치를 efficient하게 만들기 했던 트릭 기법을 하기 위해서 윈도우 리버스를 주고 윈도우 어텐션 w-msa모듈(windows attention)이다. 그 파티션이 없이 그 relational position bias를 포함해서 relational position을 사용해서 윈도우에 attention을 주는걸 만들어서 계산을 해주는 w-msa모듈이다. 똑같이 어텐션을 계산해서 어텐션 이렇게 하고 어텐션에 소프트맥스를 해주고  그다음에 드랍아웃 몇개 날려주고 이제 어텐션에 그 value , key, query로 만들어진 어텐션에 value를 matrix multiplicaion을 해서 이제 projection하고 return하는 거로 구성되어있다.
- 최종적으로 swin transformer block이라고 해서 이제 input channel이나 input resolution, 등을 arguments로 받는다. 그래서 위에서 정의했던 window attention 그 w-msa를 이제 정의를 해줌.

- forward해서 윈도우부터 먼저 자르고 그 잘려진 윈도우에 대해서 이제 어텐션을 계산함. shifted window나 그냥 window msa를. 그래서 window merge하는 부분이 있어서 이렇게 있고 마지막 이런 drop pass같은걸 해서 결과를 return해줌.

- 패치 merging하는 layer. 패치들이 이제 결국 잘려진 패치들이 합쳐져야 하니깐 나중에 classification하려면  merging하는 class도 있고 basic layer도 정의되어 있다. 패치 임베드하는 패치임베딩을 해야함. 패치가 이제 하나의 1D vector로 만들어져야 이제 그거를 transformer에 넣을수 있으니깐. 그 패치 임베드도 여기 정의되어 있음. 그래서 또 swing transforemr block을 이제 swing transformer에 결국 넣어야 된다. 결국 swin-transformer architecture를 코드로 그대로 구현한것이다.


## Swin Transformer V2
이미지!!
- motivation : scale을 키워보자. let's scale up!!
  - 모델사이즈를 키워보자는 뜻이 아니라 pre-training하면 큰모델이 있다.그럼 downstream task에서 보통 fine-tuning을 한다. 그런데 pre-training은 224x224, 384x384 등의 정해진 이미지 사이즈가 들어가는 것을 가정해서 그냥 그모델에 맞게 만들면 되는데 fine-tuning은 어떤 사이즈의 image가 들어오는지 모른다.  또 그게 기존 모델을 fine-tuning했을때 인스테이블하게 gradient가 흘러서 학습이 이상하게 될수도 있다. 그래서 그런점을 motivation으로 삼아서 나중에 이미지 resolution이 더 크게 들어올수도 있고 더 다르게 안정적으로 학습이 되도록 모델을 구현을 해보자는게 sWin-Transformer V2의 핵심이다! 
  - 그래서 최대 330밀리언 파라미터와 1536x1536 이미지에서도 사용가능.

- V1과V2가 달라진점. V1에서는 attention 구조가 multi-head self attention을 만들고 window based. 그리고 parameterized, 릴레이셔널, 포지셔널 바이어스가 들어간다. 그래서 softmax해서 attention을 이쪽 왼쪽영역에서 만들고 W의밸류값을 맨멀로 매트릭스 멀티플리케이션으로 곱해져서 다음 아웃풋을 만드는게 V1이었다.
- V2는 layer norm이 앞에서 뒤로 간다!! 그리고 input이 들어왔을때 simularity를 구할때 다 project를 하는게 아니라 key, qeury의 cosine 시뮬러리티 그리고 템퍼러처를 나눠주는 스케일드 코사인 시뮬러리티를 구한다. 그리고 릴레이셔널 포지셔널 인코딩을 하던 부분을 포지셔널 바이어스를 추가해주던 부분을 로그CBP라고해서 로그 스케일에 컨티뉴어스 포지셔널 바이어스를 더해준다. 그리고 MLP를 통과시켜서 학습을 이부분에도 상태 포지셔널 인코딩 부분에도 작은 네트워크를 통해 학습을 해주는 핵심적인 3가지 컨트리뷰션이다. 오른쪽으로 보면 노멀라이제이션을 어텐션 다음에 한다는것. 즉 adaption 1,2는 모델 스케일을 더키우는데 도움을 주도록 레이어 놈을 나중에 하고 코사인 시뮬러리티를 어텐션을 쓰면서. 그 이유가 큰사이즈의 resolution image가 들어올떄 도움이 되도록 하기 위해서! 세번째를 보면 추가하는이유는, transfer할때 더 어떤 윈도우 레졸루션에 대해서도 더 효과적으로 적용이 되도록 만들어주는것. 

이미지!!
즉, 3번쨰가 무슨말이냐면, port normalization layer를 뒤로 옮긴것이고 dot-product대신에 cosine attention을 구하고 거기에 temperature를 나눠주는 역할을 하는 거고  (곱해주는거 아님. 표가 잘못된거야.) continuous relative position은 큰이미지가 들어오면 윈도우 사이즈가 크게 잡힐거잖아. 그러면 해상도가 커지면 윈도우의 상대자 차이도 너무 커지게 되고 그런 문제가 생길수 있는데 이걸 continuous meta nework라는건 MLP이다.  이걸 만들어서 어떤 continuous position을 맞추도록 미리 정해진 디터미틱한 포지션이 아니라 continuous 포지션으 학습해가지고 맞추도록 하는것. 큰 이미지가 들어와서 윈도우가 크게 잘린다고 하더라도 포지션들에 따라서 그 포지션이 학습이 되는것이다.  그리고 추가적으로 그냥 일반적인 정비례하는 값의 코디넷을 쓰는게 아니라 log-space coordinates을 쓴다. 해상도가 커지면 상대좌표의 차이도 너무 커져가지고 그 차이 떄문에 어떤 gradient의 흐름이 언스테이블할수 있는데 그 값이 차이가 너무 커지는걸 막기위해서 log scale로 바꾼것이다.

- abilitary study 부분.
  - backbone을 기존 Swin-T나 ViT로 썼을때 여기서 제안하는 post normalization(residual post-norm)은 layer norm을 뒤로 옮기는 거랑 scaled cosine attention을 추가하고 뻇을때 성능이 달라진것. 그래서 결과를 보면 없어질수록 성능이 더내려간다.
  - L-CPB 를 빼고 안뺏을때 backbone으로 Swin V2-S를 쓰고 imagenet에서 성능과 imageNet W12, S16은 모델 사이즈도 커지고 이랬을때 L-CPB가 있어야 성능이 더 잘오르는걸 확인할수 있다.

- 논문보기
  - scale up 하는 방법들, language model, vision model에서의 기법들. 그리고 transformer transfering을 어떻게 하는지, 윈도우나 커널 레졸루션에서 트랜스포머를 어떻게 할지. 바이어스에 대한 스터디. 뒤에 포지셔널 바이어스가 있었다. 랠러티브 포지셔널 바이어스가 어떻게 더해지는지, 
  - 리니어 스페이스로 하면 차이가 너무 커지기 때문에 사이즈가 아주 커질수록 성능이 떨어진다. 로그스페이스로 하면 그 차이가 줄어들어서 성능이 안정적으로 나온다는 점. 

- 만약에 엄청 성능을 높게 내야 되는 다운스트림 태스크에서 파인튜닝을 하고 싶은데 SOTA모델을 내가 진짜 제일좋은걸 쓰고 싶다면 2022년 5월 기준 Swin-Transformer이다.(paper with code에서 SOTA모델 찾을수 있다.) 허깅페이스에도 이제 인테그레이션 되어 있으며,  여기서 모델 쉽게 다운받아서 사용가능함. 


# Representation learning
# Object detection & segmentation
# Video
# Multiview geometry
# 3D Vision

## overview
- 3D 비전이란 3차원 비전이다.
- 우리가 사는 세계는 2차원의 어떤 이미지로 표현가능.실제론 3차원으로 구성. X,Y,Z좌표가 추가된 형태
- point cloud : spot한 점들의 집합.
## 3D representation: Point Cloud, Voxel, Mesh
- 3D 모델링시 고려해야할점.
  - 어떻게 컴퓨터에서 3D object를 표현할 것인지가 중요
  - 컴퓨터와 함께 그런 representation을 어떻게 컨스트럭션 할것인지, 그런데 그걸 어떻게 빠르고 더 자동적으로 construct 할것인지.
  - 컴퓨터를 통해서 3D object를 어떻게 다룰것인지.
Q) 서로다른 object representation에 대해서 건물을 한다거나 , 3D map을 한다거나, 어떤 object를 한다거나에 대해서 서로 다른 method(voxel, mesh 등을 선택해야한다.)를 사용하는게 좋다. 상황에 적절한 representation을 사용했을때 더빠르고 더 자동적으로 어떤 3차원 구조들을 표현할수 있다.
( 내부에 대한 값들을 가지고 있을 필요가 없을떄는 voxel을 굳이 쓸필요는 없다.)
- 즉, geometry에 대해서 representation을 어떻게 주느냐가 중요하다!!
- geometry를 discriving하고자 할떄 여러가지 language가 있다.
|sementaics|syntax|
|---|---|
|values|data structures|
|operations|algorithms|

즉, 우리는 data structures위에서 어떤 동작하는 알고리즘들에 대한 문법을 가지고 있고, 이제 거기에서 value가 어떻게 달라지느냐에 따라서 모양새가 달라진다. 가장중요한건 data구조를 어떻게 정의하느냐이다!! 즉 이걸 point cloud로 정의할지, mesh, voxel로 정의할지 또는 요즘과 같은 인프리스 function(어떤 object를 가르는 표면)으로 정의할지에 따라서 어떤 구조가 달라질수 있게 된다.

- 3D object representation 예시
|Raw data|Solids|Surfaces|High-level structures|
|---|---|---|---|
|Point cloud|Voxles|Mesh|Scene graph|
|Range image|BSP tree|Subdivision|Skeleton|
|Polygon soup|CSG|Parametric|Skeleton|
||Sweep|Implicit|Application specific|

- Point Cloud
  - 3D point samples의 unstructured set.
    - 고전적으로는 range finder, computre vision을 통해서 3D point cloud 를 reconstruction(복원)한것.
    - 최근에는 LiDAR를 이용하여 3mm속도를 스캔하여 어떤 거리를 재서 point cloud를 scan한다.
    - 3차원 vision의 가장 기초가 되는 representation이다.
- Range Image
  - depth image의 pixel들을 mapping한 3차원 points
  - range scanner같은 걸로 얻을수 있다.
  - stereo match같은걸 해서 image로 부터 얻어낼수도 있다
  - range image가 point cloud와 같이 있을때 teselation으로 조금더 dense하게 만들수 있고,  range surface를 이어가지고 구조도 복원해낼수 있다.
- polygon soup
  - 여러가지 polygon 들에 unstrured set.
  - 모델링 시스템이 어떻게 만들어진 것을 interactive할지에 대한것도 큰 이슈이다. 이것은 이제 우리는 크게는 point cloud, mesh, voxel만 다루지만 이런것도 있다는걸 알아두자.

- surfaces : 3D object representation.
  - mesh : polygon 기반의 어떤 surface 기반의 data structured.
  - mesh는 polygon들의 여러가지 connected set이다. 그래서 triangles를 주로 사용한다. 또한, 어떤 표면에 대해서 삼각형들의 집합으로 이저여 있다. mesh의 특징은 꼭 closed 되어 있지 않을수도 있따는것. 어떤 표면을 따라가서 순환되는 고리가 안만들어질 수도 있고, 어떤 오픈된 기능이 있을수도 있다는 것. 그래서 mesh로 만들었을때 조금더 삼차형 구조를 더 어떤 정밀하게 복원해내는 것처럼 보일수 있다.

  - subdivision surface
    - Coarse mesh와 subdivision rule을 추가해서 좀더 스무스한 surface를 만들기 위한것.
    - 처음에 이런 어떤 메시 구조의 사람 얼굴 표면이 이제 스캔되었다고 했을때, 이걸또 devided 해서 더 촘촘한 mesh로 만들고 더 세분화시켜서 더 촘촘한 mesh로 만드는것. 이런작업들은 컴퓨터 그래픽스나 어떤 영상 합성같은곳에서 많이 수행되고 연구되고 있다.

  - parametric surface
    - spline patch의 tensor product로 이루어져 있다.
    - 그래서 어떤 패치들이 이렇게 있을때 그 표면들을 나타내는 어떤 행렬식이 있고 거기에서 continuity를 보장해주는것이다. 그래서 continuity한 어떤 mesh같은 구조를 parametric하게 구성해낸것.

  - Inplicit surface
    - inplicit한 모델을 사용해서 최근에 많이 사용되는 implicit function이다. NERF에서 많이 사용되는 구조인데 어떤 f는 xyz는 0이라는 surface 평면을 주고 저거보다 0보다 큰 경우는 바깥쪽, 0보다 작은 건 안쪽 이렇게 표현을 해서 polygon model로 표현한거랑 implicit model로 표현한게 큰 차이가 없게된다. 그래서 function형태로 어떤 surface를 정의해주는것.


- solid
  - voxels : solid한 구조체의 대표격
    - 우리가 이미지에서 pixel이 있으면 3차원에서는 voxel이라고 한다. 그래서 volumetric sample의 uniform한 grid이다.
    - 그리드별로 여기가 오피파이너가 돼있는지 아닌지, 그리고 RGB컬러 가지고 있다면 그 영역에 RGB값이 뭐가 있는지를 나타내는것.
    - 그래서 CAT나 MRI같은곳에서 얻어낼수 있다.
  - BSP tree
    - binary space partition tree라는것
    - 어떤 솔리드 셀의 label들이 있고 partition이 어떻게 나뉘는지에 대한거.
    - polygonal representation으로부터 construction을 한다. 그래서 binary spatial patrition이 이렇게 있고 그냥 object가 있을때 어떻게 선들로 나뉘는지에 대한것이 있고, 그 구조를 이제 binary tree로 결정을 해주는것이다. 4,5,6은 색칠이 되있고 나머지는 흰색으로 되면, 나머지는 surface들에 대해서 이부분이 오피파이너가 되있다고 표현을 할수가 있다.
  

- High-level structures
  - Scene graph
    - 요새는 2차원에서 어떤 object간의 관계나 human object interaction이나 이런 문제들로부터 많이 푸는데 결국엔 어떤 leaf nodes들에 대한 object의 union으로 나타나는것. 어떤 각각 씬에 대한 어떤 위치들을 이제 그래프 형태로 표현을 하는것. 그래서 어떤 관계성이 있는지 표현을 할수가 있다.
  - Skeleton
    - 어떤 반지름을 가지는 curves들의 그래프형태로 이어내는. 그리고 그래프 형태들이 이제 좀 스페셜한 structure일수 있다.
    - 사람 뼈다귀도 이렇게 어떤 조인트들의 어떤 있는것으로 표현할수 있고 스켈레톤으로 이을수도 있다. 그래서 human pose estimation과 같은 task를 풀기도 한다.
  - Application specific
    - 3D representation을 구성하는 경우도 많이 있다.
    - 그래서 이런 분자구조, 어떤 신약 개발같은거 할때 이런 분자구조에 대해서 그래프로 표현할수도 있다. 왜냐면 voxel로 표현하기에는 부적절하기때문에. 그래서 이런 분자구조나 또는 architectural floorplan, 즉 건축물에서 이제 어디가 어떤 영역에 어떤 방위인지를 표현해 주는것도 이제 사람손으로 그릴수 있지만, 어떤 카메라나 라이더나 센서같은걸로 스캔했을때 어떤 구조체로 정할지 이제 포인트클라우드가 일반적이지만 꼭 그렇게 하지 않을수도 있다. 그런부분들이 다양한 3D representation으로 구성될수 있다는점을 알아둬라.

  - Taxonomy of 3D representation
    - 3D shape이 있을때 discrete, continuous로 나눌수 있고, ....
    - topological은 mesh와 subdivision으로 나눌수 있는데 즉, 삼각형이나 사각형들의 연결된 형태로 topology를 표현할수 있다.
    - set membership은 BSP tree나 cell complex 같은 구조가 있을수 있다.

    - 여기서 우리는 continuous한 shape은 너무 computationaly heavy하고 우리는 그걸 통해서 인식을 한게 컴퓨터비전의 목적이기 떄문에 대체로의 목적이기 때문에 최근에는 그래픽 쪽이랑 많이 합쳐져서 NERF같은 model이나 generative model들이 많이 나왔지만 기본적으로 point cloud나 voxel이 어떤 의미를 가지고 그게 어떤 구조를 나타내는지를 알고 싶은게 우선적이어서 대체로는 이번섹션에서는 voxel과 point set, point cloud에 대해서 알아보자. 이것이 조금 더 raw data이고 처리되지 않은, 가공되지 않은 형태로 풀린것이기 때문. 

  - Equivalence of Representation
    - 각각의 fundamental representation은 어떤 geometric object의 모델링을 할때 충분한 expressive한 power를 가질수 있다는 것이 각 fundamental representation들의 특징이다.
    - 그래서 모든 fundamental representation을 사용하더라도 geometric한 operation들을 다 다룰수 있다는 것이 위에서 말했던 representation의 특징이다.
    - turing-Equivalence 관점에서 분석은 모든 컴퓨터는 turing-Equivalence한, 예전에 turing machine이나 오토마타 수업 들었을때를 생각해 보면 요즘에 나오는 모든 컴퓨터는 모두 turing machine이다. 즉, turing machine의 operation들로 이루어져 있는데 이제 우리는 그런 컴퓨터뿐만이 아니라 다른 프로세서들이 많이 생겼기 떄문에 그런 프로세서들이 닫힌 set으로 모든 튜링 컴플릿한 튜링 Equivalence한 구조를 생성할수 있따는게 특징이고 그런 컴퓨터 위에서 우리가 3차원 representation들을 활용할수 있다는 것이다.

  - Computational Differences
    - 우리가 데이터 structure에 따라서 즉, 3D representation에 따라서 잘 생각을 해야한다. 그래서 combinatorial complexity를 가지는지, 예를 들어 N log n같은거.. N 스퀘어의 폴리노미어란 컴플렉스티를 가지는것에 대해서 생각을 해봐야 한다.
    - space/time trade-offs : 공간을 많이 차지할수록 시간 복잡도는 낮아질수 있지만 공간은 한없이 커질수도 있다. 또는 공간을 좁게 가진 대신에 시간 복잡도가 더 복잡해질수도 있다. 그것을 원하는 application이나 하고 싶은것에 따라서 결정을 한다.
    - numerical한 accuracy나 stability도 중요하다. 그러니까 어떤..계산 불가능한 수준으로 계산량이 너무 커지면 애초에 알고리즘을 디자인해도 할수가 없잖냐. 2D 스페이스는 2차원으로만 쭉 이어지기 때문에 요즘 나오는 모던 GPU같은 걸로는 충분히 일반적인 시간내로 처리를 할수가 있는데 3차원 구조로 axis가 하나 늘어나는 순간부터 이제 100년씩 걸릴수도 있고, 알고리즘을 잘못 구현하게 되면 그런일 이 생길수 있다. 그래서 numerical하게 accuracy나 stability를 고려해서 데이터 structure를 잘 구성해줘야 한다.
    - simplicity
      - 어떤 acquisition할때 편해야 한다. 즉 데이터를 모을때 편해야 한다. 예를들어 라이더 같은거 쓰면 point cloud를 쉽게 얻을수 있다. 
      - hardware acceleration도 쉽게 될수 있어야 한다는 특징이 있다.
      - software..그 data structure를 가지고 software를 만들거나 유지 보수할때에도 편해야 한다는 특징이 있다. 
    - 그리고 usability관점에서도 user interface를 더 편리하게 할것인지 아니면 computer engine을 활용해서 컴퓨터가 인식하기 편하게 할것인지에 대한 고민도 필요하다.

  - complexity vs verbosity 즉 어떤 편리성과 복잡성의 관계
    - 대신 너무 편해지거나 간단해지면 더 inaccuracy해진다. 그렇지만 복잡해질수록 더 정확해질것이다. 
    - 그래서 pixel / voxel은 이제 좀 다루기는 쉽지만 부정확할수 있다. 왜냐하면 어떤 영역에 대해서 그냥 컴타이즈를 해가지고 한아ㅢ 오큐파이션으로 하니깐. 그래서 voxel의 크기가 어느정도 되느냐에 따라서 어플리션이 달라질수도 있다. 
    - 그리고 piecewise linear polyhedra, low degree piecewise non-linear, single general functions으로 구성할수록 더 복잡해지지만 더 정확하게 표현할수 있다. 그러니깐 한 어떤 위치들에 대해서 각각 single general한 function들에 연속으로 표현을 하는것이다. 


## point cloud recognition

- 3D representations
  - multi-view RGB(D) images
  : multi-view 이미지들에 대해서 pre-spontes를 찾고 거기에 대해서 3차원의 의자 형태를 복원해낼수 있음.
  - Volumetric
  : 비행기가 있다고 했을때 비행기를 어큐파잉하고 있는 국세열의 크기들을 정의해서 그 위치에 어큐파이가 되어있는지 아닌지 알수 있음.
  - Polygonal mesh
  : 폴리곤 구조를 기본으로 사용해서 그것들을 연결한 mesh 구조가 있을수 있음.
  - Premitive-based CAD modesl
  : premitive에 기반한 CAD모델. 그래서 산같은게 있을수 있고 연결된 형태가 있을수 있다.
  - Point cloud
  : point에 대해서 그 좌표값들을 가지고 있고, 그좌표가 각각 어디에 해당하는지. 그래서 이 좌표를 모은 포인트 클라우드가 있을수 있다.
    - point cloud를 얻는 방법은 color camera가 있을때 depth camera와 같은 방식으로 깊이까지 알아서 XYZ좌표를 그대로 알아낼수 있는, 그래서 point cloud로서 triangulated된 어떤 좌표점들을 사용해 가지고 포인트 클라우드를 얻어낼수 있다. 

  - image vs 3D geometry
  |-|Image|3D geometry|
  |---|---|---|
  |Boundary|fixed|Varying|
  |Signal|Dense|Very sparse in 3D domain|
  |Convolution|Well-defined|Questionable|

  이미지는 
    - boundary부분이 정의되어 있고 이미지의 3차원 정보가 2D projection된 정보를 받아들인다.
    - singal은 dense한 어떤 값들이 추출되는것.
    - convolution같은경우 이미지에서 슬라이딩 윈도우로 정의가 된다. 그래서 GPU로 표현을 하게 되면 matrix연산으로 빠르게 연산할수가 있다. 그래서 well-defined되어 있다.
  3D geometry는 
    - boundary가 없을수도 있다.이렇게 튀어나올수도 있고, 연결된 외역들이 있을수 있다.
    - signal이 3D domain에서 매우 spot하게 존재한다. 현재 이미지를 보면 우리가 2차원으로 프로젝션 시키기 떄문에 꽉 차 있는 것처럼 보이지만, 실제로 3차원으로 구성했을때는 10%정도밖에 되지 않는 부분들이 Occupy되어 있다. 즉, signal이 아주 sparse하다.
    - convolution을 한다고 햇을 때, 빈 위치에 대해서는 convolution을 돌리면 아무것도 없어서 의미가 없게된다. 그래서 3D geometry의 convolution연산은 Questionable하다. 그래서 우리가 다음 포인트 클라우드 Registration부분에서 이런걸 어떻게 하는지, sparse convolution같은 구조들에 대해서 한번더 이야기 할것이다.

  - properties of Point sets
    - 순서가 없다.
      - XYZ라는 정보들의 set이라서 어떤 specific order가 없는 포인트셋이다.
      - 어떤 네트워크는 N개의 3D 포인트를 사용하는데, 그것은 N에 대해서 Inverient한, 그러니깐 어떤 Permutation의 N-Factorial Permutation에 대해서 Inverient한 set의 구조이다. 그래서 Data Fitting Order안에서 inputset의 N-Factorial Permutation을 가진다.
      - point들 간의 interaction이 필요하다. 그러니깐 local structure를 capture할수 있어야 한다. 그냥 포인트들을 이제 어떤 위치에 스케터링하게 되면 근처의 point들을 가지고 어떤 pattern이 만들어질 것이다. 패턴들을 캡처할수 있는 모델이 필요하다.
      - transformation에 대해서 inverient한것. 즉, 회전 변환이나 이런거에 대해서 axis를 가지고 있다면, 그 회전 변환을 기준으로 다시 되돌릴수 있게, 회전변환을 사용해서 다시 되돌릴수 있다. 그래서 transformation에 대해서 inverient한 특징이 있다.
        - rotationg과 translation, 포인트를 그렇게 하더라도 이제 어떤 structure 자체는 변하지 않기 떄문에 global point cloud의 category나 그 포인트의 segmentation은 이제 변하지 않는다는 특징을 가져야 하는것이다. 그래서 보통 point cloud는 scale이나 어떤 translation에 대해서 inverient한 구조이다. 즉, 어떤 center point 값들을 다 normalize를 하게 되면 그냥 0,0,0에 얼라인이 될수 있다는 것이다. 근데 꼭 이게 scale이나 rotation에 대해서 inverient하게 추출이 되지 않을수도 있다. 그래서 이걸 잘 맞춰주는 axis를 찾아주는 test를 풀수도 있다.
- PointNet for Point Set Understanding
  - [PointNet 논문]
  - 딥러닝 시대의 point recognition 즉, point cloud classification, part segmentation, semantic segmentation 같은 point level의 segmentation을 하거나, 어떤 object인지 classification하거나 part들을 즉, 탁자가 있으면 탁자 다리인지 아니면 상인지 이런걸 하는 part segmentation이 이싸.

- pointnet : Architecture 
  - 어떤 MLP, classification network와 segmentation network로 나뉨.
  - classification network
    - N x 3의 input points가 있다. t-net을 통과시켜서 3x3 transform을 통해서 간단하게 transform을 진행한다. 그다음 shared MLP를 통과시켜서 각 포인트에 대해서 point feature를 어떤 좌표가 아니라 feature로 하나씩 만들어주는다. 그래서 각 포인트는 어떤 embedding을 가지고 있는 64x64와 같이 된다. 다시 t-net을 통해서 64x64 transform을 통해서 matrix multiply으로 한번 transform을 또 수행해준다. 다시 이제 n x 64의 구조가 있을때 이제 MLP를 3단계 연속으로 통과시켜서 64 x 128 , 1024까지 dimention을 키워서 하나의 point가 갖는 이제 representation power가 1024 dimention까지 키운다. 그 다음 이 결과를 global max pooling을 통해서 global feature를 하나 구해낸다. 다시 MLP릘 통해서 512, 256, k로 이제 줄여서 K개의 classification label에 대해서 classify를 하는 output score를 얻어낸다. 그래서 classification branch는 다음과 같이 진행이 된다.
  - segmentation network
    - 아까 계산했던 M x 64짜리 각 포인트에 대한 어떤 feature와 global feature를 concat한다. 그래서 N x 1088의 구조를 갖는다. 다시 shared MLP를 통과시켜서 512, 256, 128로 이제 순차적으로 차원을 줄인다. 그래서 각 point feature는 Mx128짜리 point feature가 된다. 이제 다시 shared MLP를 한번더 통과시켜서 N개의 segmentation lable을 semantic segmentation lable을 갖는 N x M짜리의 output score를 얻어낸다.

  - MLP는 어떤 point-wise로 진행되는데 independent한 operation을 수행을 한다. 즉 multilayer perceptron처럼 우리가 알고 있는 MLP fully connected layer를 통과시키는것이다. 그래서 이게 하나의 point feature XYZ 좌표라고 했을때 하나의 layer를 통과하면 64짜리 feature가 되고, 또 하나의 layer를 통과하면 64 x 64 짜리의 어떤 matrix로 weight를 update하게 되어 값이 나온다. 당연히 RELU function, non-linearity를 주는 function이 당연히 있다.

- Multi-layer perceptron(MLP)
  - input layer에 M개짜리의 neuron이 있어서 input을 이렇게 받게 되면 hidden layer에 N짜리 neuron이 있어서 Vnm의 어떤 edge score들을 가지게 되고 이제 weight가 되어서 weighted sum이 된다. 또하나의 output layer에 K개의 neuron이 있다고 하면 이제 k개가 계산이 되서 최종적으로 이런 feature들이 만들어지게 된다.  그래서 어떤 하나의 point를 update할때 나머지 전부를 고려해서 그 weighted sum을 하는게 이제 multi-layer perceptron이다. 가운데에는 물론 non-linearity를 주는 RELU나 sigmoid가 있다. 그래서 어떤 X를 Y로 변환시키는 그런 함수(f)라고 보면 된다. 거기서 X의 개수는 유지가 된다. ZX, ZM... ZM개까지 있고 Y도 Y1,Y2..YM개까지 있다.
  - 모든 mapping functiond을 가장 간단하게 표현할수 있는 방식이 MLP이다. 그래서 training할때 sharing하면서도 어떤 update를 할 수 있는 가장 기본적인 구조이다. 
  - input node를 제외한 나머지 node는 neuron으로 구성되어 있는데 non-linear activation function을 반드시 사용해야 한다.
  - MLP는 multiple fully connected layer, 최소 3개 이상의 fully conntected layer로 구성이 되어야 한다.
  - input, output dimension에 대해서 쉽게 적용할수 있다. 즉, 앞에 M개의 point cloud가 들어오든 3개가 들어오든 M개가 들어오든 K개가 들어오든 개수가 늘어나는 거에 따라서 크게 구애받지 않고 shareing weight를 통해서 update를 할수 있다.
  - 단점이라고 하면 parameter수가 너무 많아진다는것. fully connected 되어야 되기 떄문에 예시만 봐도 M x N + M x K 개만큼의 개수가 필요하다.

  - general function을 approximate하기 위해 point set 위에서의 operation이 MLP라고 할수 있다. 우리는 항상 2차원에서는 convolution neuron network를 patten을 capture하기 위해서 사용했는데, 3차원에서는 convolution net이 잘 정의가 되지 않는다. 이때까지만 해도 정의하기 어려웠고 그래서 weight를 sharing하는 하나의 point마다 update를 하는 MLP를 사용 해서 이것을 정의한다. 그래서 어떤 set에 대한 element의 transformation symmetric한 function을 적용하는것이다. 그러니까 모든 위치에 대해서 같은 operation이 적용되도록 즉, combination한 아주 많은 경우에 대해서 approximate하는 어떤 함수로 정의를 하는것이다.

- 즉, architecture를 보면 classification network 와 segmentation network로 나뉘고, 각 operation 들은 MLP로 구성이 된다는 것이고 우리가 봤던 t-net은 아까 이 t-net이 뭔가 했을텐데 그냥 affine transformation matrix를 계산해가지고 만들어주는것이다. 우리 이전 multi vision matrix part에서 배웠던 spartial transformer network 를 생각하면 된다. 그래서 어떤 point를 transformation에서 우리가 인식하기 좋은 형태로 수정해주는 mini network라고 볼수 있다. 그래서 point가 어떠한 변환에 따라 들어왔음에도 다시 머신이 인식하기 쉬운 형태로 다시 되돌려주는게 t-net이다.  그래서 matrix multiply을 계속 3x3 transformation을 통해서 모든 point를 일관성 있게 옮겨줌으로써  좀더 인식하기 쉬운 형태로 바꿔준다.  그래서 optimize가 좀더 잘되게 만들어주고 싶디만 이 transformation matrix가 feature space에서는 optimize가 어렵게 된다. 그래서 regularization term을 softmax training loss를 통해서 추가를 함으로써 좀더 확보를 한다.
  - 어떤 otrhogonal matrix에 가깝게 되도록 feature transformation matrix에 constrain을 줘서 이러한 regularization term을 통해서 affine transpose가 이제 identity와 같아지도록 이렇게 regularization term을 추가해준다.  그래서 transformation T network를 학습을 하는것이다.

- Theorical Analysis
  - point net에서 또하나의 contribution은 theorical한 analysis를 제공해준다는점이다.
  - 그래서 MLp에 어떤 universal한 approximation을 제공해준다.
    - neural network는 어떤 continuous set function을 approximation할수 있는 ability를 가지고 있다.
    - network의 worst case도 어떤 volumetric representation안에 point cloud로 convert한 것을 배울수 있도록 충분한 representation power를 가지고 있다. 그래서 point를 equal size한 voxel로 approchmation하는 구조로 이제 이게 할습이 되는것이라는 분석을 내놓음.
    - bottlenect dimension과 stability
      - point net의 expressiveness는 어떤 max pooling layer의 dimenstion에 따라서 그 strong에 영향을 받는다. 즉 1024였는데 이제 그런 dimension에 따라서 영향을 받는다는 것이다. 그리고 이제 그 f(s) 라는 function은 어떤 k개의 element보다 작거나 큰 어떤 subset에 의해 결정이 된다. 그리고 stablity는 어떤 keypoint의 sparse set에 의한 shape으로 부터 요약될수 있다. 즉 이제 keypoint의 sparse set을 각각 merge함으로써 어떤 classify를 하게 된다.
    

- pointnet segmentation network
  - 이미 point cloud set이 있으면 table이라고만 할수 있는게 아니라 table의 다리 부분을 따로 point label classification을 할수도 있고, 머그컵 같은경우도 머그컵 부분이랑 손잡이 부분을, 모터바이크도 자전거의 바퀴와 안장과 몸체 부분들을 다 따로 partial하게 classification할수가 있다. 
  - pixel level classification이 segmentation이었던 것처럼 pointnet도 어떤 part level의 classification을 이제 수행해줄수 있다는 것이다. 
  - 그래서 partial한 input이 들어오거나 완전한 input이 들어왔을때에도 각각을 잘 segmentation하는것을 또하나의 motivation으로 잡는다.

- indoor semantic segmentation
  - indoor input image에 대해서 point cloud image를 넣었을때 의자나, 칠판 이런 창문같은 부분들에 대해서 서로 다른 sementic를 잘 학습해내서 이게 point가 변화해서 camera 위치를 바꾸더라도 segment를 잘해주는걸 확인할수 있다. 

- classification results of PointNet
  - 기존의 SPH나 3DShapeNet, VoxelNet, subvolume,LFD, MVCNN과 같은 baseline과 비교했을때 기존 point net 에서 volume, mesh, image를 사용하지 않고 point cloud만 사용했을때 즉, view 하나만 사용했을떄에도 overall accuracy가 기존 SOTA와 비슷하거나 더 좋게 나온다.
- segmentation results of pointnet
  - 기존 3D CNN을 사용한 point cloud segmentation 결과보다도 여기서 제안하는 pointNet ShapeNet 방식이 ShapeNEt part dataset에서 mIoU가 가장 높게 나온다.
  - invariance를 성취하기 위한 여러가지 방법들이 여기서는 max pooling을 사용해서 만들었다.  semantric function을 만들었는데 approaches achieve 한 feature를 inverient한 feature로 만들어주는데 그 방식을 바꿔보았을때에 대한 결과이다. 그래서 unsorted input에 대한 MLP를 사용하거나 sorted input에 대해서 MLP를 사용하는 or LSTM같은 구조로 한번에 취했을 때보다도 Attention sum이나 average pooling이 성능이 더 높았고 여기서 사용하는 max pooling이 오히려 이런 classification 구조에서 성능이 더 좋았다. 
  - 여기서 사용한 MLP는 5개의 hidden layer로 구성해가지고 마지막 classification을 하도록 구성했다. 그래서 max pooling만 사용해도 가장 성능이 좋았고  이런식의 구조가 max pooling이 실제로도 써보면 invarient mapping쪽에서 굉장히 효과적이다.그래서 나중에 꼭 MLP나 이런걸 사용할 필요 없이 어떤 지역적인 부분이나 전역적으로 플랩신을 할때 간단하게 max pooling만 사용하는 것도 효과적일 거라는 것을 하나 알고 넘어가면 좋다.

- Spherical Harmonic Representation(SPH)
  - point cloud classification network의 baseline논문중에 하나였던 것.
  - point cloud의 classification을 어떻게 할지, harmonic을 통해서 제안했던 논문.

- Multi-View CNN(MVCNN)
  - multi-view convolutional neural network를 사용해서 multi view 이미지들을 CNN에 넣고 각각 view pooling을 통해서 어떤 3차원 구조가 어떤 classification label을 갖는지 prediction하는것을 수행하는 network.

- Properties of Point Sets
  - 원래 convolution을 수행하려면 2차원 이미지에서는 이렇게 필터가 슬라이딩 윈도우를 하면서 결과를 얻는데 볼륨 matrix한 경우에는 이렇게 volume이 sliding하면서 하나의 값을 만든다. 
- multiview volumetric CNN(subvolume)
  - 또하나의 baseline
  - 3D CNN을 활용한 sub volume이라는 아까 point net classification table에서 볼수 있었던것.
  - 기존의 3D CNN을 하나의 point cloud set에 대해서 통과시키는 classification을 할 수도 있었지만 voxel이었다.  여러개의 multi view의 voxel들을 3D CNN을 각각 통과시켜서 orientation pooling을 하고 그다음에 또 다른 3D CNN을 통과시켜서 classification을 하는 논문이 pointNet의 baseline으로 있었다.

- comparison PointNet with Other Invariance Approaches
  1. Sequential Model (such as LSTM)
    - 포인트들의 어떤 값이 들어아고 MLP로 이제 feature를 1024 dimension 정도로 높인 다음에 RNN cell을 통과시켜서 sequential하게 여러개의 point feature들을 하나의 classificatino label로 바꾸는 방식
  2. MLP with sorted / unsorted input
    - 기존의 MLP로 어떤 unsorted data를 classification하는 것보다 효과가 좋음. 하지만 여기서 제공하는 MLP를 통과시킨 다음에 max pooling, average pooling, attention sum을 하고 또한번 MLP를 업데이트를 하는 PointNet 스타일의 symmetry function을 사용한 MLP 구조를 취했을떄 가장 성능이 좋았음.
  3. MLP with symmetry function (PointNet architecture)

- Summary
  - PointNet  
    - 3D coordinate의 unordered set으로 부터 만들어진 것을 어떻게 classification network로 사용을 해서 classification 문제를 풀고 segmentation 문제를 풀지에 대한 첫 시도.
    - SOTA 성능 달성.
    - Lightweight, shared parameter를 통한 간단한 가벼운 구조를 사용하게 됨.
    - 심플하면서도 effective한 구조를 제안함.
  - 질문
    - PointNet이 3D shape understanding을 완전히 풀었느냐?
    - universial한 approximate을 MLP를 통해서 풀었느냐?
    - 모두 아님. 왜냐하면 어떤 local context를 global하게 스키징하는. 지금은 모든걸 한번에 스키징 했엇는데 어떤 로컬 패턴들을 고려하지 못하고 있으니깐. 그래서 어떤 로컬 geometry를 implicit하게 하고 있지만 좀 explicit하게 구해줄 필요가 있다.
    - large scale data에서는 적절하지 못했다는 단점이 존재한다.
  - 이후에 나온논문이 PointNet++ 이다.
  

- PointNet++ : Deep Hierarchical Feature Learning on Point Sets in a Metric Space

  - Motivation  
    - PointNEt은 local structure를 capture하지 못하는 단점이 존재. 그냥 어떤 point자체의 metric space poinit에서 전체를 그냥 max pooling했다. 그래서 어떤 fine-grained 한 pattern을 캡쳐할 필요가 있었고, 그 complex scene에 대해서 generalizability하게 만들기 위해서 이런일을 수행할 필요가 있었다. 그래서 PointNet++는 hierarchical한 neural network를 구성해서 어떤 input point set을 partitioning을 통해 recursively 하게 합치는 일을 한다. 그래서 이걸 보면 PointNet++가 이렇게 partition들이 있을때 hierarchical하게 어떤 update를 하고 그다음 또 하나의 빨간색 어떤 feature를 만들어서 이걸 max pooling을 해가지고 최종적으로 어떤 feature를 얻어내는 일을 한다. 왜 이렇게 하냐면, 어떤 contextual scale을 키우기 위해서 이 context를 먼저 파악하고 이 context를 파악해서 그거를 hierarchical하게 구성을 하는것이다.
  - Handling Non-Uniform Sampling Density
    - 그러면 non-uniform한 sampling density를 커버하기 위해서 저 point cloud의 어떤 샘플들을 다 똑같은 receptive field로 합칠수는 없잖아. 그럴때 2가지 방법이 있는데 multi-scale grouping(MSG)가 있고, multiresolution grouping(MRG)방식이 있다.
    - multi-scal grouping
      - 서로 다른 스케일의 어떤 그룹핑 layer를 갖는것. 그래서 이쪽 채널은 더 큰 receptive field를 가지고 이쪽 두번쨰 그룹은 더 작은 점점 작아지는 멀티 스케일의 그룹핑방식이 있다. 그래서 서로 다른 scale의 feature가 concatenated되는것이다. 단점은 이제 비싸다는것. 그래서 local PointNet은 이제 어떤 centroid point에 대해서 scale neighborhood들을 이제 가져간다는 특징이 있다. 
    - multiresolution grouping
      - 반면에, 어떤 한 그룹은 서로다른 그룹 위치들을 가지도록, 즉 alternative approach를 사용한것이다. 그래서 이런 방식의 multiresolution grouping은 좀더 computationally하게 efficient하고, 더 scale이 커졌을 때에도 feature extraction에 대해 더 오래 걸리는것을 피할수 있는점이다. 그래서 각 점에 대해서 PointNet++에 사용한 것처럼 어떤 위치에 대해서 한 feature로 그룹핑이 되고 그 위치에 대해서 또 하나를 사용하는 좀더 넓은 resolution을 보는 feature와 이제 더 낮은 resolution을 보는 feature가 같이 concat된다는 점이다. 
      
  - PointNet++ Architecture
    - 그래서 이런걸 생각해 봤을때 기존 PointNet 구조에서 SetNet straction을 다음과 같이 진행한다. 먼저 Hierarchical PointSet Feature Running part에서는 sampling과 grouping을 동시에 진행한다. 그래서 이렇게 grouping된 feature들이 PointSet을 통해서 update가 되고, 다시 또 이 sampling과 grouping을 통해서 PointNet으로 다시 업데이트를 해서 최종적으로 어떤 abstraction된 대표적인 point feature들을 구한다. 이제 이걸 사용해서 segmentation을 수행하는것이다. 그래서 각각 segmentation label들을 다시 원래에 있던 point cloud로 propagation을 통해서 segmentation을 수행한다. 그래서 다시 interpolation을 통해서 unit PointNet을 계속 통과시킨 후에 최종적인 per-point scores들을 이제 구한다. 여기서 또하나의 특징은 ResNet과 같이 skip connection, skip link를 추가해서 좀더 end to end learning 이 쉽게  만들고 classification같은 경우는 이 전체를 다시 max-pooling을 통해서 PointNet Feature을 구하고 최종적으로 MLP를 fully Connected layer를 통과해서 class score를 구한다.
  
  - ScanNet labeling accuracy
    - 기존의 3D CNN이나, PointNet이나 score를 비교했을때 PointNet++가 scanNet dataset에서 좀더 높은 point cloud classification accuracy를 가진다. 그리고 multi scale grouping이나 multiresolution grouping을 했을때 scale grouping이 조금더 좋은 결과를 보임을 알수 있다. 그렇지만 multiresolution grouping이 좀더 computation을 efficient하게 하면서도 충분한 결과를 냄을 알수 있다. 그래서 Ours와 PointNet을 비교했을때 groud Trues가 훨씬 더, 그 PointNet++가 groud trues와 비교하면 비슷하게 ScanNet segmentation결과를 얻음을 알수있다.
  - Results
    - 이 구조를 MNIST에서 digit classification을 실험해 보았을때 Network in Network보다는 조금 안좋지만 PointNet보다는 PointNet++가 더 좋은 결과를 보인다.
    - ModelNet40 shape classification에서도 기존의 point cloud 기반 기법에 비해서 더 좋은 결과를 내고 voxel이나 image기반의 multiply image 기반의 기법보다도 PointNet++가 더 좋은 결과를 얻는다. 
    - 그리고 non-rigid shape classification같은 경우에도 part 기반으로 점점 합쳐나가는 Hierarchical한 구조를 사용하기 때문에 Hose나 Cat, 앉아있는 Horse의 경우에도 비슷한 위치들에 비슷한 segmentation을 수행해준다. 
  
  - Properties of Point Sets
    - 다시 정리해보면 PointSet에 대한 properties를 이제 탐구한 2가지 논문에 대해서 살펴보았다.
    - PointNet은 unordered point set에 대해서 어떻게 처리할지에 대한 탐구를 하고 local transformation에 대해서 invariant하기 위해서 t-net, transformation network를 사용한다. 다만 단점은 limited receptive filed를 가지고 있어서 어떤 classification을 정확하게 수행할수 없다는 단점이 있다.
    - PointNet++는 좀더 uneven한 point set에 대해서 robust하게 동작하기 위한 PointNet 구조를 제안한다. 그래서 manual한 interpolation이 필요하지만, 이렇게 PointNet++구조로 성능을 더 높였고 그 이후에 PointNet variance들이 아주 많이 나온다. 그래서 Point Set structure learning을 위한 탐구들이 많이 이루어지고 있고, unordered set을 handling하기 위한 또 다른 문제들도 정의하게 된다. 


- Point Transformer
  - 최근에 point transformer구조들이 많이 나오게 된다. 최근에는 convolution neural network를 넘어서 inductive bias가 더 줄어든 transformer구조를 많이 활용하기 시작했다. 기존의 publication을 보면 예전에는 2017년에 처음에는 적었지만 BERT나 RoBERTa 이런게 나오면서 점점 많아졌고 2020년에는 거의 100개와 가까운 GPT-3나 VIT, DIT 같은 수많은 transformer기반의 task를 푸는 논문들이 CWPR, ICCB, CCB, NeurlPs, ICML, ICLR에 publish되고 있다.
  - Key Terms을 사용하는게 BERT나 self-attention, transformer를 사용하는게 점점 더 나아지고 있다.
  - Swin transformer 구조가 CWPR2021년에 best paper를 받으면서 효과적임을 vision transformer 구조도 LLM에서 transformer가 처음 나왔지만 vision에서도 많이 활용됨을 알수 있다.

  - transformer 기관의 모델, VIT와 같은 모델이 다양한 task에서 CNN보다 더 좋은 결과를 내고 있다. 그래서 classification에서 imageNet을 푸는 문제에서 VIT가 가장 좋은 결과를 냈고 object detection이나 semantic segmentation, instance segmentation에서도 이런 transformer 구조가 가장 좋은 결과를 내고 있다.
  - 3D vision에서도 transformer기반의 구조가 점점 SOTA를 달성하고 있다. 기존의 KP-conv나 S3DIS에서 3D semantic segmentation에서 point transformer가 이걸 이겼고, 3D object detection에서도 transformer 기반의 방식들이 더 좋은 결과를 내고 있다.

  - Transformer란?
    - attention with positional encodings.
    - positional encoding을 포함하는 attention구조에서 key, query, value가 이렇게 encoding이 되고, key, query에 대해서는 matrix multiplication, scale, masking, 그리고 softMax를 통해서 최종적으로 어떤 attention score를 계산하고 value랑 곱해줌으로써 이제 scaled product attention을 구해낸다. 이걸 중첩시킨게 이제 multi-head attention이다.  그걸 이제 어떤 transformer라느 ㄴ구조로 이제 연동시킨게 최종적인 구조이다. 그래서 input이 넣었을때 input embedding이 만들어지고 positional encoding이 추가가 된다. 그것을 multi-head attention을 통해서 update가 된다.   그리고 normalization되고 그 feedforward를 통해서 input을 계산을 한다. 그리고 여기에서 또 다른 output embedding에 대해서 만들어져서 최종적으로 attention된 결과를 얻는게 처음 제안된 transformer 구조였다.
  
  - Point transformer
    - 위에것을 착안해서 어떤 point cloud에서도 활용할수 있는 point transformer구조가 제안이 된다.  
    - 어떤 P라는 set을 이제 위치 position과 feature를 가지고 있는 어떤 point cloud라고 가정을 하고 , P는 어떤 3D position, 그리고 F는 이제 C dimenstion의 어떤 feature이다. 로는 어떤 normalization을 나타낸다.(수식에서는 softmax를 나타냄), 감마는 attention weight를 위한 어떤 MLP이다. 이제 attention을 계산하기 위한 MLP,  사이는 query를 embedding하는 linear layer이고 피는 key를 embedding하는 linear layer,알파는 value 를 embedding하는 linear layer가 있다.
    - 그래서 value를 embedding하는 linear layer와 그 사이와 피가 각각 key, query를 이제 embedding하는 linear layer이다. 그리고 세타는 relative positional encoding하는 MLP이다.  그래서 어떤 델타 값은 상대적인 positional encoding을 계산하는 MLP를 통과시키는 값이 이제 뒤에 추가가 되고, 각각 key, query를 encoding하는 MLP들이 있고 이제 감마를 통해서 최대 최종적으로 attention weight를 계산한다. 그래서 이제 이 앞에 있는 term이 결과적으로 어떤 attention weight를 계산해주는 term이고, 뒤에는 attention 에 곱해지는 value인거다.  그래서 얼마나 weight를 줄지를 계산해주는 것이다. 그것을 각 point의 near neighborhood에 대해서 계산을 해주는것이다. 그래서 어떤 receptive field 범위를 가지고 k-nearest neighbor에 대해서 point feature를 update하는 것이다. 이전에 PointNet++ 계산생각해봐도 이렇게 Point들의 어떤 update들이 진행되었다. 그래서 최종적으로 F' 이제 계산이 된다.

  - Point transformer achieves the state-of-the-art(SOTA) in various 3D tasks.
    - point transformer는 ICB2021년에 논문받은 paper
    - shape classification에 대해서도 기존의 KPConv, PointConv,PointNet++, Set Transformer에 비교했을때 더 높은 성능을 달성.
    - point transformer가 기존의 다양한 3D task에서 소타성능을 당시에 달성을 한다.   
  
  - PointNet++ and Point Transformer vs Sparse convolution
    - point transformer가 막상 좋은것 같지만 단점도 있다. 어떤 K-nearest neighbor를 찾을때 익스펜시브한 단점이 있다. 장점은 fine level의 geometry를 확인할수 있다는것.
    - 그리고 PointNet++ 와 Point transformer에 대한 비교는 PointNet++는 어떤 nearest neighbor의 어떤 feature들을 embedding하는데 그냥 단순히 어떤 K-nearest neighbor만 embedding하는 특징이 있지만 Point Transformer는 relative positional encoding의 차이도 계산을 해주고, key/query에 대한 attention weight를 계산을 해서 value에 곱해주는, 조금더 attention과 비슷한 구조를 사용하는게 차이점이다. 그래서 최종적으로 업데이트할때는 attention처럼 계산을 해준다. 그래서 플러스의 동그라미 쳐진게 permutation-invariant operator이다. (max pooling이나 average pooling을 만들어가는것.) 그래서 PointNet++도 이런구조로 최종적인 classification 1D vector를 구해주고 MLP를 통과시킨 것 같이 마찬가지로, point transformer도 전체적으로 묶인 걸로 1D vector를 만들어준다.
    - 반면에 이후에 배울 Sparse convolution과 같은 구조는 point들을 어떤 위치로 Quantization한다.  그다음 그 위치에 대해서 convolution을 돌릴수 있게 제한을 해준다. 장점은 efficient한 neighbor search가 가능하다는 점. 단점은 Quantization 을 통해서 약간 point들의 위치가 데비에이션 될수 있고 kernel weight가 fixed된다는 단점이 있다. 그래서 Saprse convolution이 좋은지 그냥 point cloud그대로 사용하는 pointNet이나 Point transformer기반의 구조가 좋은지는 아직도 해결중이다. 
  
  - Comparison of 3D Methods
    - Sparse convolution과 3D point transformer와의 비교는 Farthest Point sampling기법(FPS)을 통해서 Expensive하고 Heuristic grouping해야된다는 단점이 있다. 그래서 relative point 위치 인코딩을 하고 이제 이 feature를 찾지만 반지름 값을 Ball Query라는 알고리즘을 사용하는데, 반지름 값을 찾아줘야 한다는 단점이 있다.
    - Sparse convolution의 장점은 down sampling이나 이런거에 있어서 그냥 pooling하면 되기 떄문에 efficient하다. 하지만 단점으로는 Quantization artifacts가 이제 너무 커진다는 단점이 있다. 포인트들을 하나로 어떤 voxel grid로 모아야 되기 때문에 이런게 단점이다. 이런 부분에 대해서 contribution한 issue가 있따는 생각을 가지고 Quantization하는 Sparse convolution에 대해서 제대로 배우지 않고 여기를 했었는데 다음 point cloud registration에 배워보자.

- Summary of Point Transformer
  - transformer구조가 3D vision에서도 잘 활용됨을 보았다.  
  - 단점일수 있지만 PointNEt++가 제안했던 heuristics를 여전히 사용하고 있다.
    - 그래서 K-Nearest Neighbor search를 통해서 얼마나 많은 point들의 근처 attention을 개선해줄지 이미 하고 있다.
    - Ball Query를 통해서 receptive field 영역, nearest neighbor를 잡아주는 영역의 반지름을 어느 정도로 잡을지 찾아준다.
  - 3D vision에서 transformer가 어떻게 개발되어야 하는지의 방향성은 point transformer만큼 효과적이면서도 sparse convolution만큼 efficient한 구조가 이제 앞으로 더 연구되어야 할 여지가 많이 있다. 

## Point Cloud Registration
: 포인트 클라우드의 정합을 통해서 큰 맵을 만들수있는데 이제 3D reconstruction을 할수 있는 point cloud registration임.

### 3D Surface registration
#### Basic study on the convex optimization
- Gauss-Newton algorithm
: 기본적으로 convex optimization 알고리즘의 대표적인 방식
  - iteratively 어떤 variables 값을 찾는 방식.
  - sum of squares를 minimizes함으로써 가장 최적의 어떤 핏한 모델을 찾는것이 목표이다.
  - 처음에 initial 파라미터를 기준으로 이런 수식을 사용한다.
  - 우리가 알고있는 gradient decent에서 어떤 1차원 gradient decent방식이기도 하다. 어떤 2차원 수가 있을떄 이걸 미분하면 가장 기울기가 낮은 함수가 되는데 (기울기가 0에 가까운쪽을 선택) 이떄 최적이 된다는 점을 생가하면서 넘어가면 된다.

  - Chain Rule
    - 딥러닝이라 이런부분에 많이 사용되는 properties인 chain rule이 있다.
    - 어떤 중첩된 합성함수가 있다고 했을때 이걸 F(x)라고 하고 이제 f(g(x))라고 했을때, F'(x)의 미분한 값은 이제 Y를 g(x)라고 하고 Z를 f(y)라고 했을때 이제 chain rule이 적용될수 있다.
    - 이런 chain rule을 다시 rewriting하면 저런 형태로 적을수 있고 이제 이것을 각각 이렇게 f'(y)g'(x) 라고 적어서 최종적인 형태로 표현가능하다. 이것을 chain rule이라고 하고, DX를 결국 dz/dy, dy/dx 이렇게 어떤 합성함수의 곱형태로 표현할수 있는게 chain rule이다

  - Pairwise Registration:RGBD Image Alignment
    -  Energy function minimize intensity를 배워보자.
    - enery function은 어떤 RGBD 이미지 사이 2장의 RGBD 이미지 사이에서 그 픽셀 intensity의 inconsistency를 minimize하는게 목적이다. 그래서 I(x')과 Ij(x)값에 대해서 X를 minimize하는것이다. 그래서 최종 minimize하는 값을 얻는게 목표이다.
    - 즉, 여기서 보면 어떤 X라는 점이 있을때 이제 2장의 RGBD이미지가 있다고 했을때 이제 3차원 점을 알수있다. 거기에 카메라 포즈의 차이 RT를 바꾼 다음에 다시 back-projection을 시키면 이제 X'이라는 점이 된다. 즉 transformation과 back-projection, projection을 각각 알게 되는것이 우리가 아는 RGBD image Alignment이다. 

  - Pairwise Registration : Enery Minimization
    - 그래서 이런 energy function을 minimize하는데 이제 그 projection된 어떤 값과 back-projection된 이미지의 차이를 구하는것이 값이 된다. 결국 task는 어떤 차이의 그 jacobian matrix의 patrial derivation의 차이를 계산하는것이 task이다. 그래서 결국에는 Gauss-Netwon method로 이제 convex optimization수식을 통해서 최소화하는 어떤 포즈를 찾는게 목표인것이다.
  - Pairwise Registration: Enery Function
    - 그래서 이런 에너지 function을 minimize한다고 했을때 수식을 이렇게 적을수 있다. 어떤 X와 어떤 D의 j(x)에 대해서 이제 H라는 함수는 back-projection이다.  어떤 점을 다시 3차원 점으로 옮기는 것을 back-projection이라고 한다. 즉, 여기 어떤 값이 RT를 통해서 또다른 카메라 포즈에 대한 어떤 수식으로 변경이 되는것이다.
    - 마지막으로 guv가 projection시키는 것이다. 즉, 또 다른 이미지의 어떤 평면으로 projection시키는 과정이다. 그래서 어떤 두장 사이에 이제 매칭되는 부분을 레지스터하는것. 즉, 매칭이나 레지스트레이션이나 코리스폰더스를 찾는다는 것에 대한 똑같은 말이다. 그렇게 레지스터하는 3가지 과정으로 수식을 x'이라는 값을 이 x로 표현할수가 있다.
  - Ingredients: Projection and Back-projection
    - 이 과정은 이전에 classical computer vision이나 multi visionality시간에서도 배웠던 것처럼 projection과 back-projection은 이런 homogeneous coordinate로 표현할수 있다.
    - 그래서 이런 인트렌즈 camera값과 이렇게 결구 전개된 값, 그리고 u,v 이렇게 표현할수 있다.
    - x,y에 대해서 각각 카메라 center와 폴랭스, 그리고 어떤 평면에서의 좌표값으로 x,y를 표현할수가 있다. 최종적으로 이값에서 x,y,z값을 이렇게 표현할수가 있게 되는것이다. z는 depth이고, x,y는 인트렌즈 카메라 파라미터와 어떤 이미지에서의 좌표로 표현을 해야 될수 있는것이다.

  - Ingredients: Linearized Transformation
    - 이걸 많이 사용한다. 호모그래피를 사용한 트랜스포매이션은 3x3이상 매트릭스 폼으로 써야되느네 거기에는 이제 0값도 있고 matrix를 다 가지고 있는것은 너무 많은 메모리를 차지할수도있음. 그래서 조금더 쉬운 form이 필요. 그래서 small motion update를 위해 linearize된 transformation matrix를 사용하게 됨. 

  - RGBD Image Alignment
    - 에너지 function을 optimize하는것.  
    - 이거는 back projection하는 과정을 나타낸것.
    - 어떤 포인트 T를 다시 3차원 좌표 X로 옮기는 과정. 두번째 과정은 transformation 과정인데. 이전에 있던 방식이 아니라 small transformatino을 표현하기 위해서 이렇게 linearize된 작은 transformation을 이렇게 표현하게 된다. 
    - g는 projection 3차원 좌표를 다시 2차원 좌표로 옮기는 과정을 표현.
  - Energy function
    - chain rule로 표현가능.
  - Partial Derivation
    - 이미지의 한 방향에 대해서 구해줄수 있다.
    - x,y방향의 gradient를 구하는 convolution filter를 통해서 map을 얻어낼수 있다.
    - 최종적으로 RGB fixel과 depth image에 대한 alienment를 수식으로 표현가능.

  - Video Representation
    - RGBD Odometry, Point cloud registration에서 이런 방식을 다 사용가능.
  
  - Surface Registration Summary
    - Colored Point Cloud Registration Revisited, ICCV.2017 논문에서 전반적인거 알수있다.
    - Initialization은 매우 중요.
    - Initial T가 surface가 이렇게 있다면 어떻게 매칭을 할지 처음 구해주는것이다. 점점 움직이면서 맞춰져 나가는 과정인데 조금씩 움직여 나가면서 맞춘다.
    - energy function을 정의하는게 중요하다. 
      - chain rule을 사용한 patrial derivates를 고려해야함.
      - partial derivate는 differentiable하기때문에 chain rule로 update해야한다.
    - 어떻게 neural network가 optimized 될수 있는지 배워보는 과정이었다. 
    - Gauss-Newton을 적용하는것뿐 아니라 더 나은 optimizer를 사용할수도 있다.이건 가장 기본적인 convex optimization기법이다
  
  - Extension : Colored Point Cloud Registration
    - 이전에는 RGBD이미지에서 이미지상의 어떤 대응된 지점을 찾기 위한 registration기법을 배웠다면,  colored point cloud 에 대한 registration을 배울수 잇다.
    - point cloud도 intensity alignment에 대한 수식은 동일.
      - projection 그리고 이동하고 back projection 된 point cloud와 또 다른 반대편 이미지 반대편 point cloud에서 있던 대응된 point cloud를 봤을때 이걸 최소화하는것이다.
    - depth alignment도 동일하게 진행된다.
    - 2개의 alignment를 각각 combine해서 optimize를 하는것이다.
    - 여기서의 trick은 image plane과 같이 point cloud를 parameterize하는것. 또한, first order taylor expansion을 통해서 C(p)와 depth의 어떤 연계된 덧셈으로 연결된 수식으로 표현 가능. 그래서 plane icp energy function을 optimize하는것과 equivalent한것이다.

    - initial alignment를 준다음에 point to plane ICP를 해서 point alignment하는것을 계속 수행해서 최종적으로 point cloud registration을 수행하는 것이다.
  
  - Point Set Registration
    - 이것이 어떻게 진행되느냐? 처음에 point set이 2개가 있다고 하자. 처음에 iterative closest point알고리즘을 돌리기 위해 처음으로 initial registration을 구한다.  그다음에 대응값을 찾고, 이렇게 P와 이동된 Tq에 대해서 optimize를 수행한다. 즉, 점점 가까워지는 과정을 수행한다.
    - stage1에는 coarse alignment를 한다. 이때 RANSAC or another sampling방식을 사용한다.
    - stage2에는 local refinement를 ICP알고리즘을 통해서 수행한다.
    - 그래서 이럴떄 단점으로는 expensive하고 inelegant하단점이다. nearest-geighbor queries를 inner loop하여 계쏙 한 포인트에 대해서 뭐가 제일 가까운지 서치를 해가지고 optimize해야 되고, two stage로 direct alignment에 비해 2-stage로 해야한다는 단점이 있다.

  - Fast global Registration
    - 위 문제점을 해결하기 위한것.
    - P와 Tq가 있을때 이거를 한번에 optimize하는 로우 라는 값을 도와주고 optimize를 하는것이다. 로우 라는 함수에 어떤 2차함수와 비슷한 폼에 뮤+x제곱 분의 x 제곱 이라는 수식을 통해서 optimize를 수행하는것. 저런 optimization function을 쓰면 더 빠르게 global registration이 수행된다는것이다.
  
  - Optimization
    - 원래 수식이 있었을떄 l 이라는 함수를 introduce해서 추가적으로 optimize하는것이다. 그다음 뒤에 regularization term을 둬서 L이 어느정도 이상으로 보상하지 못하도록  이런 값으로 추가적인 optimize를 수행해준다. 그래서 너무 멀어진 값에 대해서는 optimize를 하지 않고 어떤 비슷한 패턴에 대해서 optimize가 될수 있도록 optimize가 되는것이다. optimize가 될수록 어떤 한부분이 더 뾰족하게 되고 덜될수록 밑으로 내려고게 된다. 최종적으로 transformation function이 학습이 되는것이다.

  - Matching Two Point Clouds
    - 2개의 point cloud set을 optimize할때 GoICP알고리즘 돌리거나 PCL알고리즘을 돌린것보다 Fast Global Registration이 ECV 2016년 논문에 보면 더 optimize가 잘되었음을 알수 있다.


  - Multiway Registration
    - Open3D의 multiway registration을 보면 다양한 example과 직접 실습해볼수 있게 되어있다. 
    - point cloud를 load하고 geometry를 그려보고, pose graph를 만든 다음에 이것을 점점 optimize를 수행하는것을 할수 있다. 그래서 point cloud registration된 결과를 open3D visualization tool을 통해서 geometry를 visualize해볼수 있다.

  - Generalization for n-frames
    - 2개의 point cloud set이 아니라 N개의 frame에 대한 registrarion으로 할수있다.
    - pairwase registration은 edge에 대해서 이렇게 동작할수 있음.
    - pose graph의 diagram을 optimize하는것으로도 활용가능.
      - N개의 frame을 optimize할수 있는데 저 밑에 있는 수식과 같이 어떤관계, 인접한 pose와 관계를 알수 있고, 멀리 떨어진 view에 대해서 optimize를 추가적으로 수행한 term을 둘수 있다.
    
  - Challenge
    - global registration을 잘하는것이 어렵다는점이다. 
    - global하게 cols하게 alignment를 하고 fine하게 리파인먼트를 해주는 2가지 과정을 수행하는데 그게 아니라, 하나의 알고리즘으로 글로벌 레지스트레이션을 하는것은 쉬운일이 아니다. 그런다고 했을때 n-frame에 대한 일반화로 이런 수식으로 퍼뮬레이션을 수행할수 있다. 그냥 global registration을 했을떄 망가지는 결과를 얻을수도 있다. 그래서 이걸 잘 optimize하는게 중요하다. 그러기 위해서 조금더 중요도가 있는 edge에 가중치를 둬서 optimize를 수행하는 term을 뒤에 2개를 추가하게 되는것이다. 그래서 2번째 term을 optimize를 진행한 term이고 뒤에는 레귤러라이제이션 term이다. 
  - 이것에 대한 간단한 데모가 partment에 대한 씬을 리컨스트럭션하는 과정을 볼수 있다. 
    - 빨간색 점에서 책상쪽을 보았을떄에 대한 리컨스트럭션 뷰나, 노란색점이나 저런 파란색점, 등 잘 리컨스트럭션을 해낸다.
    - bedroom에서 얻어진 point cloud의 여러개 veiw를 registration했을떄에도 깔끔하고 정확하게 리컨스트럭션을 해낸것을 볼수 있다.

    - 또한 이 레지스트레이션 알고리즘을 어떤 회의실과 같은 경우에 대해서도 굉장히 깔끔하고 정확하게 global registration을 성공시킨것을 볼수 있다.  


## Sparse Convolution
  - 우리가 알고있는 convolution은 sliding window를 이미지 전체에 대해서 진행한다. 
  - 그러나 sparse convolution은 실제 어떤 image or point cloud에서 occupy된 영역들에 대해서 따라가면서 convolution을 진행하게 된다.  이것을 다시 한번 보게 되면 특정위치에 대해서 오큐파잉된 영역에 대해서만 convolution을 수행하게 되는것이다. 그래서 convolution 수행이후에는 인접한 영역에 대해서만 feature가 뽑히게 된다. 

- Image Classification
  - 기존의 image classification은 input image가 있을때 feature learning하는 부분이 있다. 그래서 convolution, RELU, pooling, convolution, pooling 이런 과정을 거쳐 최종적으로 어떤 feature를 얻게 되고 이걸 flatten시킨 다음에 fully connected하고 softmax를 통해서 최종적인 classification을 수행. 
  - 그럴떄 우리가 사용하는 convolution operation은 이런 kernel이 있고 이미지 위를 해당 kernel이 sliding window 패션으로 쭉 순환해서 값을 얻어내는 형태로 2D convolution을 수행하게 된다. 
  - 그래서 그렇게 해서 얻어진 convolution CNN을 통해서 얻어진 feature map을 이렇게 또 pyramid pooling module같은 것을 사용해서 최종적인 semantic segmentation map을 얻어낼수도 있다. 즉, pixel level의 classification을 진행할수 있다. 그래서 이런 형태로 feature map의 size를 유지해서 최종적인 upsampling을 통해서 최종적인 final prediction을 얻어내는 semantic segmentation을 수행도 가능하다. 또는 deconvolution network를 통해서 max pooling을 통해 줄어들어진 어떤 레이턴트 vector를 다시 upsampling과 upsampling convolution, deconvolution을 통해서 segmentation map을 얻어내는 deconvolution deconvnet도 있다.

- motivation
  - large scaled scene에 대해서 convolution을 수행하려고 한다면 어떻게 해야할까?
  - 기존의 local region analysis을 통한 sliding window방법은 이제 아무것도 없는 영역을 통과할때도 비효율적인 연산이 많아진다. 그리고 receptive field도 제한된다는 단점이 있다.
  - 기존 pointNet같은 방법은 receptive file도 제한이 된다.
  - 이런것에 착안하여 최근에 제안되는 방법이 sparse convolution을 활용한 방식이다.

- sparse convolution
  - sparse data를 analyze하고 싶다면? 
    - 기존의 pointNet은 각 point들에 대해서 XYZ를 통과시킨 MLP를 사용해서 symmentric function을 통해 global feature를 얻어낸다.
    - symmentric function은 max pooling이나 average pool 또는 다양한 풀링방법들이 있다. 그래서 보통은 attentional pooling을 할수도 있지만 max pooling이 가장 잘되었다고 pointNet에서는 이야기 했었음.
    - 하지만 이런 방식은 주변의 영역들을 볼수 없어서 pointNet++에서는 이렇게 주변 방법을 하이로피컬하게 엮어내는 방식도 사용을 했었음.
  
- PointNet - Question
  - pointNet에서 몇가지 리미테이션에 대한 얘기가 생길수 있음.
  - larger 3D scene에 대한 extendable할수있나?
  - point cloud가 많으면 하나가 share하는 MLP들이 해야하는일이 더 많아지기 떄문.
  - 또한 spartial relationship을 조금더 efficient하게 활용하는 방법을 pipeline으로 디자인을 했어야 했음.
  - fully convolutional한 방식을 3D space에서 어떻게 구현할지도 이슈임.
  - convolution과 deconvolution을 같이 사용하는 unit shape같은 모양의 downsampling과 upsampling을 어떻게 point cloud set에서 구현할수 있을지도 핵심이다.
  - 그러나 pointNet은 그런 up/down sampling을 고려할수가 없다. 그냥 하나의 point에 대해서 feature update를 하기 떄문에.
  - rotation과 translation에 대해서 invariance하게 만들기 위해서는 기존의 pointNet은 T-Net 을 사용해서 3x3 transform로 어떤 point cloud set을 다시 pose invarient하게 바꿔주었다. 근데 이것이 바로 답일까? 기존의 spartial transformer network같은것도 dense grid에서는 활용할수 있는데 그런것을 더 잘 활용하는 방법에 대한 고민이 있었음.

  - 기존의 PointNet은 unordered point set을 다루기 위해서 local transformation에 대한 envariant한 T-Net을 활용하고, receptive field가 제한되어 있었다.
  - pointNet++에서 좀더 robust하고 좀더 불균일한 pointNet을 다루기 위한 manual interpolation, 그 어떤 주변의 값들을 보고 하이로키컬하게 구성하는것을 했다. 그렇지만 이건 manual enterpolation이 필요했다.
  - 그래서 다양한 PointNet variants가 나왔지만 우리가 더 이거에서 낮게할수있는지가 이제 sparse convolution을 활용한 feature extraction방법의 motivation이다. 

  - Revisiting Volumetric Representation
    - 이제 volumetric representation을 다시한번생각해보자.
    - multiple한 voxel을 multiple view points에 대해서 어떻게 활용할지 한번보자.
    - 기존의 3D convolution을 활용한 network volumetric representation에서 3D CNN은 다음과 같이 통과된다.
      - MLP convolution을 통과해서 volumetric map을 계속만들어서 최종적으로 어떤 partial object에 대한 prediction을 하거나 전체 object에 대한 prediction을 해서 3D object segmentation을 수행하기도 한다. 그래서 이런 multi view 이미지에 대해서 각각 3D CNN을 통과한다. 그리고 orientation pooling을 통해서 다시 한번 3D CNN을 통과해서 class prediction을 해서 어떤 multi view의 이미지가 어떤 클래스를 나타내는지 추정한다.
      - 이방법의 단점은 network가 충분히 deep하지 못했다는단점.충분한 layer수가 부족함. 그리고 up/down sampling을 하지 않았다는점. 그리고 rotation invarient feature를 학습하기에 충분하지 못했다는 단점도 있음.
  - Key Observation
    - volumetric convolution을 활용하자는게 key이다. 이걸로 network를 구성하게 되면 up/down sampling도 자유롭고,  receptive field도 convolution을 manner로 자연스럽게 구성할수 있다. 그러나 여기서의 issue는 real-world의 point set은 매우 sparse하다. 전체의 어떤 volume-set에서 10%정도밖에 어큐파인이 되어있지 않으니깐. 그리고 dense voxel grid를 고려하지 못한다는 점도 있다. 그래서 volumetric convolution을 그대로 활용한것이 아니라 occupied된 voxel에 대해서만 convolution을 수행하는것을 진행해보자. 그것을 바로 sparse convolution이라고 한다.

  - U-Shape Network Architecture
    - 그렇게 활용한 convolution을 manner로 fully convolution방식으로 network를 구성하게 되면 U-shape Network architecture를 구성할수 있다. 그래서 spartial reasoning을 위한 ability를 증가시킨다는 점이다. 
    - point cloud가 sparse tensor로 바뀐 input이 들어왔을때 sparse convolution을 통과해서 Low-level geometry feature를 구한다. 그것에 대해서 ResNet의 바틀렉 구조와 비슷하게 residual connection을 넣고 pooling을 하고 다시한번 sparse conv, 그리고 high-level feature를 얻은 다음에 다시 transposed conv를 통해서 upsampling된 high-level feature를 구한다. 이 upsampling된 high-level feature와 low-level feature를 concatenation을 해서 concatenated feature를 구하고 여기에 다시한번 sparse convolution을 통해서 output score function을 구한다. 그래서 core elements는 sparse convolution과 transposed convolution, 그리고 pooling, upsampling같은 2D에서와 똑같은 구조를 point cloud에 대해서도 적용할수 있다. 당연히 convolution operator뒤에는 RELU와같은 non-linear function이 주어진다. residual sparse convolution구조는 input feature가 있을때 sparse conv을 통과하고 input feature와 더해져서 output feature를 얻어내는 형태로 구동이 된다.

  - Sparse Convolution
    - sparse data를 어떻게 analyze할지, 이제 mesh와 point cloud에 서 sparse convolution을 어떻게 진행하는지에 대한 블로그가 있다. open3D로 어떻게 point cloud or mesh를 global grid로 sparse global grid로 옮기는지에 대한 코드와 설명이 있다.
    - 그래서 이런 sparse data가 있다고했을때 기존 pointNet과는 달리 이제 sparse convolution은 다음과 같이 진행이 된다.
  - Generalized Sparse Convolution
    - 일반적으로는 coordinate Quantization을 한다. 이렇게 partialy affine된 pointset을 근처에 인접한 global grid로 색칠을 하게 된다.즉, point cloud가 접하고 있는 모든 영역에 대해서 색칠을 해주게 되면 국채grid가 나오게 된다. 그다음으로는 convolution을 수행한다. 그러면서 sliding window를 진행하고,  conv feature에 대해서 input feature에 대해서 indicator matrix를 이제 곱하고 weight (=convolution filter)도 곱한다. 그래서 bias를 구하는 이런 convolution을 수행하게 된다. 다만 이런 convolution을 수행하는데 이제 이걸 곱하게 되면 최종적으로 이밑에 있는것과 같은 form이 나오고 뒤에 bias를 더해서 최종적으로 summation을 해가지고 하나의 feature point를 만드는것이다. 그리고 마지막으로 non-linear function을 통과해서 이 point feature에 대해서 non-linearity를 추가해주게 된다. 그래서 이 전반적인 과정은 일반적인 convolution과 아예 동일함을 알수 있다. 다만 occupied된 영역에 대해서 sparse하게 operator를 진행해준다는게 다른점이다. 
    - Transposed Convolution
      - Transposed Convolution을 다시 짚고 넘어가면 기존의 pooling방식은 어떤 위치에 대해서 하나의 pooling을 한다. 그리고 switch variable이 어떻게 스위칭되는지에 대한 값을 가지고 있다. 반면에 unpooling은 그 스위칭된 위치를 가지고 input에 대해서 다시 그 위치로 unpooling을 시키는 것이다. 또한 convolution역시 어떤 feature map이 있을때 feature map의 특정한 영역에 대해서 이렇게 convolution kernel을 곱하고 summation을 해서 convolution을 수행해 준다. 반면에 transposed convolution은 어떤 weight가 있다고 했을때 하나의 위치에 대해서 다 곱해져가지고 반대로 convolution을 수행해주는 형태이다.
      - 그래서 convolution mask는 transposed될수 있고 weighted가 될수 있고 attached될수 있다. 이러한 성질을 활용해서 feature map을 이렇게 convolution을 수행해서 feature map update를 할수 있는것이다.
    - Efficiency
      - 그럼 이런 sparse convolution의 efficiency를 얘기해봤을때 얼마나 효과적이냐라고 하면, Occupying된 위치에 대해서만 convolution을 수행하기 떄문에 hash structure를 사용할수 있다. 그래서 Hash table을 사용한 구현을 할수 있다. 기존의 input data가 이렇게 XYZ 위치에 대해서 Occupying되어 있다고 했을떄 Quantized도니 데이터는 이렇게 Occupying된 위치가 Index로 구성이 될수 있다. Quantized된 Coordinate을 Hash Key로 사용하고 Occupancy를 그 key에 대해서 Occupying이 되어있는지 안되어있는지를 사용한 Hash Table형태로 이제 convolution, Operation을 수행할수 있다. 그래서 Hash Structure는 Fast inferenceing을 위해서 아주 효과적인 작동을 한다. 예를 들면 Key값만 있으면 되기 떄문에 Time Complexity가 이론적으로 O(1)이다. 그래서 Occupying이 안되어 있는 것은 No라고 나오고, 되어있는것은 Yes라고 나와서 Yes인 위치에 대해서 adjacent한 voxel의 값을 이제 kernel을 곱해서 update를 할수 있는것이다. 

    - Revisiting Volumetric Representation
      - Volumetric Representation을 다시한번 짚고 넘어가면 우리가 PointNet Variants, MLP를 사용하는 경우는 Large Scene에서 sliding window로 모두 구해야 된다. 하지만 Sparse Convolution은 쉽게 할수있다.
      - Spartial Relationship을 인코딩할때는 PointNet variants는 Handcrafted Gouping을 통해서 Multi-Sclae Grouping이나 multi-resolution Groupin을 했었다. 그러나 sparse convolution은 hashing을 통해서 이런 spartial relationship을 인코딩할수 있다. fuuly convolutional을 하는가 보면, 즉 GPU operator 전에 대해서 아주 friendly하게 구현할수 있냐고 했을떄 PointNet variants는 MLP로 되기 떄문에 그렇지 않다. 하지만 sparse convolution은 fully convolution을 하게 network를 구성할수 있다. 그리고 U-shape network같이 인코딩, 디코딩 구조를 활용할수 있느냐고 했을때 pointnet variants는 implicit하게만 구성할수 있다. 그것의 어떤 그룹핑을 통해서 구현할수 있다. 하지만 sparse convolution은 deconvolution도 구현할수 있기 때문에 쉽게 U-Net shape을 구현할수 있다. rotation과 translation에 대해서 invariant하게 feed forwarding을 할수 있는냐 했을때 pointNet variants는 T-net, transformation net을 활용해서 그렇게 했다. 그렇지만 sparse convolution 역시 deep network를 통해서 그것을 자연스럽게 구현할수 있다.
  
  - fully convolutional geometric features
    - 이방법을 처음 제안해가지고 활용한 논문이 밑에 있는 fully convolutional geometric fetures, ICCV,2019이다. 
    - 이렇게 geometric feature를 구한다음에 match를 구했을떄 2개의 point cloud에 대해서 match를 구했을때 더 정확한 match들을 구할수 있다. FCGF의 전체적인 아키텍처는 다음과 같다.
    - 먼저 3D conv를 한번 통과시키고, 3D conv에 이게 다 sparse convolution이다. sparse convolution을 각각 residual block과 3D conv를 통과시키고 3D Transposed Convolution을 통과시킨 후에 최종적으로 3D conv를 통해서 geometric feature를 구한다. 여기서 이해하고 넘어가야할 부분은 input이 단순한 voxel grid가 아니라 Quantized된 sparse voxel grid라는 것이다. point cloud가 sparse하게 된 voxel grid라는 점이다!! 그래서 효율적이로 효과적인 operation인 sparse convolution을 엔진인 민코프스키 엔진을 활용했다는점이 이논문의 또하나의 contribution이다.
    - 그래서 이 논문에서는 FCGV에서는 U-Sahpe network Architecture를 사용하고, 이렇게 sparse reasoning을 하는 방식을 본다. Low-level feature를 구하고, upsampling된 high-level feature를 구해서 이걸 concatenation해서 하나의 블럭을 구성하는 형태이다.
    - 그래서 전체적인 architecture는 아까 앞에서 보셨다시피 u-shape network의 architecture를 가지고, 3D conv로 구성되어 있지만 모두다 이제 3D 에서 Sparse convolution으로 되어 있다. Transposed Convolution으로 추가적으로 되어 있다. 그리고 skip connection을 활용한 U-shape network architecture를 활용하고 더 나은 performance를 위해서 Residual Block을 사용한다.  그리고 batch normalization도 이제 fully convolution구조이기 떄문에 사용할수 있다. 그래서 이와같은 방식으로 u-shape network architecture로 point cloud에 대한 geometric feature를 구현한다.
  
  - performance
    - 기존의 handcraft 기반의 feature extraction방법들이 있었고,  그래서 histogram이나 surface normal에 대한 histogram voting을 통해서 하는 방식이 있었다. pointNet이나 CGF같은 어떤 feature를 compresion하거나  우리가 잘 아는 MLP를 사용한 pointNet방식이 있었다. TSDF volume을 활용해서 Vocal Lumetric feature를 얻는 3D match나 perfect match와 같은 방식도 비교군에 넣었다. 그리고 pointNet variant들, FoldNet, PPFNet, DirectReg, Capsulate와 같은 Different Orientation이나 이런 인코딩하고 학습하는 pointNet Variants들이 있다.  반면에 여기서 말하는 Ours는 FCGF이다.Fully Convolution하고  Large Receptive field를 가질수 있는 network이다. 이것을 평가한다고 했을때 빨간색이 가장 빠르고 reliable하다. x축은 1초에 처리할수 있는 number of feature를 나타냄. y축은 feature match recall. 그러니까 얼마나 매치했다고 판단했는지에 대한 값이다. 기조방식들은 이렇게 perfect match는 feature match recall score는 높았으나 아주 오래 걸린단점이 있었다. 또한 handcraft기반의 방법인 FPFH는 아주 빠르긴했지만 match가 조금밖에 나오지 않았다. 그래서 이런 Far-append optimality가 이제 스피드와 accuracy에 대해서 trade-off가 있었는데 FCGF가 이렇게 호가 높이는 결과를 얻었다.
  
    - geometric feature를 T-SNE를 사용한 Colored로 visualize해봤을때 kitchen에 대해서도 비슷한 영역에 대해서 비슷한 색깔로 잘 매칭이 되고, 호텔도 잘 매칭이 됨. study room도 같은 의자끼리 잘 매칭이 됨. 또한 길거리 씬에 대해서도 키티씨니안 이런 차량 라이더나 레이더로 얻어낸 point cloud scene에 대해서도 정성적으로 잘 된다는걸 확인할수 있다.

  - Summary of FCGF
    - 3D 에서의 첫 fully convolutional 구조를 제안함.
    - 3D point cloud에 대해서 accurate, efficient한 결과를 얻음.
    - 플립 convolution하기 떄문에 U-shape Net과 residual network를 그 자체로 그냥 2D에서 가져와서 구현해낼수 있게 되었다.
    - spartial relationship을 이제 utilize해서 feature 인코딩할수 있게 되었다.
    - 좀더 large filed of view를 구현해낼수 있음.
    - pointNet varient보다 더 manual한 configuration이 좀더 적어졌다는 장점이 잇다. 그래서 hashing과 quantization을 위한 freprocessing이 필요 없다는점이 있고,  TSDF volume이나 grouping을 이제 필요로 하지 않는다는 점이다. 그래서 이런걸 project에서 쉽게 활용할수 있고 코드도 잘 릴레이즈 되어있다.
  
  - deep global registration
    - 이후에 나온것으로 CVPR2020년 oral paper가 있다.
    - 이전의 fast global registraion이나 colored point cloud registration을 update한 논문에 나온 방식을 fully convolutional하게 대체하고 이제 거기에서 추가적으로 6D convolutional update를 통해서 조금더 point cloud match를 잘 뽑았다는게 contribution이다.
    - 6-dimensional convolutional network는 inlier likelihood prediction과 같은 역할을 한다. 이전에 multivisiomat에서 NCNET을 배웠다. 그 neightborhood 컨센서스 network와 스팟스 네이버드 컨센서느 네트워크를 배웠는데, 거기에서와 마찬가지로 6D convolution을 통해서 point cloud set의 match를 update한것이다. 그래서 이 network는 U-net structure를 사용하고, 그리고 residual block을 strided convolution을 통해서 구성이 된다.
    - 전체적인 아키텍처는, 이앞에 어떤 3D-3D coord-input이 있다. FCGF와 모양이 비슷. 그래서 6D convolution으로 이 2개의 어떤 코릴레이션관계를 업데이트하여 다시 6D transformed convolution으로 다시 upsampling을 한다. 최종적으로 인라이어 로짓을 구해서 파이널 prediction을 한다. 인라이어 프리시션을 했을때 이렇게 조금더 잘 나올수 있다는 점이다.
  
  - results on 3DMatch benchmarks
    - 3D match benchmark에서 기존의 deep global registration, DGR논문이 기존의 RANSAC이나 fast-global registration or GO-ICP(Iterative Closest Points알고리즘) 이나 근야 ICP, pointNet이나 DCP보다 트랜슬레이셔널 에러나 rotation error에서 조금더 정밀한 카메라 포즈 시스템에서 성능을 낸다.
    - 이 결과는 어떤 point cloud의 2개의 set을 이렇게 매칭했을때 registration했을때 조금더 정밀한 결과를 정상적으로도 보인다.
  
  - Comparison  
    - DGR의 정성적, 정량적 결과이다.
    - 여기서 말하는 Recall이나 translational error(TE), rotational error(RE)도 더 좋게 나오고, 타임도 꽤 좋게 나온다.
    - FGR에 비해서 성능도 향상됨.시간은 조금더 오래걸리지만 GPU를 활용할수 있기 때문에 GPU 발전에 따라서 더 발전가능있음. 그리고 이 아키텍처는 마찬가지로 이전 FCGF와 같이 sparse convolution을 활용.그래서 RANSAC과 FGR을 그대로, 그것보다도 훨씬 더 좋은 성능을 낸다. 기존의 pointNet 기반의 architecture보다 훨씬 더 좋은 성능을 낸다.그래서 real-world의 아파트나 보더룸이나 오피스와 같은  syntheric 등 보면 3D 리크러스트럭션을 잘 해내는것을 볼수 있다. 기존의 learning based method는 real-world scan에서 많이 실패했었다.그런데 DGR은 그걸 성공시킨 장점이 있다.
  
  - Conclusion
    - 3D 촬영에서 이렇게 얻어낸 씬에 대한 어떤 point cloud set의 registration과 2D는 아닌 indoor scene에서 이렇게 얻어진 2개의 scene을 잘 registration(등록)하는 demo를 얻을수 있다.
    - 먼저 surface registration에 대해서 기존의 deep learning 이전의 방법들로 energy minimization으로 optimization을 하거나 handcraft 알고리즘으로 optimization해서 point cloud registration하는 기법에 대해서 이야기해보았음. 이제 그 이후로는 sparse convolution을 활용한 플립 convolution을 geometric feature FCGF나 deep global registration visual방법에 대해서 공부했음.
    - 다음 시간은 3D scene을 리컨스트럭션하거나 보이지 않은 뷰에 대해서 복원하려고 하는 노벨비 신테시스에 대해서 큰 하나의 흐름을 만들어낸 NERF를 보겠다.

## 3D vision - Neural Radiance Fields(NeRF)
- 몇장의 multi view 이미지로 3차원 복원하는 것을 자연스럽게 또 novel하게 잘 만들어주기 떄문에 많이 사용되는 컨셉이다.
- NeRF는 input으로 camera pose를 알고있는 pixel들의 레이드를 알고있어서 그걸로 렌더링하는게 목표.
- 5D neural radiance field를 optimize하는것이 목표임.
- 그래서 spartial location, pixel의 X,Y,Z 좌표와 viewing direction, 즉 어디서 보고 있는지 쎄타와 로우 그사이를 구하는것이다.
- 데모를 보면 하나의 뷰를 가지고 안보이는 뷰들도 이렇게 신테사이즈해서 그 뷰에서 찍은 것처럼 복원해낼수있는게 NeRF의 큰 장점이다.
- 비디오를 보면, 카메라가 있고 카메라가 찍은 3차원 물질들이 있다고 했을때 이렇게 레이가 만들어지고 레이는 XYZ라는 포인트와 쎄타, 파이라는 값으로 나타낼수 있다. 그거를 RGB 알파라는 값으로 복원해내는게 목적이다. 그런것을 학습하기 위해서 통과되는 어떤 픽셀값을 카메라로 부터 복원을 하게 되고 이런 레이들의 교집합으로 어떤 3차원 object를 복원해낼수가 있다. 꼭 object가 아니어도 scene같은것도 복원해낼수 있다. 기존의 SRN방법보다도 NeRF가 훨씬 더 자연스럽게 복원가능. depth map같은것도 복원가능. 

- NeRF
  - NeRF의 oveview를 하면 Neural Radiance Field Scene Representation을 복원하는것. 그리고 Differentiable Rendering Procedure가 있는것이다. 
  - 5D coordinates를 먼저 sampling을 해서 이미지를 Synthesizing한다. 그래서 camera rays, location과 viewing Direction으로 이렇게 3가지로 분류가 되는것이다.
  - location에 대해서 MLP로 feeding을 통해서 color와 volume Density를 Reconstruction한다.  
  - 다음에 volume rendering technizue를 통해서 이러한 밸류들을 이미지로 composite한다. 그래서 이렇게 통과되는 ray를 알면 어떤 이미지의 이미지를 알아낼수 있다. 결국에 우리가 가진 모니터나 이런거는 다 2차원이니깐 렌더링을 해야한다.(3차원을 2차원으로 잘보이도록 하기 위해서) 그래서 그 과정이 이제 쓰이고 그 볼륨 렌더링하는 과정이다.
  - ray의 direction, 카메라 포즈를 이미 알고 있기 떄문에 이거를 쉽게 끌고 와서 복원해낼수 있다. 이 direction과 RGB color를 통해서!
  - 그리고 이 렌더링 과정은 differentiable해서, 이거를 어떤 gradient 디시전 알고리즘으로 residual을 minimize하는 방식으로, 이제 ground Truth observed image를 사용해서 이런 reconstruction. 렌더링 로스를 사용해서 학습을 진행할수 있다.

  - 위에는 전체적인 과정이고
  - 실제 학습되는 부분은 position과 direction으로 어떤 이미지들의, 즉 렌더링되는 point들과 color의 density를 구현한다.

- NeRF : Network Architecture
  - input vector는 green으로 표현되고, intermediate hidden layer는 blue, output vector는 red, 각각의 안에 있는 숫자들은 이제 feature dimension이다. 그래서 어떤 input값이 들어오고 network가 통과를 하고 여기서 그 input과 feature가 합쳐진다. 그리고 또 통과를 해서 input과 feature가 합쳐지고 여기서 Sigma라는 density가 얻어진다. 그걸 다시 128 dimension으로 쭉 이루고 RGB color로 최종적으로 구현한다.  그래서 fully connected layer로 이제 모든 layer가 구성이 된다. 그리고 ReLU activation을 포함하고 있고 dashed line은 이제 sigmoid activation으로 RGHB를 최종적으로 구현한다. 그리고 concatenation인부분도 있다. 이건 XYZ그리고 camera direction을 concatenation을 통해서 중간에 추가적으로 더해준다.

- NeRF : Volume Rendering with Radiance Fields
  - 5D neural radiance Field는 어떤 volume and density나 di-rectional emitted된 radiance 를 통해서 어떤 space의 point들도 다시 reconstruction할수 있다는게 특징이다.
  - 데모에서 봤듯이 어떤 카메라 포즈들이 있으면 그걸 통과하는 ray들을 얻어내고 그걸 사용해서 2차원으로 렌더링을 하는것이다.

  - view-dependent eitted radiance를 visualization했을때 view1에 대해서는 조금더 훨씬 정교하고 정확하게 얻어내는걸 알수있다. view2는 더 돌아갔을떄에도 어떤 조도 변화나 이런걸 잘 복원해내는것이다. 그래서 spatial position과 viewing direction을 이제 RGB color로 ouput을 얻어내는것이다.

- NeRF : Optimizing a Neural Radiance Field
  - full model에 대해서 visualize해 봤을때 view dependent 한 값을 넣지 않았을때와 positional encoding을 넣지 않았을때 망가지는 것을 보여주는 abililation 정성적 결과이다.
  - 그래서 input coordibate만 통과했을때는 조금더 안좋고 positional encoding같은거 추가하거나 view direction까지 추가 했을때 더 안전한 결과를 얻게 된다. 그래서 veiw dependece, 그러니깐 아까 P와 view direction을 제거해주면 모델이 어떤 specular reflection정보를 잃어버려서 bulldozer의 어디가 어떤 조도를 받는지 이런부분들이 사라진다. 
  - 그리고 positional encoding 정보를 없애면 모델이 어떤 high frequency geometry나 texture를 보낸것을 까먹어버린다. 그래서 스무스하게 잘못 appearance를 모델링해버린다. 
  - position encoding과 input view direction까지 같이 넣어줬을때 거의 groud truth와 흡사하게 되는 결과를 얻을수 있다.

- Implementation Details
  - dataset은 scene에 대한 RGB image를 사용. 그와 대응되는 카메라 포즈와 instrinsic parameter, 그리고 scene bounds는 알고있다고 가정한다. 그럼 이런 intrinsic parameter와 카메라 포즈는 COLMAP SfM pipeline을 통해서 쓴다. COLMAP은 이런 multiview geometry 시간에서도 배웠던것. 어떤 이미지들 사이에서 카메라 포즈를 추적해주는 software이다. 기본적으로 Shift와 RANSAC으로 구성 되었다.
  - loss 즉, lost function으로는 Total squared error를 사용. 그래서 렌더링된 컬러와 True pixel color를 각각 비교한다.  그래서 쿨스파인 매너로 쿨스한 어떤 픽셀 컬러와 파인한 픽셀컬러를 각각 비교하는것이다.그래서 N_c = 64 dimension, N_f=128 dimension 으로 각각 레이드를 가진다. 총 4096개의 ray를 batch size로 사용한다. 
  - Adam optimizer와 exponential learning radial decay를 사용해서 이런값으로 트래이닝을 한다.
  - optimization은 100K - 300K iteration을 수행한다. Nvidia V100 GPU로 1~2days 소요됬다.

- Results(Quantative) : 정량적 결과
  -  비교하는 baseline은 neural volume랑 scene representation network(SRN), local Light Field Fusion(LLFF)이다.
  -  Neural volue은 어떤 boundry volume안에서 뒷그라운드 앞에 놓여있는 object를 novel view synthesize로 복원하는 것이다.
  -  Scene Representation Network는 XYZ coordinate을 받아서 MLP를 통과해서 continuous scene을 복원하는 novel view synthesize알고리즘이다.
  - LLFF는 photo realistic novel views를 얻기위해서 facing scence에 대한 잘 샘플된 그런 example에 대해서 학습을 하는 놈이다.

  - 결과 비교했을때 Diffuseㄷ, synthtric, 360도 나 realistic, synthetic, 360도나, real, forwardd-facing이라는 데이터에서 모두 PSNR과 SSIM이 높은 성능을 낸다.

  - example codes(Tiny NeRF)
    - Tiny NeRF로 실험해보았을때 positional encoding(PE), view Dependence(VD),Hierarchical 하이러키컬(H) 하게 학습하는것이 없을때 가장 낮은 성능 나옴.
    - 즉, positional encoding과 view dependence가 상당히 성능에 중요한 영향을 미침을 알수 있다.
    - 멀고 fewer한 image로 학습하거나, 더 작은 이미지로 학습했을때 이미지 개수가 25로 줄어들어가, 50으로 들었을때도 성능이 어느정도 감소한지 확인 가능.
    - fewer frequencies가 fewer하거나 더 많아졌을때 어떻게 성능이 변하는지도 확인할수있음
    - complete model에 대한 결과가 최종 맨 마지막 결과임.
    - No PE, VD, H 일때 cols한것의 feature를 256 size로 바꿔줫음을 알수있다.

- NeRF(Qualitative, 정성적결과)
  - test-set view에 대해서 physically-based renderer를 함께 사용한 synthetic dataset을 활용했을때의 기존 결과 비교.
  - NeRF는 texture나 표면들에 대해서 정밀하고 정확하게 복원해줌.반면에 LLFF나 기존의 SRN, NV같은 알고리즘들은 이런 디테일한 텍스쳐 같은것을 잘 복원못한다. 그런결과를 보았을때 이런 NeRF의 알고리즘이 더 잘되는것을 증명함.

  - real world scene의 test-set views의 결과. 공룡뼈다귀 T-REX구조나 꽃에 대해서도 NeRF가 기존의 알고리즘보다 조금더 정밀하고 정확하게 결과를 나타낸다. novel view를 synthetic하는것 확인가능.

  - deep voxel에서 사용했던 synthetic dataset에 대한 비교. 이런 숫자들에 대해서도 기존 알고리즘들보다도 NeRF가 더 정확하게 숫자나 문자들을 복원하고 pad style,,,이런 조형물같은것도 더 정확하게 복원하여 groud truth와 가장 흡사하게 변환해냄.

  - NeRF의 공개된 코드를 보자.
    - Tiny NeRF코드를 보자.
      - image와 camera pose, 포컬레이스를 미리 콜맵으로 구해놓은것을 로드함. 이미지는 106장의 이미지가 100x100x3짜리임.
      - pose도 4x4짜리의 fundermental mathic로 얻어져 있음.
      - 그래서 각 값들에 대해서 테스트 이미지를 하나 비쥬얼라이저 해봤을때 이런 이미지가 있음을 알수있다.
      - NeRF를 optimize하는 과정을 보자.
        - 포지셔널인코딩하는 함수가 있고, model initialize하는 함수,레일을 얻는 함수, 레일을 렌더링하는 함수를 정의함. 
        - 모델 initialize한 후에 케어스에서 바닥 optimize를 추가.그래서 이미지 i에 대해서 계속 루프를 돌리면서 렌더링 레일 하는 함수를 돌려서 학습을 해줌.그래서 gradient를 업데이트해 주고, 최종적으로 점점 시간이 지날떄마다 visualize하면서 iteration과 PSNR을 비교했을때 iteration이 돌았을때 아무일도 없지만 점점 지나갈때마다 PSNR이 증가함. epoch이 증가할때마다. 그래서 iteration이 100x 돌았을때 PSNR이 점점더 좋아짐. 그래서 선명해지고 어떤 특정 븅 ㅔ대한 카메라 포즈에 대한 이미지가 생성이됨.  그래서 나중으로 갈수록 텍스처나 질감이나 이런것들이 복원이 됨.
        - interactive visualizer를 하고 싶다면 해당 코드를 실행해서 카메라 뷰 포인트도 바꾸면서 비주얼라이저 되는것을 확인할수 있다.
        - 360도 비디오로 비주얼라이저 하는 코드도 있음.

  - NeRF : Conclustion
    - 다이렉트하게 어떤 기존의 부족한점을 많이 보완.
    - 그런데 MLP를 사용해서 object나 scene을 continuous function으로 표현하고자 하는 기존의 단점을 극복한것.
    - 그래서 Scnec을 5D neural radience field로 representing해서 더 나은 rendering결과를 얻었다는게 핵심!!
    - MLP를 사용해서 output volume density와 view-dependent emitted radiance를 3D location과 2D viewing direction으로 표현했다는것도 핵심이다
    - NeRF의 limitation은 sample-efficient가 부족하다는 점. 그래서 train, test할때 너무 여러장의 이미지가 필요할수도 있다는점.
    - interpretability도 단점이다. 즉, voxel grid나 meshes로 이제 representation을 sample할수 있어야 하는데 그게 아니라 여기서는 3D 좌표와 view point로 하기 떄문에 rendering을 어떤 voxel이나 mesh처럼 표현할수 없다.   인플리스 function으로 레이의 움직임에서 어느 정도에 위치한지로 그런거로 표현을 하게 되니까 더 가벼울수는 있어도 이제 mesh나 voxel로 표현 할수없다는 단점이 있다.

    - Follow-up works of NeRF(Nerf의 variations)
      - NeRF이후 팔로잉되고있는 아이디어들.
      - [free features Let Networks Learn High frequenct function in Low Dimensional Domains]이라는 논문.
        - positional encoding을 어떻게 하면 좋을지에 대한것을 탐구
        - 간단한 free feature mapping이 multi-layer perceptron의 high frequenct functiond을 학습하는 것과 비슷한 역할을 한다는것. Low dimensional framework domain에서. 그래서 이것을 이런 coordinate기반의 MLP를 free feature를 사용했을떄 조금더 free feature가 없을떄보다 더 잘동작하게 한다는것. 그리고 학습도 조금더 안정적으로 됨.
      
      - Multiscale Representation
        - MIP NeRF이다.  그러니깐 Anti-aliased effect를 없애고 조금더 정확하게 multi-scale 이미지를 사용해서 NeRF를 개선한 논문.
        - 기존의 NeRF 렌더링은 어떤 single-layer를 사용해서 생성헀지만 multi-image processing으로 이렇게 MIP NeRF를 적용하게 되면 조금더 많은 receptie field의 정보들을 얻을수 있고 예전에 image pyramid에서 얻을수있었떤 효과들을 다 얻을수 있다는 장점이 있음.
        - 똑같이 MLP를 통과하지만 multi-image processing을 추가적으로 진행함으로써 더 좋은 결과를 얻음.
      - Learned Initialization
        - initialize 자체르 배우는 meta-learning과 같은 기법으로 한 논문.
        - 기존의 코디넷 기반의 neural representation방식을 조금더 잘하기 위해서 meta-learning을 통과시켜서 수행하는게 핵심!
        - 그래서 좋은 initialize parameter를 넣는것이고 결과는 photo tourism을 적용했을때 다양한 이미지에 대해서도 더잘 continuous한 복원들을 얻어낼수 있는것.
        - 이미지의 wide-based eye사이에 있는것들을 복원해낼수 있다는점에서 아주 인상적인 paper임!

      - Relighting
        - 조도변화, 어떤 그림자의 변화같은것들을 잘 캐치해내는 결과를 얻을수 있음. 그리고 material editing같은것도 할수 있음. 그래서 어떤 light visibility나 direct illumination, indirect illumination같은 것들을 decompose해내서 이런것들을 조절해가면서 한 이미지에 대해서 rendering을 진행하거나, 알베도나 roughness, shape map, shadow map, 그리고 indirect한 값들을 synthesize하고 복원해낼수 있는 결과를 얻을수 있음.

      - 최근에는 NeRF in the dark라고 해서 2020년에 나온것.
        - high dynamic range view synthesis from noisy raw image라고 CFT 2022년에 oral presentation으로 발표됨.
          - 그냥 NeRF를 한게 아니라 raw sensor image를 받아서 NeRF를 진행한것. 그래서 High Dynamic Range HDR views들을 렌더링을 하고 post processing을 통해서 더 좋은 결과를 얻음. 
          - 그리고 in the dar가 들어간 이미지는 그냥 다크한 이미지가 아니라 extremely dark한 이미지를 복원해낸다는것이 특징이다. 
          - 마찬가지로 denosing도 NeRF in the dark를 통해서 수행해줌.
          - 사실 denosing과 raw light는 어떤 듀얼 프로그램의 관계이다. 단순히 어두운 이미지를  밝게 만들면 노이즈가 심하게 생긴다.(보라색으로!) 그걸 막기 위해서 이렇게 노이즈한 이미지가 생기는데 그걸 어떤 처리를 통해서 깔끔하게 만들어 내는게 NeRF in the dark의 목적인것이다.

  - GIRRAFE라는 논문. CVPR 2021년에 best paper받음.
    - 어떤 object가 있고 shape과 appearance를 input으로 넣었을때 feature field를 샘플해낼수있음. 거기에 pose 값을 통과해서 멀리보내거나 자동차를 돌리거나 이런 결과를 얻을수있음. 그래서 이걸 쭉 더했을때 어떤 3D scene representation을 구현할수 있고, 이미지를 카메라 뷰포인트와 함께 컴포지션을 했을때 디코드를 통과시키면 어떤 이미지가 합성된 렌더링된 이미지를 얻을수 있다.
    - 3D 상황에서의 어떤 복원들을 깔끔하게 해줄수 있다는 장점이 있다.그래서 레이턴트 코드를 이렇게 이미지로 복원해낼때 어떤 pose variation에 대해서 직접 조절할수 있게 해주고, 이미지 컴포지션도 매우 자연스럽게 해줄수 있는 결과를 얻음.
    - 그래서 기존에 controlable scene generation이나 GAN 기반의 방법들과 결합해서 아주 좋은 결과를 얻은 논문임.

    - 이런방식을 local feature에서 활용하고자 하는 NeRF방법도 있고, 또다른 이미지 인해스먼트로 다시 풀려고하는 움직임도 있다.
 

## 3D sensors, datasets, open3D, COLMAP
- open3D : 3D 라이브러리인 opencv
- COLMAP : 3D + multi visualmetric libarary

- 3D sensor
  - 다양한 센서들로부터 추출해낼수 있음
  - 이러한 데이터 처리하기위해서 오픈소스 라이브러리인 open3D, COLMAP이 있다.

- open3D
  - 3D data, 특히 point cloud, voxel, mesh등을 다룰수있는 operator들이 존재.
- COLMAP
  - 멀티뷰이미지들을 사용해서 SFM 파이프라인을 돌려서 어떤 point cloud set를 reconstruction하거나 또는 MVS 멀티뷰 stereo 알고리즘을 통해서 랜드마크들을 reconstruction할수 있는 알고리즘들이 내장되어 있음.

- 3D sensors: depth map
  - depth map은 per-pixel distance measurement를 통해서 depth를 얻어야 하는것.
  - pixel map 사이에 hole이 생길수 있음. 특히 mirror, metal surfaces, dark object에서 hole이 생길수 있음. 특히 거울같은경우 반사되는 어떤곳이기 떄문에 visualy의 어떤정보를 취득했을때 어떤 문제가 생길 여지가 조금씩 있다. 그래서 거울만을 다루는 어떤 3D task도 있다.거울에 대해서 잘 robust하게 측정되도록!
  - 즉, 이런 이미지들이 있을때 truth depth map이 존재하고 있고, 이런 이미지가 있을떄 뒤에는 빨간색 그리고 앞부분은 파란색 ㅏㅌ이 컬러 코딩된 이미지로 depth map을 구할수도 있다.

- various approaches
  - 다양한 approaches로 depth map을 얻을수 있다.
  - depth camera의 variants들이 있다.
  - 기본적으로 passive stereo : 오른쪽 왼쪽 카메라의 스테레오 비전을 통해서 3D 정보를 스트라이앵길레이션을 통해서 얻어내는것.
  - active stereo : projector를 통해서 object까지의 거리가 얼마나 있는지와 2개의 카메라로부터 depth정보를 얻어낼수도 있다.
  - structured light : 프로젝터와 카메라 사이에서 프로젝터와의 거리와 카메라의 관계성을 통해서 object를 construction할수 있다.
  - Time of flight(TF카메라) : TF카메라를 통해서 레이저 소스로부터 인프라 레드 라이트를 통해서 카메라와의 depth를 추정해야 하는 방식도 있다.

- structured light
  - key idea : artificial texture로부터 사용해서 stereo matching을 수행하도록 하는것. 그래서 어떤 패턴 , scene의 패턴들을 프로젝터를 통해서 카메라 이미지와 프로젝터를 사용해서 그사이의 관계성을 통해 depth를 추정하는것. 그래서 scene에 대한 camera observe가 있다.
  - stereo matching을 하는데 알고있는 패턴과 관측된 패턴 2개 사이에 스테레오 매칭을 수행하는것.

- depth camera(prime Sense & kinect)
  - scene은 패턴이 좋지 않을수 있다. 그래서 key idea는 어떤 imfra-red인 invisible rays을 통해서 emitter를 통해서 어떤 visuable한 영역에 이상의 인프라레드파장을 통해 어떤 depth를 추측하는것. 그래서 자외선부터 적외선으로 갈때 어떤 파장과 그 피지컬한 정보를 활용해서 depth추정을 하는것. 
  - depth카메라를 활용해서 인간의 모션같은걸 복원해서 사람이 증강현실같은 앱같은걸로 활용가능.
- depth camera(Time of Flight)
  - depth정보를 얻어낼수 있는 대표적인 센서중 time of flight camera가 있다.
  - round trip time을 measuring해서 즉, 어떤 시그널을 발사해서 돌아오는 시간을 measure해서 어떤 artificial signal을 레이저나 LED로부터 얻어내는것. 
  - 그래서 periodic signal이나 phase shift같은것을 계산해가지고 최종적인 depth를 계산해야되는것.

- Lidar Sensor
  - 어떤 차의 머리통에 붙어가지고 빙글빙글 돌면서 주변에 있는 point cloud를 추출해내는것. 그래서 이것도 time of flight센서의 extenstion 버전이다. 좀더 강한 emitter와 receiver를 가지고있다.
  - 그리고 fast spinning을 통해서 360도를 모두 추출한다는 특징이 있다.
  - 라이다 센서에 감도를 세게 하면 사람몸이 뚫려버릴수도 있고 이레이저 때문에 조심해야 되는것중 하나이다.
  - 이런정보들을 가지고 레지스트레이션을 통해서 현재 위치를 추정하거나 맵을 복원하는 SLAM(Simultaneoud localization and mapping)도 수행

- 3D scene datasets / 3D object datasets
  - RGBD dataset
    - RGB 이미지가 있고, 그에 대한 depth map이 있고, 또한 semantic segmentation정보도 포함되어 있는 dataset이다.
    - 1449개의 densely labeled pair, aligned된 RGB와 depth map이 있다. 
    - 3개의 도시에서 464개의 scene을 추가적으로 구현.
    - unlabeled frame도 같이 있어서 이런정보들을 활용해 여러가지 test를 풀수 있음.
    - 그래서 주어진건 color image와 depth image, semancti segmentation이 주어진다.
  - ScanNet
    - RGBD기반의 inddor scene reconstruction위해서 만들어진 dataset 
    - scanNet 벤치마크에서 존재하는 레이블은  3D semantic label, 3D semantic instance, 2D semantic label, 2D semantic instance, Scene tyle classification까지 수행가능.
    - 이걸 업그레이드해서 ScanNet V2가 나옴.
    - 기존 scanNet의 단점을 보완한, 좀더 사이즈도 커지고 좀더 리얼리스틱한 환경의 데이터셋을 릴리즈함.
  - KITTI
    - 차량에 붙어있는 카메라나 라이더를 통해서 추출해낸 정보들을 활용하는 데이터셋. 모든씬이 차량에서 찍힌 이미지로 되어있음.
    - velodyne laserscanner를 ㅌ오해서 reconstruction을 해냄.
    - calibration된 카메라와 레이저 센서를 통해서 카메라 포즈도 알고있고, groud truth정보는 레이저데이터와 398개의 스테레오이미지, 8만개의 object detection bounding boxes와 depth, sceneflow, optical flow, 그리고 Automatry, tracking, semantics까지도 포함.
  - CityScape dataset
    - 차량에서 찍은 도시사진들.
    - large scale semantic segmentation dataset
    - 드라이빙 시나리오를 가정하는 데이터셋 
    - 30개의 클래스와 50개의 독일도시들, 5천개의 fine annotation된 이미지와 20000개의 coarse annotation된이미지
    - pixel-level의 semantic label과 instance level의 semantic labeling이 되어있다.
  - Semantic KITTI
   - 28개의 클래스가 semantic segmentation annotation되어있는 kitti와 같은 환경에서 찍힌 데이터셋.
  - Waymo Open dataset
    -  센서데이터로는 1개의 mid-range lidar와 4개의 shot-range lidar, 그리고 5개의 camera로부터 얻어진 값들을 가지고 센서퓨전등의 일들을 수행.
    - labeled data는 vehicles, 보행자, 자전거타는사람, 도시표지판들이 있다. 12만개의 3D bounding box label이 트래킹 아이디와 함께 주어져 있다.
    - 카메라에 대한 데이터에 대한 high-quality label들이 있다.
    - 2D bounding box label도 존재
  - KITTI 360 dataset
    - KITTI에서 360도 정보까지도 포함하고 있는 데이터셋
  - Indoor LiDAR dataset
    - 내부를 사람이 라이다나 RGBD scan을 통해서 촬영한 데이터셋
  - AI Habitat - HM3D
    - 주거공간에 대해서 reconstruction이 되어 있는 정보가 있는 dataset

  [3D object datasets]
  - ShapeNet
    - single object에 대해서 다양한 카테고리의 object가 있는것.
    - 결과에 대한 3D 모델이 있고, shapeNet-core는 55개의 카테고리에  51300모델을 가지고 있다.
    - 단점은 오버나이즈가 되지 않는다는것.
    - 이걸로는 classificiation과 part segmentation task를 풀수있다.
  - ABC dataset
    - 레이블되지 않은 1만개의 CAD part model들이 가지고 있어서 curves나 patches수를 가지고 있다.
    - potential 한 application으로는 patch decomposition이나 shape feature detection, shape reconstruction, 그리고 normal estimation 을 할수잇는 benchmark이다.
  - ABO Dataset(Amazon Berkeley objects dataset)
    - 어떤 프로덕트의 메타데이터와 프로덕트 카탈로그이미지, 그리고 360도 이미지, 3D 모델이 있는것. retail이나 소비재 같은거에 대한 정보들이 있는 dataset
    - 아마존 프로덕트에 대해서 어떻게 정보들이 있고, 그것을 정리해놓은 데이터셋.
  - Google Scanned Object
    -  장난감이나 수납장들같은거의 어노테이션과 3D모델들이 있는 데이터셋.

  - public benchmark에서 모델을 개발하게 되면 이전 previous method를 이기는 것만으로도 노벨티가 생기고, 또 자기가 디자인한 알고리즘에 대해서 얼마나 효과적인지를 확인할수 있다.


- 3D library인 open3D와 COLMAP
#### Study surface registration
##### Convex optimization based
#### Learn how energy functions can be minimized

### Sparse convolution on the quantized point cloud representation in voxel grid
: 딥러닝 시대에서 어떻게 convolution을 point cloud에 적용하는지 알아보자.


## Implicit Funcion : NeRF
## Open3D
## COLMAP (SfM, MVS)
## 3D reconstruction
## Human Reconsturction
## SLAM and LiDAR

# Generative models and graphics


---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}