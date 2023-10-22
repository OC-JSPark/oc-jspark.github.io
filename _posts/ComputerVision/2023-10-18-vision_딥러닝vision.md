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