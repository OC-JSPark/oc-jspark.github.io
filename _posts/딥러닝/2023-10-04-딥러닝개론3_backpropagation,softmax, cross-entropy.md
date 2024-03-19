---
title: "딥러닝개론3 : backpropagation,softmax, cross-entropy"
escerpt: "backpropagation,softmax, cross-entropy"

categories:
  - DeepLearning
tags:
  - [AI, DeepLearning,backpropagation,softmax, cross-entropy]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-10-04
last_modified_at: 2023-10-04

comments: true


---


# 10. 역전파 (backpropagation)

- 참고자료
  - [CS231n lec4 - Introduction to Neural Neworks](https://www.youtube.com/watch?v=d14TUNcbn1k)
  - [CS231n Backprogagation 강의자료 한글](http://aikorea.org/cs231n/optimization-2/)
  - [CS231n Backprogagation 강의자료 영어](https://www.youtube.com/watch?v=d14TUNcbn1k)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/401ff1e1-c950-4e09-9ba6-4f12bcb4e982)

- 데이터가 들어온다.
- 신경망구조에는 3개의 특성이 입력으로 들어오게 설정.
- 데이터가 2개 들어온다고 가정하자.
  
- 데이터가 모델에 들어와서 가중치가 곱해져서 예측을 하게 되고
- 예측과 정답의 차이인 오차들의 합(error들의 합) 인 cost를 구한다.
- 그 cost를 **역전파**시켜서 모델에 학습시켜서 모델이 조금더 좋은 예측을 하게 하는 거로 update시킨다. 이때, **역전파시켜서 변화되는건 모델의 가중치 부분이 update되는것이다.**

- 가중치 matrix안의 숫자들을 바꾸는것이다. 
  - 첫번째 가중치는 feature가 3개이기 때문에 3개의 input이 있고 5개의 output이 있는 matrix이다.(3x5 matrix)
    - feature는 행을 의미한다.
    - node수는 열을 의미한다.
  - 그러면 input data가 2개였기 때문에 output도 2개의 data와 5개의 feature가 나온다.

- 가중치를 업데이트하는 기준은 optimizer를 사용하든, 다양한 방법이 있다.
- 기본적으로 gradient descent방법사용한다고 하면, 
  - w1 = w0 - α(∂C/∂w0) 
  : w1의 기울기가 증가하거나 감소할때 cost(Σ error)의 증감을 고려한다. 
- 이를 수많은 가중치에 대해서 일일히 모두 고려해야하는가?
  - 이러한 방법을 간단하고 효율적으로 하는 방법이 **chain rule** 이라한다.
  : layer_1에서 layer_2가 갈때의 변화율과 layer_2에서 layer_3로 갈떄의 변화율 등의 관계를 이용해서 곱해줌으로써 direct로 가는 걸 구할수 있다.


## 10-1. 연쇄법칙이란?
:a에서 c로 갈때의 순간 기울기 = a에서 b로갈때 순간기울기 x b에서 c로갈때 순간 기울기

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/cc3b14f4-82ba-4bbe-b741-b6cd92456dcd)

  - a->c : a가 증감할때 c는 얼마나 변화될까? (순간변화율의 개념)
  - 이걸 2단계로 나누어서 계산가능.
    - b->c : b가 1만큼 증감할때 c는 얼마나 변화할까?
    - a->b : a가 1만큼 증감할때 b는 얼마나 변화할까?
  - w1 -> w'
    - w' = w1 - α(∂C/∂w1)
      - **weithgt_1이 변화할때 cost가 얼마나 변화하는지를 구하고자 하는게 목적이다!**

  - 이건 w1 x w2로 cost의 변화를 구할수 있다.
  - 이땐 3가지로 나눌수 있다.
    - O: output으로 예측값이다.
    - 예측값과 cost의 기울기
    - O = z1w2 + z2w2' + z3w2'' 
      - 이떄 z1w2 뺴고는 뒤에는 모두 상수취급한다.
      - 즉 z1이 변화할때 O의 변화값은 w2이다. 

  - w1이 변화할때 z1이 얼마나 변화하는지만 알면 w1이 cost에 미치는 영향을 알수 있다. 

### 10-1-1. 역전파 pytorch 예시

```
class Net(nn.module):     ## pytorch에서는 class를 사용해서 Network의 구조를 정의한다.
  def __init__(self):
    ...
  def forward(self,x):
    x=...

net = Net()               ## net = Net() : Network를 object로 만든다.
output = net(input)       ## output을 출력한다. 
loss = criterion(output,target) ## output과 input의 y값을 비교해서 loss를 구한다.(criterion : loss function이다)
loss.backward()           ## loss.backward() : 역전파가 실행된다.
```



### 10-2. 역전파에서의 ReLU 미분

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/4b50de26-fadf-43de-9519-a4eefcfbc478)

- 실제로 신경망구조 따를때 선형변환을 통해 나온값을 가지고 다음 layer에 전달시 비선형변환(relu, sigmoid 등)을 거치게 된다. 그러므로 relu 변환에 대해서도 기울기의 변화율을 계산하여 역전파 과정에 포함시켜야 한다.

* 새로운 함수나 forward를 정의할때 새로운 layer를 추가하고 싶을때, 그것이 미분가능한지 check해주어야 한다. 미분이 불가능하면 역전파를 할수 없기 떄문이다.**(differentiable)**



# 11. 소프트맥스(softmax)
:분류 문제를 풀기 위한 방법(회귀->분류로 변형필요)

- 출력노드가 하나일때(=예측값이 1개)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/191190c9-6c01-4bf1-9da1-e3d5341220e9)

  - workflow 
    - 1) input : 3개의 feature
    - 2) 선형변환과 비선형변환을 통해 최종적으로 예측값이 나오게된다.
    - 3) 예측값(h) - 실제값(y) = error를 구하게 된다.
    - 4) error들의 합 = cost로 계산한다.이를 역전파시켜서 모델을 학습시킨다.

- 출력노드가 2개일때(=예측값 2개, binary classification)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/6c2d3772-e13e-487d-b914-13e6b8653185)

  - regression(회귀) 에서는 값을 예측하기만 하면된다.
  - classification(분류)에서는 binary인 이진분류가 있다
  - ex) 정상/비정상, 개/양이 등 분류. 
  - 이때는 출력노드를 2개로 지정함으로써 각각의 class에 대한 예측값을 output으로 지정하고 예측값이 더 높은 class에 대해서 예측했다고 판명한다.

  - 그럼 이럴때, cost에 대한 error는 어떻게 구하나?
    - ex) 이미지는 개이며 그것에 대한 예측이 맞았는지 틀렸는지 계산해보고 싶다.
    - one-hot encoding을 이용하여, **예측값 - (one-hot encoding으로 계산한 실제값)의 차이 = error를 구한다.**



## 11-1. 소프트맥스함수 설명
: 예측값들을 확률값들로 바꿔주는 함수이다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/b8f1939f-fe29-4ecb-9ffe-989254bcafe1)

  - K는 class를 말한다.
  - z는 예측값을 의미한다.

- 특징
  - 이걸 계산하면 확률값이 나온다.
  - 확률값의 총합은 반드시 1이다.(100%가 나온다.)

- 소프트 맥스 사용이유?

  - 2가지의 문제점 해결을 위해서 사용.
    - 예측값을 그대로 사용한다면 문제가 생긴다.
    - 1) 예측값이 음수가 나온다면...확률값계산과 매칭이 안된다.
    - 2) 예측값의 sum이 0일떄는 divide by zero 에러가 뜬다.
  - solution
    - 우리는 e라는 자연상수를 이용해서 지수함수를 이용한다.
    - e의 지수로 사용.  
    - 모든 실수범위에 대해 0~무한대까지 scale의 range를 바꾸게 된다. 


# 12. 크로스엔트로피(cross-entropy)
:분류 문제를 풀기 위한 방법(회귀->분류로 변형필요)


## 12-1. 정보량

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/46ff2cf0-32e5-4614-8856-a68b4e4b7a73)

- 정보: 어떠한 사건이 있을때 해당정보를 얼마나 갖고있는지.
- Information(정보량) = 사건A가 일어날 확률 p(x) 에 -log 취한다.
  - 확률은 0~1사이의 값을 갖는 값이다.
  - p(x) -> 0 : I => 무한대 
  - p(x) -> 1 : I => 0 으로 수렴.

- 즉, 해당사건이 발생할 확률이 적을수록 정보량(information)이 많아진다.
  - ex) "내일 해가 동쪽에서 뜬다" : 발생할 확률이 100%이므로 정보량=0이 된다.

## 12-2. Entropy(=average information,정보량의 기대값) 
: 불확실성

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f96f4f73-7e7e-4a06-a043-f6a3fcc14bc4)

  - 사건A가 일어날 확률p(x) x 정보량 을 곱해서 전부 더해주면된다. 

- 어떠한경우에 정보량의 기대값인 entropy가 커지는지가 중요하다.
: 같은비율로 확률로 존재할떄 entropy의 값이 최대화가 된다.

## 12-3. cross entropy
: p와 q의 두 분포사이에 entropy를 계산하는것이 cross entropy이다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/1fd7b14f-732f-44a9-8b1c-1a8b5696d8ac)

  - q(x) x 정보량을 곱해서 전부 더해주면 된다.
  : entropy에서 사건A가 일어날 확률 p(x) 대신 분포 q(x)를 넣어주면된다.


- 이걸 어떻게 사용할수 있을까?
  - 분류문제에서는 예측값이 여러개 나올수 있다. 

|p|q|
|---|---|
|소프트맥스 처리한 예측값|onn-hot encoding한 정답값 y|

  - p와 q 분포사이에 어떤 entropy를 구할수 있다. 
  - 정답인것만 one-hot encoding에서 1이고 나머진 0으로 잡히기에 계산식에 넣으면 정답인것만 남게되어 entropy를 쉽게 구할수 있다.

- entropy는 정보량의 기대값이다. 정보량이 많으려면 해당사건이 발생할 확률이 적어야 하고 즉, 불확실성을 의미한다.
- 즉, p와 q 예측과 정답사이에 불확실성을 구하게 된다면 두분포간의 차이를 계산할수 있으며, 그것을 통해서 cost or loss function으로 사용할수 있다는 의미이다.

- 왜 cross entropy를 loss/cost function으로 사용할까?

|오/정답 구분|loss/cost function|비고|
|---|---|---|
|100% 정답|0|완전히 정답예측한 경우|
|100% 오답|무한대|완전히 틀린예측한 경우|

  - 이것이 우리가 원하는 loss/cost function의 역할이다.
  - 왜냐하면, **오답을 냈을경우 cost를 굉장히 크게 잡아서 가중치변화를 크게 잡을 것이고 정답을 100% 맞추었다면 가중치를 바꾸면 안되기에 0이 나와야 한다.**

- 이러한 이유로 cross entropy를 사용해서 classification 문제에서 다양한 확률값을 output으로 하는 모델에서 p라는 예측값과 q라는 정답값의 분포를 비교해서 정보량의 평균을 계산해서 분포간의 차이(차이가 클때 cost가 커져야 하기때문에) 그러한 맥락으로 우리는 cross entropy를 사용하여 cost를 계산하고 classification문제에서 가중치를 업데이트하는데 사용된다.





---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}