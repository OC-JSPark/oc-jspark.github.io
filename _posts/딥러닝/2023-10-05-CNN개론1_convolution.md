---
title: "CNN개론1 : convolution"
escerpt: "CNN 개론 공부"

categories:
  - DeepLearning
tags:
  - [AI, DeepLearning, CNN, convolution]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-10-05
last_modified_at: 2023-10-05

comments: true
 

---

# 1. 이미지필터(filters)

## 1-1. pixel
: 이미지의 최소한의 단위

- image grayscale(흑백)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/513bbefa-fc3d-4e33-94e5-ca0b230cc2f3)

- pixel value
: 0(흑) ~ 255(백)

  - 0~255 -> 0~1 로 **normalization** 하는경우도 있다.
    - 모든 pixel의 값을 255로 나눠주면 된다.

## 1-2. RGB
: image는 3channel이 있다.(R,G,B)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/4f62544d-c635-4534-865f-60a09920490c)

- (R,G,B)를 가진 pixel이 w x h만큼(여기선 4x4), 즉 이미지의 pixel만큼 존재할것이고 그것이 채널갯수만큼 존재한다.
- 즉 3차원 데이터라고 생각하면 된다. 조금더 쉽게 2차원 이미지가 3개 존재한다.채널축은 R,G,B를 의미하는 3개의 이미지가 존재한다

* grayscale: "흑백" 을 의미한다.

## 1-3. 이미지필터
: 3x3 filter(=kernel)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/bd7d10e5-f901-47c9-9fe3-fe0c7f7d66b2)

  - sharpen filter : 선명효과
  : edge부분이 조금더 선명해진다.  

  - blur fileter : 흐림효과
  : edge부분이 흐려져서, 조금더 자연스럽고 부드럽게 뭉게지는 효과
    - center부터 1/2씩 줄어드는 값을 갖고있다.

  - filter 적용시 input image 보다 output image의 feature map 크기는 작아진다.

- 실습해보기
  - [Image Kernels Explained Visually](https://setosa.io/ev/image-kernels/)

# 2. 합성곱 연산1 (convolution)

## 2-1. 신경망에서의 이미지 입력

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d858152d-c3bf-410e-a84e-9fa4b7e146d3)

  - 5x5 pixel을 지역적 위치 정보를 없애고 한줄로 만든다.
  - 그렇게 25개의 feature를 가진 matrix가 생성된걸 input node에 들어오게 된다. 
  - 기본적인 feature들의 가중합을 통해 새로운 feature들을 추출한 layer_1이 만들어지고 또한번추출하여 layer_2만들어지고 마지막으로 주어진 데이터들의 다양한 feature들을 추출해서 하나의 예측값을 출력하는 구조이다.

  - 문제점) 이미지내의 region 부분이 필요한데 이러한 정보도 input으로 넣어줘야 인간이 image를 판단할때 그런 정보 활용기법이 유사해진다. 
  - solution) 그래서 cnn기법을 활용하여 region정보도 신경망 모델에 입력시켜서 최종적으로 예측할수 있게 진행되었다.

## 2-2. cnn에서의 이미지 입력
: 각 pixel의 정보를 그대로 사용하는것이 아니라, region정보를 하나의 특성으로 사용하고 싶다. 그러기 위해 어떠한 필터를 사용하여 연산하여 region정보가 포함된 **압축된 feature**(하나의 pixel을 의미하는게 아님)를 만들고 이를 input neural network에 넣을것이다.  

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f4544d54-2888-40ee-b0b2-9d56d86ffc03)

  - layer가 깊어질수록 region의 size가 증가하게 된다.
  - 즉, **우리는 input image의 region 정보를 충분히 활용해서 convolution 연산을 통해서 지역적인 정보를 추출해서 다음 layer로 압축한 정보를 넘겨주게 되고 그것을 가중합을 통해서 feature를 생성하고 이런한 과정을 반복한 다음에 최종적으로 굉장히 상세하고, 많은 정보를 담고있는 feature를 활용해서 예측을 진행하게 된다.**

- cnn에서의 이미지 입력 workflow 
  - 1) pixel matrix 만든다 
  : input image를 normalization을 활용하여 0~1 사이의 matrix를 만든다. 그리고 grayscale(흑백) 으로 설정한다.

  - 2) convolution 진행한다 
  : input image에 3x3 filter를 적용하여 연산시킨다. 즉, input image에 해당 region의 3x3 filter를 통해서 연산을 통해 나온 **압축된 정보**를 의미한다.


## 2-3. padding
: input data의 외각에 지정된 pixel만큼 특정 값으로 채워 넣는것을 의미. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e677b088-eeec-426f-a56e-a3e61b55384e)

1. 이를 통해 convolution layer의 출력 데이터의(feature map) 사이즈 조절 가능
2. 외곽을 "0"값으로 둘러싸는 특징으로 부터 인공 신경망이 이미지의 외각을 인식하는 학습효과도 있음.
3. **input size와 output size(feature map) 를 동일한 size로 조절가능**
  - channel수 = 사용되는 filter의 수

## 2-4. stride(=window)
: filter(kernel)가 지정된 간격으로 image를 순회하는 간격을 stride라고 한다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/213a3e5c-0a46-4213-95bc-d861b3d748b6)
  - 위 이미지는 stride=1인 경우이다. feature map은 3x3 형태
  - stride = 2로 주게 되면 feature map은 2x2 형태
    - 2칸씩 띄어서 image를 순회하기 때문에 2x2 형태의 feature map이 만들어 진다.

# 3. CNN 1 (convolutional neural networks)

## 3-1. convolution filter
: 값이 고정되지 않은 필터이다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/246fe56b-2d92-4bc1-877c-f9a29e05bacd)


  - 신경망구조를 cnn에서 그대로 이용가능하다. 왜냐하면 convolution filter를 통해서 featrue map을 만들때 그 과정의 연산이 가중합(weighted sum)의 선형회귀의 연산과 동일하기 떄문이다.

  - 즉 convolution 연산도 weighted sum을 출력하는 연산이기 때문에 신경망구조와 동치의 의미로 받아들일 수 있다.

  - 우리는 image data를 활용해서 신경망을 활용한다. pixel하나하나를 사용하는게 아닌, 여러가지 pixel, 근처에 있는 범위를 하나의 특성으로 뽑아내는 그런 과정으로서 convolution을 사용한다. 이것의 연산자체가 weighted sum이기 떄문에 우리는 신경망 구조로 표현할수 있고 실제로 표현이 된다.

- 필터n개 = 다음 layer의 node 수

## 3-2. channel
: 필터를 몇개 사용하는지에 따라서 몇개의 output이 나오게 되고 그것이 출력되는 channel수가 된다.

  - input 이미지에 filter 3개를 적용하여 관련 feature를 뽑개되면 3개의 feature가 뽑히며, 그것이 바로 3개의 feature map이 된다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/a5e5b9ad-42f9-4d95-803a-6dfec2fec0f0)

  - (c,w,h) : channel ,width, hegith
  - filter 5개 : filter마다 각각의 weight가 들어있다.
  - output으로는 channel수가 나온다. 이전 층의 channel수는 상관없게 된다

  - padding을 줫다고 가정했기 떄문에 전체적인 size(256x256)는 변하지 않음.


  |channel 수|의미|
  |---|---|
  |1|grayscale(흑백)|
  |3|rgb|


## 3-3. Max pooling
: 채널수의 변화는 없으며, feature map의 size가 w,h가 변한다.

- 사용이유 ?
  - parameter 수 감소를 위해서. 
  - parameter수가 크면 연산량이 많아지게 되고 리소스가 많이 든다. 
  - 즉, 비용이 커진다.
  - 예를들어 이미지 512x512 x channel =  layer의 parameter 수


![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/f740f57d-f3e0-436d-b00f-ac55c1162933)

## 3-4. VGG16 Architecture 해석

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d1a9ff55-dee0-45b5-9b6b-bd693fc10b2f)

  - channel기준으로 해석해보기
    - input : 3channel (RGB)
    - 1 layer : 64 channel 
      - filter개수가 64개 사용됨을 의미.
    - 2 layer : max pooling 1번, 128 channel
      - w,h 가 1/2씩 줄어듬.
      - filter가 128개짜리 사용됨
    - 마지막 layer (1 x 1 x 1000) : 1000 channel = 1000개의 classification
      - 이후 softmax를 통해서 확률값으로 변환하고 classification task를 수행하는 모델이 된다.
      - = fc(fully connected) layer = flatten layer(한줄로 폈기 때문)라고 한다.

## 3-5. 신경망 vs cnn

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/46cc4469-749e-49ef-b53f-666fe6bf64e8)

  - 신경망은 데이터가 들어올떄 feature값이 들어온다. 이것들의 가중치조합을 통해서 새로운 feature를 추출하고 다음 layer에서도 new feature를 추출하고 최종적으로 예측하게 된다.

  - cnn에서는  이미지가 들어오면 여러가지 filter들을 통해서 여러가지 featrue map을 생성한다. 전체 map을 사용해서 convolutional 연산을 통해서 새로운 feature map이 만들어진다. 몇개의 node가 다음 layer에 나올지는 channel수에 따라 달라진다.
    - 즉, filter = n개 => channel = n개 가 나온다.
    - fc layer의 node개수 = classification할 class수

## 3-6. LeNet Architecture 해석

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/de5911bb-1c10-4a6a-a0fd-e5aa359c1f42)

## 3-7. CNN 코드 해석

- class를 통해서 신경망 구조를 정의하게 된다.
- 64개의 output channel 의 의미는 64개의 filter를 사용한다는 의미

```
class Net(nn.module):
  def __init__(self):
    super(Net, self).__init__()           ## 기존의 nn.module에서 사용되는 기본 구조를 상속받는다.
    self.conv1 = nn.Conv2d(3, 64, 5)      ## conv2d(input channel 수, output channel수, kernel size)
    self.conv2 = nn.Conv2d(64, 128, 5)    
    self.fc = nn.Linear(128 * 5 * 5, 10)  ## Linear : fc layer 구성하는것. 128개의 채널에 존재하는 모든 parameter들을 계산한것. 최종적으로 10개의 classification 예측하는것.

  def forward(self, x):                   ## 구성해둔 layer들을 어떻게 연결시키는지 정의하는 부분.
    x = F.relu(self.conv1(x))             ## conv layer를 통과시킨후 반드시 비선형변환(relu)를 거쳐야 한다.
    x = F.max_pool2d(x, 2)                ## 2x2 max pooling 진행
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)               ## flatten으로 한줄로 표현
    x = self.fc(x)                        ## fc layer 통과시킴.
    return x

net = Net()
output = net(input)
loss = criterion(output, target)          ## criterion : cross entropy같은 loss function을 사용해서 output과 정답target의 loss를 계산
loss.backward()                           ## 해당 loss를 역전파시킨다.
optimizer.step()                          ## optimizer(SGD같은거)를 통해서 다음 스텝으로 넘어가면서 가중치를 업데이트 시킨다.
```


# 4. CNN 역전파

## 4-1. forward

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/035dd5b5-2c3c-452c-8cd2-b0f1b1663349)

## 4-2. backward

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e0401ac6-5b2f-41c1-bb73-71e39046155a)


  - layer들이 존재한다면 loss 값이 계산되었을것이다.
  - 이것이 layer들을 반대로 올라가면서 역전파 될것이다. 
  - node 입장에서는 연결되었던 부분에 한해서 loss의 영향이 주어지고 node와 연결된 node들이 영향을 받게 된다. 
  - 해당 output3개에 대해서 δ(델타)의 loss값이 역전파되었다고 가정해보자.
  - 지금 하고자 하는것이 **가중치를 업데이트 시키는것**이다.

  - **즉, 가중치의 변화량이 loss의 변화량에 어떻게 영향을 끼치는지...즉 가중치가 1만큼 변할때 loss가 얼만큼 변하는지**  
    - w1 -> w1' * ∂L/∂w1

  - loss = δ1 + δ2 + δ3 이라고 생각함.

    - 커널이 첫번째 연산을 시작했을때 w1 입장에서 x1 x w1 => o1로 갔다 
    - w1입장에서 forward를 보면 x1과w1 곱해서 o1이 만들어짐.
    - 그러면 **∂O1/∂w1** 의 의미는?
      - w1이 output1에 미치는 영향력 은 어떻게 되는가?(=순간변화율) = x1 이 나온다.
        - 왜? w1이 1만큼 증가하면 뒷부분은 편미분이니깐 지워지고 1x x1만큼 증가하는게 o1이다. 그러므로 x1이 나온다.

  - output과 loss와의 관계를 보자.
    - δ1 + δ2 + δ3 이 각각 전달된다. 
    - 이것들의 합이 w1이 변할때 loss가 얼마나 변하는지에 대한 값이 나온다.
    - 왜냐하면 chain rule떄문인데,, 전파되어온 loss에 대한 기울기 값이 !!

  - 이걸 활용해서 학습시킬수 있다. 
  - cnn의 구조가 결국 신경망이기 떄문에 크게 다르지 않다. 신경망의 역전파와 크게 다르지 않다!!





---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}