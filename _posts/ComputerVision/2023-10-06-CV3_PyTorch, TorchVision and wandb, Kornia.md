---
title: "CV3 : PyTorch, TorchVision and wandb, Kornia "
escerpt: "Computer Vision 기본이론 3"

categories:
  - Vision
tags:
  - [AI, Vision, PyTorch, TorchVision, wanddb, Kornia ]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-10-07
last_modified_at: 2023-10-07

comments: true
 

---

# 5. PyTorch, TorchVision and Kornia 
: 딥러닝 하기위한 프레임워크인 pytorch

## 5-1. Pytorch

- GPU와 CPU를 사용해서 deep learning 을 하기 위해 최적화된 tensor library이다.
- 원래 Lua라는 프로그래밍 언어로 된 Torch였으나, 오픈소스로 여러사람이
개발하다가 facebook AI 가 개발에 참여하면서, 사용성이 좋아지고 있다.
- PyTorch로 넘어오면서 특히 좋은 점은 자동 미분 모듈(auto gradient)이 있어서 따로 backward함수를 구현하지 않아도 된다는 점이다.
  - forward를 하면 gradient를 구한후 gradient descent algorithm으로 모델이 업데이트 진행된다. 실제적으로 backward할때 계산하지 않아도 pytorch에서는 작동이 된다.
- [모듈, tensorflow와의 비교, 장점](https://ko.wikipedia.org/wiki/PyTorch)

  - tensorflow 특징
  : 그래프를 먼저 만들고 한번 어떤 data를 통과시킨 다음에 static한 그래프로 바꾸기 까다롭다.
  - pytorch 특징 
  : 그때그때 layer를 추가할수 있어서 dynamic한 그래프를 만들어서 computation을 구축하고 마지막에 output을 만드는 형태이다.

- [pytorch 설치방법](https://pytorch.org/get-started/locally/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch tutorial](https://pytorch.org/tutorials)


## 5-2. TorchVision

- PyTorch project 의 일부로서 이미지를 처리하는 다양한 인터페이스(함수, 클래스…)를 제공한다.
- Numpy array (또는 PIL) 이미지의 torch.Tensor type으로의 형변환해주는 함수 (transforms.ToPILImage(), transforms.ToTensor())와, image를 자르거나 photometric jitering을 주는 등의data augmentation 기능을 제공한다.
- 또한, 대표적인 CNN model(ResNet, Dense layer, VGG 등)과 그 pre-trained weights을 쉽게 사용하도록 해준다.
- 설치: 보통 pytorch 설치할 때에 같이 설치한다.
- [TorchVision documentation](https://pytorch.org/vision/stable/index.html)
- 추가적으로 image augmentation 할 때에 사용하는 albumentations 라이브러리가 있다.  
  - torchvision에서의 augmentation이 한계가 있을수 있기에 추가 라이브러리를 활용하면 조금 더 좋다.
- [albumentation 예시](https://albumentations.ai/)
- [albumentation Documentation](https://albumentations.ai/docs)

## 5-3. Anaconda

- 아나콘다 : 패키지 (라이브러리 및 dependencies) 관리를 쉽게하기 위해 사용하는 오픈소스 패키지 관리 시스템.
- 도커나 다른 방식을 사용하기도 하지만, 보통 anaconda 를 사용한 패키지 관리가 간단하고 편리하기에 많이 사용한다.
-  Conda 와 같은 package manager를 쓰면 프로젝트마다 패키지나 버전 관리가 간단하고, 프로그래밍 그 자체에 좀 더 집중할 수 있게 도와준다.
- [설치방법](https://www.anaconda.com/products/individual)
- 위에서 installer 설치 후 
  - bash Anaconda-latest-Linux-x86_64.sh 또는 GUI 로 설치.
- 새로운 environment 만들기
```
$ conda create –n {desired name} python={desired version}
```

- Using an environment

```
 $ conda activate {desired env name} 
 $ conda deactivate {desired env name} 
```

- Remove an environment
```
conda remove –n {desired env name} --all 
```

- [anaconda More informations](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.ht)


## 5-4. pytorch 주요기능

- [About Tensor Documents](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)

### 5-4-1. pytorch 설치

```
conda install pytorch torchvision torchaudio -c pytorch
```

### 5-4-2. Tensor: creation

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/10168bf3-db99-4110-91c9-e4d81927a770)

  - Tensors
  : pytorch에서 array같은것. inputs, outputs, model parameters 모두 tensor로 표현한다.
  - numpy의 ndarray와 비슷하다.tensor는 gpu에서도 사용가능하다.

  ```
  import torch
  import numpy as np
  data = [[1,2],[3,4]]
  x_data = torch.tensor(Data) ## 2x2 tensor 생성
  x_data = torch.tensor(3.14159)  ## scalar tensor 생성
  x_data = torch.tensor([]) ## empty tensor생성
  ```

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/a3c9ab95-7074-4775-b48a-061011e49c2a)

  - From numpy arrays
  : numpay array로부터 tensor를 만들고자 할때

  - Tensor creation using APIs
```
>>> shape = (2,3,)
>>> torch.rand(shape)   ## 2x3짜리 랜덤한 tensor 생성
>>> torch.ones(shape)   ## 2x3짜리 1만 들어있는 tensor생성
>>> torch.zeros(shape)  ## 2x3짜리 o만 들어있는 tensor 생성
```
  - Tensor creation from other tensors
  : 다른 tensor로부터 같은 shape의 tensor생성도 가능

  ```
  x_ones = torch.ones_like(x_data)  ## retains the properties of x _data
  x_ones = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
  ```

  - 딥러닝할때 tensor의 shape을 아는게 중요하다. 그때 비슷한 shape을 만들고 masking하고 연산할때 주로 사용가능하다.

### 5-4-3. Tensors: attributes and GPU

: ensor type이나 처리된 device를 알고 싶을때 

- 참고자료
  - [Torch.Tensor Datatype](https://pytorch.org/docs/stable/tensors.html)

 ![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e0b85fa3-471f-4309-99fb-2318936003b8)
  
  - input tensor가 특정한 device에 들어가 있다면 operation도 해당 device GPU에서 이루어져야 한다
 
```
>>> a_tensor = torch.rand(3,4)
>>> a_tensor.to('cuda')   ## 특정 device GPU위에 올리기
>>> a_tensor.cuda()       ## = a_tensor.to('cuda')와 동일한 명령어다.
>>> a_tensor.cuda(1)    ## 1번 device의 GPU위에 올리기

>>> b_tensor = torch.rand(3,4).cuda(0)
>>> a+b   (x)         ## a는 1번, b는 0번 device의 GPU위에 올라가 있기때문에 연산이 안된다.
```

### 5-4-4. Tensors: operations

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/65a28df3-42fe-40c1-848a-d3591fc3b215)

  - indexing사용해서 numpy처럼 첫번째 열에다가 0 넣기
  - cat(concatenate)은 붙인다는 의미이다, 즉 dimension 1방향으로 concatenate(사슬같이 잇다)
  - 덧셈,뺄셈,곱하기 가능하다
  - matrix multiplication도 가능
    - @ 는 python에서 곱셈역할.


### 5-4-5. Loding data : Dataloader

- 참고자료
  - [Pytorch Dataset & Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/32ecd48d-d28c-4df7-b052-43d6b88d6b9d)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/ac3385f1-842e-4f96-88d9-e18e56b07166)

  - dataset class를 상속받으면 pytorch dataset형태로 처리가능하다.
  - dataloader는 dataset을 읽고 multi-process로 바꿔주거나, mini-batch로 처리해준다.
  - 즉 train epoch마다 iteration즉, bacth를 돌려주도록 구현한다.
  - 1epoch = dataset의 모든 example을 1번보는것
  - batch size = train dataset을 몇개의 구분으로 볼것인지.
  - 실제도 model은 GPU에서 처리되고 data는 hdd나 sdd에서 돌아간다.
  - 즉, storage에 있는 data들을 memory에 올려서 모델에서 돌아가도록 해주는 하드웨어적인 연산방법이다.


- open dataset 이용 & custom dataset
  
  - [pytorch 에서 이용가능한 open Dataset 종류 documents](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

  - custom dataset
  : loading data은 download 받은 open dataset을 우리의 입맞게 맞게 custom이 필요하다. 아래 3가지 class는 무조건 쓴다.

  ![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/c1be55ec-830c-4e0a-ac57-024ce058315c)

    - __init__(self,[]) 
    : python class initialize할때 사용
      - 데이터셋 루트 디렉토리 정보 필요
      - transform : 어떻게 해당 dataset을 변환할것인지
      - image path
    - __len(self) 
    : 데이터셋의 length(크기) 반환
      - 이걸 해주는 이유는 나중에 dataLoader에서 dataset 넣어줄때 length함수 기반으로 batch를 만들기 떄문
    - __getitem__(self,idx)
    : index 인자를 받아서 index번째 data를 return해준다. 그래서 dataloader가 이 함수를 call한다.


  -[Writing custom dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)


### 5-4-6. Build the neural network : Network
: 모델을 만들었다면 data를 가져와서 모델에 넣어줘야 한다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/3465d71e-f458-49a3-bcd5-da452e6b82cf)

  - NeuralNetwork class에 __init__과 forward함수가 있으니 사용하면 된다.
  - nn.module을 상속받아서 class를 만들면된다.
  - __init__ : torch.nn.namespace에서 상속을 받고 모델구조를 정의
    - flatten : 이미지의 special resolution이 있기 때문에 하나의 vector로 펼쳐주는 model이다.    즉, 하나의 dimension(차원)으로 펼쳐주는 역할을 한다. 
    - self.linear_relu_stack이라는 함수를 정의해서 layer를 만든다.
  - forward 함수 : logit을 만든다

### 5-4-7. Training

: 그 후 학습데이터로 logit이 특정한 class를 가르키도록 model 학습을 시키는 과정이 training이다. 그래서 training data로 neural network 모델을 학습시킬수 있다.

- 참고자료
  - [Optimizing model parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/5064e990-a09d-4f41-92f8-01b34f1980c7)

  - Define optimize : torch.optimize하면 된다.
    - adam이나 SGD를 정의해주면 된다.
  - Iterate the below lines until reaching the max_epoch
    - 4번순서를 반복하면서 train이 overfitting하지 않게 하면된다.
  - Backpropagation으로 보통 optimizer를 진행한다.
    - optimizer.zero_grad() : gradient 초기화
    - loss.backward() : 전체 backward값(gradient값)이 계산이 된다.
    - optimizer.step() : 정해진 optimizer algorithm에 따라서 모델 weight가 업데이트 된다.

  - logging
    - if batch % 100 == 0: 이하 부분 이 logging부분이다.

### 5-4-8. Save and Load Model Weights

- 참고자료
  - [Save and Load the model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/3eb3ab0a-3356-43de-ae5e-bdaaed6376f0)

  - model layer들을 모두 dictionary형태로 뽑아주는 함수이다.
  - model = models.vgg16(pretrained=True) : pretrained된 vgg16모델을 가져온것.
  - model.load_state_dict() : model의 dict에 model weight값들을 박아넣어준다는 의미
  - model.eval() : evaluation하는 함수

```
$ import torch
$ python
>>> import torch
>>> import torchvision.models as models
>>> vgg16.models.vgg16(pretrained=True)
>>> vgg16
>>> torch.save(vgg16.state_dict(),'vgg16_weights.pth')

## load하기
>>> b= models.vgg16()   ## b는 random하게 initialize된 vgg이다.
>>> b.load_state_dict(b.load('vgg16_weights.pth))
>>> b
```

### 5-4-9. Transfer learning(with torchvision model)

- 참고자료
  - [Finetuning the ConvNet](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/4a4c357f-2c77-4947-9cfb-68ea7ebb0c4f)

  - pretrain의 input feature size를 가져온다. 
  - fc는 보통 마지막 layer이다. 이것을 바꿔주자. 이것이 보통 1000개 classification하게 되어있는데 이것을 2개 classification하도록 바꾼다.  
  - 그리고 이것을 cuda device에 올려주고 loss를 정의해준다. 
  - 그후 optimize 정의해주고 scheduler 정의해주면 된다.
  - 이렇게 되면 pretrain된 모델을 2개만 classified하는 모델로 fine tuning해주는 행동을 할수 있게 된다.

# 6. Loggig, Wandb 
: logging은 중간중간 기록하는것. deep learning logging tool인 wandb를 활용해서 어떻게 디버깅할지

## 6-1. logging

- 컴퓨터비전 / 딥러닝 모델 개발 뿐 아니라 모든 프로그래밍에 있어서 logging 모듈을 잘 구현하는 것은 매우 중요.
- Python 에도 logging library 있음
- 다양한 이벤트 수준에 대해서 정의할 수 있고, logger 와 handler가 이를 적절히 처리해 줌.
- handler가 계속 log를 어떻게 가져올지 처리해준다.
- 설치: pip install logging 
- [logging 자습서 Documentation](https://docs.python.org/ko/3/howto/logging.html)


## 6-2. wandb

- Wandb (weights & Biases)는 학습 모델 개발 시에 하이퍼파라미터나,시스템 메트릭, 결과를 로깅해주는 패키지
- tensorboard/tensorboardX나 hydra(하이드라) 같은 패키지들과 비슷.
- Wandb 장점 : 웹서비스 를 통해서 다양한 모델의 결과를 확인/관리 할 수 있다는 점
  - 웹브라우저에서 단순히 실행된다는게 아니라 서버자체에서 가져와서 report도 만들어줄 수 있다.
  - sweep 이라는 기능은 autoML (grid search, Bayesian optimization 등) 을 통해 하이퍼파라미터 서치를 지원. 베스트 모델을 찾기 위해서 하이퍼파라미터 search가 필수적인데, 이때 매우 유용.
    - grid하게 다 찾아서 shell script만들어서 model parameter searhc를 하는데 이걸 자동으로 해줄수도 있다.
- 설치 : pip install wandb
- [wandb Documentation](https://docs.wandb.ai/)


## 6-3. kornia

- Kornia 는 AI를 위한 컴퓨터비전 알고리즘의 SOTA(State-of-the-art)구현체를 모아놓은 라이브러리로 비교적 최신인 2019년부터 개발되기 시작한것.
  - SOTA(State-of-the-art) : 사전학습된 신경망들 중 현재 최고 수준의 신경망, 현재 최고 수준의 결과
- 특히 딥러닝 기반의 컴퓨터비전 모델을 연구/개발할 때 유용
- 구현하기에 까다로운 보조함수들이 구현되어 있고, GPU 기반의 deep learning 프레임워크인 PyTorch와도 호환가능
- 설치: pip install kornia
- [kornia Documentation](https://kornia.readthedocs.io/en/latest/)
- [kornia Github](https://github.com/kornia/kornia)


---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}