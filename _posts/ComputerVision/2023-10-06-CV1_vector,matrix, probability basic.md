---
title: "CV1 : vector,matrix, probability basic"
escerpt: "Computer Vision 기본이론 1"

categories:
  - Vision
tags:
  - [AI, Vision, vector,matrix, probability basic]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-10-06
last_modified_at: 2023-10-06

comments: true
 

---
# 1. Preliminary
- computer vision 
: 인간의 시각적 정보를 기계도 인지할수 있게 하는것.
- 자율주행
  - 카메라 
  : object detection 진행을 통한 obejct 인지
  - LIDAR 
  : 3D point cloud data 획득
  - RADAR 
  : 거리측정
- depth camera 
: 이미지를 받아들였을때 깊이정보까지 받아들여, 3차원정보를 획득가능.
- application
  - visual Localization and SLAM 
  : camera pose와 위치 localization 해줄수 있으며 3차원까지 복원할수 있음
  - Face Detection and Recognition 
  : 사람 얼굴 찍었을때 사람이 어디있으며 그사람이 누구인지 알수있음.
  - Medical Image Analysis 
  : 의료영상분석 
  - Human Pose Estimation 
  : 사람 포즈가 있을때 사람의 keypoint들을 추출하고 keypoint들을 통해서 3D mesh를 반환해줄수 있다. 이러한 정보를 통해 사람에게 옷을 입힐수 있고 가상세계에 사람을 rendering 할 수도 있다.   
  - Human Computer Interaction 
  : 인간의 움직임이 컴퓨터에도 적용되는것.
  - Autonomous Vehicles 
  : 차량에 여러 카메라를 달아서 그것을 통해 여러 데이터를 분석 후 차량의 decision을 도와주는것.
  - Image Synthesis 
  : (신세시스) 이미지 2장을 합성하는것
  - Image Generation(Style Transfer) 
  : 실제 카메라로 촬영한 이미지가 있을때 고흐나 몽크같은 스타일을 입혀서 스타일을 변환시켜주는것.

- Related research field 
: 관련된 연구분야 
  - marchine learning 
  : 머신러닝에서의 알고리즘들을 컴퓨터비전에서 많이사용한다. 반대도 마찬가지!
  - robotics
  - human computer interaction
  - computer graphics 
  : 이미지를 만들어내는 또는 3차원정보를 복원해내는 task
  - image processing
  - Multimedia
  - Algorithm
  - Pattern recognition

- Topics
  - visual recognition 
  : 이미지에 어떤 object가 있는지 분류 + 어디 위치에 있는지 + pixel 단위의 classification
  
  ![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/344a4253-5a87-4fc3-96ad-cc1847c4818b)

  - Video Recognition 

  ![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/fa33607d-7d0d-42fc-8a3a-0d980018729c)

    - optical flow(motion Estimation) 
    : 움직이는 비디오 상에서 각 pixel들이 어디로 이동하는지 
    - video action recognition 
    :  실시간으로 사람들이 무슨 행동을 하는지
    - object tracking 
    : 사람한명에 대해서 bbox를 frame별로 이동하는것.

- State-of-the-Art in Artificial intelligence 
: 이론과 실습 그리고 그런 알고리즘들이 컴퓨터 비전에서 어떻게 적용되는지 

- Cameras and 3D geometry 
: 카메라는 기하학적인 정보를 활용해서 많이 연구되고 공부됨. 카메라에 대한 이해, 카메라를 acquisition하는것. sensing하는 방법, 그것을 preprocessing하는 방법, 그 정보들을 이용해서 3차원적인 기하학적 정보를 어떻게 복원하는지, 그것을 통해 3차원이미지를 어떻게 이해할지

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/70d43580-535a-4c5b-8c0e-7c5f3dd95dd2)


  - lens = cornea(각막)
  - aperture(조리개) = pupil(홍채,동공)
  - lens = lens(망막)
  - 각막에서 빛의 각도를 굴절시켜, 망막에서 해당 빛의 초점을 맞추는곳!


# 2. Linear algebra basic: vector and Matrix 
: vector와 matrix는 이미지의 기본적인 structure이다. 이미지도 결국 rgb와 wbg를 가진 matrix의 tensor의 연속이다.그것을 사용해서 어떻게 연산할지

## 2-1. Linear Algebra in Computer Vision
: 벡터와 행렬의 연산만 알아도 된다. 

- 참고자료
  - [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

### 2-1-1. Image description(Vector)
: **vector space**를 수학적으로 보자.이미지에 어떤 feature가 있는지를 분석한다.(Image description)


![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/5302b79f-2bca-4da1-aee1-5543efff22d9)

  - 고전적으로는 image의 x,y축의 gradient를 구한 후 그것의 orientation(방향성)을 regression해서 voting해서 histogram을 만드는 방식이었다. 즉, histogram을 vector로서 사용을 하고 local feature들을 전체 aggregation(집합)하여 image 전체의 description을 구하거나, local descriptor를 사용하여 matching or object detection을 이용하여 local한 정보를 얻어내는것도 할수 있다.

### 2-1-2. Decision making(function)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/c191cad7-a258-4724-a339-33acfa108fc7)

  - input x(image data) 가 들어갔을때 어떠한 output이 나오는 함수가 있는데 그 함수가 무엇인지 아는게 우리의 목표이다.
  - 이러한 수학적인 개념을 컴퓨터비전에 적용시켜보자.
  - input x인 이미지에 대해서 output인 사람인지 아닌지에 대해서 classifier하는 함수가 있고 그 함수를 학습하거나 디자인을 하는, decision making을 하기 위한 function을 찾는일을 컴퓨터 비전에서 하게 된다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/480bf9fe-6127-4870-b8ec-2570d9c4981b)

  - 그래서 decision boundary를 찾으면 사람인지 아닌지를 판단하는 함수 f(x)를 찾는게 컴퓨터비전의 목표이다.

### 2-1-3. vector space의 수학적 개념에 대한 정리

- Frequently used in
  - Image description
  : vector space라는 개념은 image description에 사용함
  - Computation of similarities and distances
  : similarities나 distances를 계산하는데 (유사도와 거리를 알아야 비슷한것끼리 clustering할수도 있고 decision boundary를 통해 classifier할수도 있기 때문에) 사용한다.
  - Finding algebraic solutions
  : vector space를 이용해서 algebraic solution(대수적정답)을 찾는데도 사용가능하다  
  - Transformations
  : geometry적인 transformation(변환)에 대한 정보를 표현하는데도 사용된다.
  - Optimization
  : Optimization(최적화)에도 사용된다.

## 2-2. Vector
: vector란 크기와 방향성을 가진 geometric object라고 말할수있다

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/734042e7-48d0-4738-9c78-6d64ff35aa4b)

  - notation(표현법,표기법) 
  : x가 R인 실수범위에 n개의 채널을 가지는 놈이다. x라는 벡터는 x1,x2,xn까지의 value를 가지는 한 일차원 길이의 list이다.

  - Transpose는 row로 표현가능하다.
  - Magnitude(규모,중요도)
  : 각 요소들에 제곱하여 더한후 루트씌우고 크기측정을 하면 된다.
    - 0벡터에서 x벡터까지 갈때  거리는 각각의 값들을 제곱하여 더한후 루트 씌우면된다.

### 2-2-1. Operations
: 내적과 외적의 차이는 내적은 x에 transpose가 걸려서 각각의 위치에 대해서 곱해져서 하나의 scala value가 나온다. 외적은 matrix value가 나온다.

- inner product(내적)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/c31b341d-adeb-4259-b52b-5f2639929808)
  
  - 대칭적, 분배, 선형성보장,  내적된 결과는 항상 0보다 크다.

- outer product(외적)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e8fad84d-5e12-4b2d-9332-907609720663)

  - 외적을 하게 되면 matrix가 나오게 된다.

### 2-2-2. Vector Norm
: vector의 length를 말한다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/04f4c1ff-5026-4958-89f1-3a204e261bbc)

  - 2-norm = Magnitude와 같은 의미.
  - p-norm : element에 대해서 x에 p를 제곱한다음에 더한 후 전체를 1/p 하는것

|norm|그래프에서의 특징|
|---|---|
|1-norm|optimization할 때 뾰족해서 sparsity(희박,희소)를 보장해주게 된다.|
|2-norm|원점으로부터 모든 거리가 같은 원|
|infinite-norm|가장큰값을 가지고 오게 되는 사각형좌표|


![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/edc175b8-5ac2-4407-99d7-829c9f45274b)


### 2-2-3. Lienar Dependency
: 벡터의 성질중 linear dependecy라는 성질도 있다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/bccf91e8-34a9-4528-bbce-81993ab786b0)

  - linearly dependent : x2를 x1과 x3의 덧셈과 스칼라곱으로 표현할수 있는경우.
  - linearly independent :  x1,x2,x3가 직교하고 있을때는 아무리 더해도 벡터가 x를 각각 만들수 없는 경우.

### 2-2-4. Basis(베이시스)
: 서로 직교하는 linear independant한 vector들로 구성된 set을 basis set이라고 한다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/2e170e23-516c-45f9-ab9e-92d6a1137f1c)

  - 각 dimension의 위치에 대해서  1을 넣은 벡터가 있을때 공간의 base를 다 표현할수 있다.
  - 예시) 5,2,3 이라는 벡터의 transpose값이 있다면  각 basis에 scalar value를 곱한형태로 표현할수 있다
  - projection(투사,투영)
    - 각 vector의 basis에 대해서 x의 transpose를 곱해주면 해당 위치로 vector가 투사되어 사용가능하다.
  - 각 basis끼리는 Orthogonal(올소고날) : 직각이다.
    - e1과 e2를 내적(즉, 곱한다면) 하면 직각이므로 0이 나온다.
  - 서로다른 basis에 대해서 transpose한걸 곱하면 1이 나온다.


## 2-3. Matrix
: 숫자들의 직사각형의 array형태

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/770a31bb-276e-412a-87db-bd87908980d3)

  - Identity matrix: 대각선의 값들이 모두 1이고 나머지는 0인경우

### 2-3-1. Matrix Oprations

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/aece6d22-da4b-40c6-b2c4-d670de4c6843)

  - Transpose : 행 vector <-> 열 vector change
    - 행,열 바꿔주기역할이다.

### 2-3-2. Rank
: 행렬에서 선형독립적인 row or columns 의 수 = basis의 개수

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/434cb523-52c7-46e5-8a0d-5cc101f32bb3)

  - 예시
    - (-1,2)에 -4를 곱하면 (4,-8)이 되므로 결국 하나의 vector만존재. 즉 1개의 basis만 존재
  - rank의 여러 속성(성질)
    - 행(m)과열(n)의 짧은것보다 X의 rank는 클수없다. 
  - square marix가 있다면 보통 vector가 basis로 변환될수 있기 때문에 특이성이 없는게 된다(=non-singular)
  - rank가 하나라도 부족하다면 특이행렬이다.
    - 3x4 matrix인데 rank=2라면 그것은 특이행렬(=singular)
    - singular matrix는 추후 optimization할 때 solution을 구할수 없기 때문에 이 컨셉은 기억해둘것.

- Square Matrix and Linear Equations
: 선형적인 수식이 있다면 정방향의 square matrix로 표현가능하다. 

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/31962135-492f-4793-9eac-bf9738cd820e)

  - 1) 양변에 inverse matrix를 곱해준다.
  - 2) determinant : system이 unique한 solution을 가진지 아닌지를 결정해준다.


### 2-3-3. Determinant
: square matrix에 solution이 있느냐 없느냐를 결정해주는 결정값.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/a1fa75bb-07c1-4460-96c8-05c1b40bb04a)

  - singular하다면 determinant=0 이다. 반대도 성립가능.(왜? 분모에 0값이 들어가면 값을 구할수 없기 때문에.)
  - python에서는 numpy에서 linear algebra 라이브러리 사용하면 된다.

### 2-3-4. Inverse Matrix
: 2가지 조건을 가져야한다. squre(NxN 형태)형태 + non-sinular matrix여야 한다.(모든 basis값이 matrix와 같아야 한다.)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/92f46896-9aa7-4018-a728-a209f2a41996)

  - diagonal matrix D : 나머지값이 0이고 대각선값만 D로 채워져 있는경우
  - Orthonormal matrix : 첫번째 column과 두번쨰 column을 각각 곱하면 0이 나오고 각각의 크기는 1이 나온다.
    - Rotation matrix(회전변환) : 첫번째 column과 두번째 column을 각각 곱하면 0이 나오고 column끼리의 크기는 1인 matrix

- Determinant, Inverse Matrix, and Linear Equations
: 각각의 관계에 대해 알아보자.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/2891950e-0b1e-4664-92a6-eff46c43adab)

  - ad-bc =0 이면 정답을 구할수 없다.
  - singular vector : 2개의 벡터로 이루어진것을 말한다.
  - non-singular : scalar value로 서로를 표현할수 없기 때문에 x,y축으로 2개의 basis로 나누어 진다. 즉, ad-bc=넓이가 determinant value가 된다.

## 2-4. Eigen Decomposition
: 앞의 개념들을 종합하여 나온 개념, equation(방정식)의 solution을 쉽게 구하기 위해서!!

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/07f5105b-5c2c-4eb1-9ea3-b219e0f01b13)

  - eigenvector : 고유벡터(x)
  - eigenvalue : 고유값(람다)

  - suqare matrix A가 있다면 (N x N행렬) 람다는 어떤 상수이고, non-zero vector x가 있다면, Ax = 람다x가 성립되면 람다를 eigenvalue라 하고 x를 eigenvector라고 한다.(A는 matrix형태겠군.)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/fe5e516c-4ce2-4942-a9ac-93de39c07031)

  - V : eigenvector들을 가지고 있는 orthonormal matrix
    - 각각은 서로 직교(orthonormal)하고 eigenvector들로 이루어져있다
  - D : diagonal matrix (대각선만있는 매트릭스)
  - Eigen 성질
    - eigenvalue에 하나라도 0이 있으면 square matrix는 singular이다.
    - 대각행렬이 모두 positive value면 A의 matrix는 positive로 정의됨.반대면 모두 negative이다.
  
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d47539c7-0cb5-4416-b695-f34aeff66633)
  
  - eigen compute를 왜 하느냐?
    : eigenvalue과 eigenvector만 이용해서 solution을 바로 구할수 있기 때문에.
     



# 3. Probability basic : Random variable nad Bayes 
:  대부분의 머신러닝들을 확률기반이다. 알수없는 사건에 대해서 판단 하는데 필요함

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/94b19a82-18b9-4231-8e00-b35813b143fe)

  - 컴퓨터비전에서 확률이란 왜쓰일까?
    - visual sensing했을때 대부분이 noise와 uncertain난다.즉 카메라로 어떤 화면을 찍었을때 조도변화나 카메라의 흔들림등 다양한 이유때문에 noise가 많이 낄수 있고 불확실성을 가지고 있다.
    - 그러므로 prababilistic model은 가장 효과적으로 불확실성을 다루기에 가장 좋다. 왜냐하면 불확실성에 대해서도 계산하여 수치적으로 정량화 시킬수 있기 때문이다.

## 3-1. Random variable
: 그래서 확률이론에 가장 기초가 되는 representation 개념중에 random variable(확률변수)이란 개념이 있다.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/66e81a4f-c334-49ee-af70-60b2458d4da0)

  - sample space(오메가) : 실험에서 사건이 일어날수 있는 확률
  - random variable : 값을 할당해주는 함수, random variable은 확률이 아니라 변수일뿐이다.사건을 수치화시켜주는 함수.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/02ccef19-1af8-47c7-93e2-e320171445d8)

  - Event space : 일어날수 있는 모든 가능성
  - Probability measure
  : 모든 확률의 합은 1이다. 

## 3-2. probability의 중요한 특성

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/bc0a18d8-2c4c-489c-8485-6711f4fa2bf6)

  - Set inclusion
  : A가 B에 속하는 사건이라면 A의 확률은 B의 확률보다 무조건 작거나같다.
  - Complement (여집합)
  - Notation(교집합)
  - independence(독립)

## 3-3. Conditional Probability
: 조건확률

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/0537cf0d-617e-413e-8c86-6a70441ffd4c)

- B가 주어졌을때 A가 일어날 확률
  - 비가 오고 해가뜰확률 = 해뜰확률 x 해가 떠있는데 비가 올 확률

## 3-4. Bayes'Theorem(베이즈 이론)

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/96a47b74-2a1b-48d9-9ef9-6730baa5868b)

  - P(A,B) = A,B가 같이 일어날 확률
  - 조건확률이 아닌 Bayes 이론으로 풀어낼수 있다.
    - **사후확률 ∝ 가능성 x 사전확률**

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/bded5284-cb03-4503-b409-d3870b2bc316)

  - likehood : 사과를 먹었을 확률
  - posterior : 즉 먹었을때 사과일 확률

## 3-5. Gaussian Distributions

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/52f57d8e-c1d2-4c67-8f4c-90f03751e603)

  - univariate Gaussian distribution : 변수가 하나밖에 없는 가우스분포
    - 확률변수 x가 있을때; 평균(μ,뮤)과 분산(시그마 제곱)으로 분포를 표현할수 있다.
  - 가우시안 분포를 2차원 평면에 plotting한것(그래프해석)
    - 평균값이 그래프의 중간값이다. μ의 value에 따라 위치가 바뀐다.
    - 시그마는 그래프의 너비, 즉 분포된 정도를 말한다.
      - 분산이 작으면 뾰족한 그래프, 크면 넓은 그래프가 나온다.
    - 확률을 distribution하여 모델링 할수 있다

  - 이를 확장하면 multivariate Gaussian distribution이 된다.
  : 여러 변수를 넣고 gaussian distribution 한것.
    - covariance(공분산) : 두개의 확률 변수가 있을 때 이들 상호간의 분산을 나타낸 값. 이를 관측단위에 관계도를 표현한것이며, 정확도를 나타내는 척도로서는 공분산을 표준편차하여 두 변수간의 상관관계를 나타내는 척도인 correlation coefficient(상관계수)가 사용된다.

## 3-6. Gaussian Distribution을 사용하는 이유?

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d5287352-632e-4f5a-9bde-69f66093b65d)

  - 알아야할 parameter가 2개만 있으면 되니 편리하니깐 쓴다고 생각해라.
  - Gaussian Distribution으로 사건을 modeling하려면 평균과 분산만 알면 된다. 또는 변수가 많다면 covariance matrix와 μ의 벡터만 알면 된다.각 인자들의 나올 평균값과 얼마나 분포되어있는지만 알면된다. 그것만 예측하면 되기에 편하다!

  - 더 정확한 modeling을 하고 싶다면 ?
    - Gaussian x Gaussian = Gaussian 성질이 나온다.
    - 이렇게 되면 좋은점은 Bayes 이론에서 처럼 linklihood x prior 곱으로서 posterior를 구할수 있기 때문에 Gaussian의 곱으로 posterior를 구할수 있다 self-conjugate(self로 활용시키다.)

---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}