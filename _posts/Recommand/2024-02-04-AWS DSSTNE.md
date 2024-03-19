---
title: "AWS DSSTNE"
escerpt: "AWS DSSTNE"

categories:
  - Recommand
tags:
  - [AI, Recommand, DSSTNE]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2024-02-04
last_modified_at: 2024-02-04

comments: true
  

---


## 1. AWS DSSTNE(데스티니)
DSSTNE : Deep Scalable Sparse Tensor Neural Engine

- 추천 시스템을 훈련할 때 갖는 평가 데이터 같은 희소 데이터를 사용한 딥 뉴럴 네트워크를 설정가능(코드 필요없음)
- 단순히 훈련 데이터를 DSSTNE expects 포맷으로 바꾸고 원하는 신경망의 위상과 최적화 방법을 정의하는 짧은 환경설정 파일을 쓰면 나머지는 알아서 된다.
- DSSTNE는 GPU에서도 실행되며 방대한 양의 데이터를 아주 빨리 처리할 수 있음.
- 컴퓨터가 다중 GPU를 갖고 있다면 GPU를 가로지르는 로드를 분배하고 병렬로 추천을 계산 가능.
- 그리고 규모를 더 늘리고 싶다면 DSSTNE와 Appache Spark를 통합해 더 큰 클러스터를 통해 실행하게 할 수 있음.
  

### 1-1. DSSTNE 사용예시) 클러스터 규모 어떻게 늘릴수 있나?
- ex) movieLens dataset 이용하여 3층 신경망을 훈련하는 config파일 

```
{
  "Version" : 0.7,
  "Name" : "AE",
  "Kind" : "FeedForward",
  "SparsenessPenalty" : {
    "p" : 0.5,
    "beta" : 2.0
  },

  "Shufflelndices" : false,

  "Denoising" : {
    "p" : 0.2
  },

  "ScaledMarginalCrossEntropy" : {
    "oneTarget" : 1.0,
    "zeroTarget" : 0.0,
    "oneScale" : 1.0,
    "zeroScale" : 1.0
  },

  "Layers" : [
    {"Name" : "Input", "Kind":"Input", "N":"auto","DataSet":"gl_input", "Sparse":true},
    {"Name" : "Hidden", "Kind":"Hidden","Type":"FullyConnected", "N":128,"Activation":"Sigmoid", "Sparse":true},
    {"Name" : "Output", "Kind":"Hidden","Type":"FullyConnected","DataSet":"gl_output", "N":"auto","Activation":"Sigmoid", "Sparse":true},
    
  ],

  "ErrorFunction" : "ScaledMarginalCrossEntropy"
}
```

  - 피드 포워드 오토인코더를 만들 건데 구체적으로는 희소 오토인코더라고도 하고 여기에서, '희소'는 은닉 노드 수가 많더라도, 흥미로운 패턴을 찾도록 강요하려고 은닉층에 제한을 두는 것. 
  - SparsenessPenalty의 값은 그런 제한을 정의하는데 이런 값이 나타내는 세부 사항은 **Andrew Ng의 CS294A 강의**에 나오는 강의 노트에 있다.(희소 오토인코더에 대한 스탠포드 대학 강의)
  
  - Shufflelndices : false
    - 지수를 바꾸고 싶지 않다고 명시
  - Denoing : p:0.2
    - 20%의 가능성에 대해서 잡음제거 적용하여, 임의로 한 입력을 0으로 뒤집는다.
    - 은닉층이 일대일 관계에 모이게 하는 것 대신 더 견고한 특징을 발견하도록 오류가 생긴 버전에서 입력을 재건해서 강요.
    - 이렇게 하면 신경망이 더 열심히 일하도록 강요하면 더 나은 결과를 얻을 수 있다.(그것으로부터 정보를 숨기는 것이다.)
  
  - ScaledMarginalCrossEntropy 
    - 손실 함수에 대한 파라미터를 정의
    - 실제로 기본값은 0.9, 0.1, 1.0, 1.0이다.

  - Layers
    - 1층
      - Name : Input
        - 입력층
      - Sparse : true
        - 희소적이라고 명시.
        - 이걸로 DSSTNE가 특별해지고, 대부분의 입력 데이터가 결측된 경우를 위해 만들어진다. 
        - 입력은 이전에 봤고 각 사용자에게 네트워크를 훈련하는 RBM 예시와 더 비슷. 
      - Kind : input
        - 제품을 사용자가 평가하면 하나가 정해지지만 대다수 제품이 주어진 사용자에 의해 평가되진 않는다. 
        - 이건 입력 데이터가 아주 희소적이라는 걸 의미하고 이 층을 희소적이라고 명시하면서 데스티니는 이걸 알고 효율적으로 다룰 수 있다. 
      - N : auto
        - Node를 의미
        - 데이터 파일에 입력이 있는 만큼 많은 입력을 설정하겠다는 뜻. 
      - Dataset : gl_input
        - 그 데이터 파일은 gl_input이라는 파일에서 주어진다.
    - 2층
      - Name : Hidden
        - 은닉층
      - N : 128, Activation : Sigmoid, Sparse : ture
        - 128개의 노드와 시그모이드 활성화 함수로 완전히 연결되며 희소적이다.
    - 3층
      - Name : Output
        - 출력층
      - Dataset : gl_output , N : auto
        - 훈련 목적으로 gl_output 파일에 답을 주고 Node를 자동으로 설정해 예측하려는 gl_output 내 특징의 구조에 적응할 수 있게 만든다.
      - Activation : sigmoid, 
        - 시그모이드 활성화 함수가 있는데 일반적으로 이진 입력에 사용하는 것. 
    - 해당 구조가 이진 입력과 출력을 가정하는 것처럼 보이지만 MovieLens 평가 데이터는 1부터 5까지의 단계로 존재.

  - 이 예시에서 결국 실제 평가값 자체를 버리고 영화의 흥미에 대한 암묵적인 이진법 표기로 어떤 평가도 받아들이고 잘 작동함.
  - RBM으로 한 것과 같은 기술을 쓰고 개별 이진 분류로서  더 정교하게 다듬어진 결과와 예측을 얻기 위해 각 평가값을 만들 수 있다.

## 2. amazon DSSTNE
[github - amazon DSSTNE](https://github.com/amazon-archives/amazon-dsstne)

- example 따라하기
  - [example 따라하기](https://github.com/amazon-archives/amazon-dsstne/blob/master/docs/getting_started/examples.md)


  - move to directory(MovieLens dataset directory)
  - download 2천만 평가데이터set
  - 실제로 필요한 건 평가 데이터뿐이라서 zip 아카이브에서 추출하고 ml-20m_ratings.csv로 저장
  - DSSTNE는 CSV 파일로 처리할 수 없고 라인 당 평가 하나로 조직된 데이터를 원하지도 않기 때문에,NetCDF 형식의 파일 형식이 필요하고 파일을 변환할 수 있는 도구를 제공
  - 해당 변환 도구에도 요구 조건이 있기 때문에 평가 데이터를 NetCDF로 변환 가능한 형식으로 바꿔야 함.

    - 첫 단계는 데이터 배열이 필요하기 때문에, 모든 열이 단일 사용자를 나타내고 그 사용자에 관한 모든 평가 입력 목록이 뒤따른다.
    - 우리가 원하는 건 사용자 ID를 포함된 각 라인인데 탭과 콜론으로 구분된 사용자가 평가한 각 제품 ID에 관한 항목이 뒤따른다.
    - 기본적으로 입력층의 희소 표현인데 각 개별 사용자에 관한 것이고 라인 당 한 사용자가 해당된다.

  - amazon은 CSV 파일을 중간 형식으로 바꾸기 위한 AWK 스크립트를 제공한다.
  ```
  $ cat convert_ratings.awk
  ``` 
    - 주어진 사용자에 관한 모든 평가 데이터가 기존 무비렌즈 데이터에 함께 그룹화된다는 점을 활용.
    - 각 사용자에 대한 모든 평가를 수집하고 완료되면 각 사용자에 대한 라인이 표시됨.
  - convert 완료하자.
  ```
  $ awk -f convert_ratings.awk ml-20m_ratings.csv > ml-20m_ratings
  $ head ml-20m_ratings
  ```
    - 출력을 보면, 훈련 데이터가 포함된 걸 볼 수 있다.
    - 한 라인 당 한 명의 사용자고 희소 형식에서 사용자가 평가한 제품도 있다.

  - 이제, 이걸 DSSTNE가 요구하는 NetCDF 형식으로 바꾸자.
    - 3개의 파일을 만드는데 NetCDF 파일 자체와 각 뉴런의 표시가 있는 인덱스 파일 모든 샘플의 표시가 있는 인덱스 파일이다.

  - 입력 파일을 위해 만든 것과 같은 샘플 표시를 써서 출력층의 데이터 파일을 만들자.

  - 이제 준비는 끝!

- config.json
  - 신경망의 위상과 훈련 방법을 정의하는 파일

- train

```
$ train -c config.json -i gl_input.nc -o gl_output.nc -n gl.nc -b 256 -e 10

```
  - 배치 사이즈 256과 10 에포크를 나타내는 파라미터로 train진행

- predict

```
$ predict -b 256 -d gl -i features_input -o features_output -k 10 -n gl.nc -f ml-20m_ratings -s recs -r ml-20m_ratings
```
  -  10은 상위 10개 추천을 원한다는 걸 의미하고 gl.nc 파일의 모델을 통과한다.
  - 예측 명령이 자동으로 필터링도 한다는 점이 check point이다.
  - 이렇게 사용자가 이미 평가한 제품을 거를 수 있다.

- 결과확인

```
$ head recs
```

  - 결과는 recs 파일에 있으니 확인해보자.
    - 각 라인은 사용자 ID이고 뒤따라 추천된 제품 리스트와 점수가 나온다.
    - 인간이 읽을 수는 없지만 부분 점검을 몇 개 해보자.
    - 엑셀에서 movies.csv 파일을 열었고 이렇게 추천된 제품 ID를 살펴볼 수 있는데 ID에 대응되는 영화도 볼 수 있다.
    - 사용자 1에 대한 영화 몇 개를 살펴보자. 1196, 4993 등이 나오는데 이는 엑셀에서 보면 스타워즈, 반지의 제왕영화네. 
    - 사용자 1은 공상 과학과 판타지 팬으로서 역대 최고의 공상 과학 및 판타지 영화 몇 개를 추천했음을 알수있다.

- 정리) 딥러닝을 추천시스템에 적용하는 목적으로, 3 layer을 통해 2천만 평가dataset을 GPU를 활용하여 신경망을 가속화하여 train진행. 

## 3. scaling up DSSTNE 
하나의 GPU가 감당할 수 없을때까지 규모를 더 늘리고 싶다면?

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/d77a1d6c-4ade-49e9-bbbd-7f72b48590e8)

- [learning : scaling up aws-DSSTNE](https://aws.amazon.com/ko/blogs/big-data/generating-recommendations-at-amazon-scale-with-apache-spark-and-amazon-dsstne/)
  - AWS 블로그에서, 아파치 스파크와 아마존 데스티니로 '아마존 스케일의 추천 생성'이라는 기사를 찾아보자.

  - DSSTNE + Apache Spark 결합
    - idea)
      - Spark로 데이터 분석과 처리 작업을 분배.
      - Spark Executor Node의 CPU에서 한 것처럼 NetCDF 형식으로 데이터를 전환하는 것.
      - **Spark를 통해 데이터가 분할되면 GPU 컴퓨팅을 맡는 다른 클러스터로 넘겨지고 거기서 DSSTNE가 데이터의 신경망을 훈련시킴**.
      - Spark는 GPU의 클러스터를 관리할 수 없어서, 서로 다른 두 개의 클러스터를 쓴다.
        1) 데이터 랭글링하고 분할하는 Spark Cluster
        2) Amazon container service인 ECS로 관리되는 또 다른 GPU cluster
      - 아마존 S3로 두 Cluster 간의 데이터를 전송
        - Spark는 CPU 클러스터 /  ECS는 GPU 클러스터이다.
      - Spark는 데이터를 앞뒤로 전송하고자 S3를 써서 전체를 조직한다. 

  - 특정한 GPU 슬레이브에서 Amazon-DSSTNE는 데이터를 처리하기 위해 실행할 수 있는 몇 가지 기술 중 하나일 뿐이다.
  - 아마존 웹 서비스가 필요하면 이런 노드를 자동으로 할당하고 할당 해제할 오토 스케일링 기능이 있다는 점이 좋은점이다.


  
---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}