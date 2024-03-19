---
title: "Deep Factorization Machines"
escerpt: "Deep Factorization Machines"

categories:
  - Recommand
tags:
  - [AI, Recommand, TFRS, DFS]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2024-02-03
last_modified_at: 2024-02-03

comments: true
  

---


## 1. Deep Factorization Machines

- 행렬인수분해는 SVD(Singular Value Decomposition, 특이값분해) 를 이용한다. 이는 범주형 데이터(숫자로 측정하고 표시하는것이 불가능한 자료) 의 관계에 대해서도 파악가능하다.

- Factorization Machines Model(FM)은 SVD보다 범용성이 높고 SVD가 발견못하는 특성간의 관계를 찾아낼수 있다.

### 1-1. Deep Factorization Machines Model(DeepFM)
- Factorization Machines과 deep layer을 결합하여 더욱 강력한 추천시스템을 만드는게 목표.
  - 혼합형 접근법은 독립적 Factorization Machines이나 신경망보다 성능에서 앞서지만 차이가 크진 않다.

  - [논문 DeepFM: Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
    - 하얼빈 공대 팀
    - idea : DeepFM은 **Factorization-Machine과 심층 신경망의 특징을 결합**
      - 1) **인수분해 머신**은 고차원 특성 상호작용을 밝혀낼 수 있지만 **저차원 feature** 상호작용에 더 능하다.
      - 2) 반면 **심층 신경망**은 **고차원 feature** 상호작용에 더 뛰어나고 DeepFM은 양쪽의 장점을 모두 취하는 것.

#### 1-1-1. 고차원, 저차원 특성 상호작용의 뜻은 뭘까?
- ex) 앱 스토어에서 사용자에게 다운로드할 앱을 추천해 주는 추천 시스템을 만든다고 가정.(논문예시)
  - 음식 배달 앱이라면 이른 저녁인 식사시간데 인기가 있을것이다.
  1. 즉 app category + time 을 feature로 interactions을 보자.
    - 여기서 특성들의 차수는 2이고 배달 앱이라는 앱 카테고리와 사용자의 시간대 사이에 잠재적 관계가 존재하는데 이 특성들을 분리해 주는 사용자 데이터의 인수분해가 효과적일것이다.
  2. app category + gender + age 를 feature로 보자.
    - 10대 남자아이의 경우를 생각해 보면 데이터는 남자이자 10대라는 두 사실로부터 1인칭 슈팅게임을 선호한다는 식의 잠재적 관계를 시사하고 있을수도 있다.
    - 여기선 차수 3의 특성 상호작용이 있는데 각 특성은 1인칭 슈팅게임이란 앱 카테고리와 사용자의 성별과 나이이다.
  - 배달 앱 예시는 인수분해 행렬이 잘 적용되는 부분이고 슈팅게임의 경우 딥러닝 네트워크가 더 잘할 것이다.
  - DeepFM은 두 접근법을 결합하는데 저차원과 고차원 모두에서 좋은 결과를 얻을 수 있다.
  - 주의) 추천 알고리즘은 10대 남자란 이유로 슈팅게임을 추천하는 알고리즘이 일부에겐 불만일수 있음. 즉, 훈련받은 데이터에 내재하는 편견을 유지하거나 심지어는 강화 가능한것들을 주의할것.

#### 1-1-2. Architecture
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/069ce7d7-caf6-4731-8f7a-77e4a5a8faab)

  - 혼합형 방식
  - 훈련 세트의 특성 데이터가 인수분해 머신과 심층 신경망에 병렬적으로 제공됨
  - 여기선 은닉층이 둘이고 인수분해 머신과 심층 신경망의 아웃풋이 결합하여 클릭이나 구매를 비롯한 행동을 예측하는 최종 sigmoid function에 전달됨.
  - 인수분해 머신은 넓기에 저차원의 특성 관계를 잘 규명하는데 심층 신경망은 고차원의 관계를 잘 찾아내기 때문에 그 둘을 합하면 더 낫다
  - 하지만 나은 정도는 데이터의 성격과 특성 간에 실제로 어떤 잠재 관계가 존재하는지에 달려있기 때문에 너무 복잡한 접근법 같아 보이지만 이는 추천을 위한 앙상블 접근법의 한 예일뿐. 각자 강력한 서로 다른 알고리즘을(인수분해 머신 + 심층 신경망) 결합한다는 것이 point!

#### 1-1-3. ensemble approach(앙상블 시스템)

- 잘 디자인하면 복잡도 제한할수 있는데 구성 요소 시스템들을 독립적이고 병렬적으로 구동하고 결과를 마지막에 합치는 식으로.
- 한 구성 요소가 꼬이게 되더라도 그것만 정지시키고 다른 구성 요소들로 시스템을 돌릴 수 있음.

- 넷플릭스는 SVD와 RBM의 결합이 매우 효과적임을 알아냈음
- DeepFM은 그런 방식을 더 강화한 것.
  1. SVD를 좀 더 일반적인 인수분해 머신 모델로 대체
  2. RBM을 더 일반적인 심층 신경망으로 대체함.


## 2. word2vec 적용

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/8d1adcf5-454a-42d7-a952-1a696038b6a5)

  - word2vec이라는 신경망 모델을 추천 시스템에 적용
  - 추천에 활용하고자 하는 사용자의 히스토리나 맥락을 가리키는 단어들로부터 시작
    - ex) 누군가의 문장을 완성하는 문제가 있다면, 해당 문장의 모든 단어를 인코딩해서 embedding layer와 hidden layer로 통과시켜 학습이 일어나게 한 후 hidden layer의 output에 대해 softmax 분류를 실행해서 각 단어가 인풋으로 제공된 단어들과 연관될 확률을 구하는 것.
      - 여기서는 'to boldly go where no one has' 라는 단어들과 연관된 시그널들을 인풋으로 하고 잘 훈련된 word2vec 네트워크는 'gone' 이란 단어를 가장 연관된 단어로 분류함.
  - 이 테크닉을 추천 시스템에 적용하게 되면,다른 식당들의 메뉴에 근거해서 식당 주인에게 메뉴 아이템을 추천. word2vec을 많은 종류의 메뉴들과 메뉴에 등장하는 단어들로 훈련가능. 그 후 특정 식당의 메뉴의 단어들을 인풋으로 넣으면 그 식당이 메뉴에 올리려는 음식과 어울리는 단어를 추천가능

  - 해당 문제를 순환 신경망인 RNN을 적용도 가능. 

## 2-1. extending word2vec
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/49648931-4ef6-4d3e-ad0b-769dd114eff5)

- word2vec을 단어 외에 적용할 때.
  - ex) Anghami 음악 스트리밍 서비스가 사용자의 이전 음악 선택 기록을 통해 음악 추천을 하는 데에 이를 활용
    - 작동 방식은 문장 내 단어에 대한 예시와 동일하지만 문장 내 단어 대신, 사용자 스트림의 일부분인 개별 음악에 적용
    - 아웃풋은 단어 예측이 아니라 사용자의 과거 스트림 기록과 부합하는 노래가 된다.
  - **이미 확립된 아이디어를 가져와 목적대상을 바꿔 완전히 새로운것을 만든어내는 예시**

## 3. 3D cnn's for session-based recs

- [논문 : 3D cnn's for session-based recs](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p138-tuan.pdf)
  - 2017 RecSys conference
  - 3D 합성곱 신경망을 세션 기반 추천에 활용하는 걸 제안.
    - 순환 신경망을 사용해 세션 기반 추천을 생성
    - idea: 두 차원을 추가해서 추천되는 아이템들의 특성 데이터를 활용하겠다는 것.

    ![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/5ad1a34f-ef0a-4888-af24-ee3e267c7678)

      - 그것들이 사용자의 클릭 스트림에서 소비되는 방식에 대한 데이터에 더해서 모든 특성 데이터는 문자로 인코딩되었고 이는 CNN으로 이미지 인식할 때 이미지 데이터를 다룬 방식과 동일하게 다룰 것.

### 3-1. architecture
![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e6a03e01-3214-4d51-9a2f-61b61c1e16f0)


  - 복잡하지만, CNN 자체가 다른 분야에서도 널리 쓰이고 있기 때문에 CNN 자체의 복잡도 문제는 어느 정도 해결된 문제이지만, 이를 클릭 스트림 세션에 적용하는 데는 많은 변형이 필요
  - 서로 다른 길이의 세션을 처리하는 문제 등 때문에!!
  - **CNN은 사이즈가 동일한 데이터 블록에 적용하는 것이고 이게 논문이 해결해야 했던 문제임**!!!!
  - 복잡도 문제만 제쳐두면 결과물 자체는 좋은데 RNN의 결과를 간단히 뛰어넘는다.
  - 하지만 이는 특정 문제(클릭 스트림 세션에서 아이템을 추천하고 동시에 콘텐츠 속성을 고려하는 것)에 대한 해결책임을 기억해야 함.

### 3-2. deep feature extraction with cnn's
- CNN을 추천 시스템에 적용한 사례로서,
  - 추천하려는 것들로부터 **특성 데이터를 추출하는 데 쓰는 것**.
  - ex)  합성곱 신경망을 통해 특정한 이미지를 분류하듯이 또한 CNN을 사용해서 소리 파형의 음악 장르를 분류하는 것.
    - 음악 추천 시스템을 구축한다고 할 때 추천해야 할 노래들에 대한 충분한 메타데이터가 없을 수 있지만 CNN을 통해 장르와 같은 메타데이터를 자동으로 추출할 수 있다.
    - 그러한 장르 정보는 콘텐츠 기반 추천 등에 활용될 수 있으며 **협업 필터링이나 행렬 인수분해 모델을 보강하는데 활용** 가능.
    - 심지어 CNN을 통해 그림을 분류하거나 사람들이 과거에 좋아했던 그림을 근거로 추천가능.
    - 어쨌든 **CNN**을 훈련해야 하기에 모델이 알지 못하는 새로운 분류를 하는 데에 이를 쓸 수는 없지만 **데이터가 없는 새로운 항목에 대한 속성 정보를 채울 때 활용**할 수 있다.  

---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}