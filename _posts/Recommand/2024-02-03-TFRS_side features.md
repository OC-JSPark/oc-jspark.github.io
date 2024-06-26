---
title: "TFRS : side features"
escerpt: "TFRS : side features"

categories:
  - Recommand
tags:
  - [AI, Recommand, TFRS]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2024-02-03
last_modified_at: 2024-02-03

comments: true
  

---

## 1.  TFRS : side feature(부가 기능 및 심층 검색 통합)
TFRS는 Keras에 추가된 얇은 layer일 뿐이며, 원하는 요소들도 추가할 수 있고 어떤 형태의 모델이든 정의할 수 있습니다. 즉, side feature 등을 추가할 수 있다.

- ex)
  - query tower는 사용자 ID들을 정수로 변환해서 이를 임베딩 층에 넣는 것뿐이었지만 side feature 활용하면 더 추가 가능.
  - 사용자 ID 외에도 추가적인 데이터를 넣을 수 있고 평점 생성 시점의 타임 스태프도 포함 가능
  - 사용자에 대한 어떤 메타 데이터도 쿼리 타워에 넣을 수 있는데 단지 추가하려는 데이터를 열의 형태로 오른쪽에 추가하면 된다.

  - 사용자 ID만 있던 기존 임베딩 층이 있는데 정규화된 타임 스탬프를 가진 또 다른 임베딩 층이 생겼는데 쿼리가 만들어진 시간을 일정한 시간대로 정규화 가능
  - 오래된 평점이 갖는 의미가 다소 떨어진다고 생각한다면 모델은 최근의 평점에 더 주목할 수 있고  타임 스탬프들을 이산변수로 만들 수도 있음.
  - 요일 기준으로 구분도 가능
  - 평일과 주말의 평점에 차이가 있다고 생각한다면 그걸 반영해야 하는데 어느 달인지 중요함.크리스마스 시즌에는 행태가 변했을지 모르니깐.
  - 이런 측면들을 모두 쿼리 타워에 포함하는 게 가능
  - 해당 정보들을 포착한 임베딩 층만 더 추가한다면 모두 가능.

- 그 정보들을 모두 연결하면  형성된 쿼리 타워 아웃풋은 검색 모델로 주입된다.
- 주의)
  - 일단 모든 데이터가 유용하지 않음.
  - 즉, query tower에 포함하려는 정보가 추천에 도움될 맥락을 제공하는지 먼저 판단해볼것.
  - 신경망에 맞는 특정 형태로 데이터 전처리 필요.
  - 범주 데이터가 알맞은 형태로 임베드 되게 하고 타임 스탬프 같은 연속 특성들은 정규화해야 함. 그래야 다른 데이터들과 잘 merge됨.
  - 한 가지 데이터가 단지 스케일 문제로 다른 데이터를 압도해선 안 되며, 표준화필요
  - 표준편차 1의 종형 커브로 정규화하는 게 맞는 데이터가 있다면 변형 필요.
  - 가끔은, 변수들을 이산변수로 만드는 것도 좋음. 타임 스탬프를 요일, 월, 연도 등으로 바꾸는 것.
  - 텍스트 데이터는 단어별로 토큰으로 벡터화해야 하는데 텍스트를 그대로 신경망에 주입할 수는 없으니깐.

  - ex)
    - 영화 제목이나 장르가 중요하다는 정보가 있다면 후보 타워에 동일한 방식으로 포함할 수 있음. 모델의 정보 부족문제 완화가능.


## 2. TFRS : deep retrieval models(심층검색 모델)

 - 랭킹 모델 쪽의 다층 퍼셉트론을 검색 모델 쪽에서도 할 수 있다.
- 사용자 ID와, 알맞게 전처리한 타임 스탬프 등을 포함했던 쿼리 타워를 가져와서 정보를 더 추가해도 좋음
- 모든 임베딩들을 연결해서 연결 임베딩 층으로 만들 수 있고 발전시킬수 있음.
- 그것을 다층 퍼셉트론에 제공해서 64개나 32개의 뉴런을 갖는 밀집 층으로 전할 수도 있다.

- 이런 방식은 톱다운 방식이라 바텀업 사례들과는 반대입니다

## TFRS : 다중작업 추천자 Deep & Cross Network ScanN 및 Serving

## Deep Factorization Machines


---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}