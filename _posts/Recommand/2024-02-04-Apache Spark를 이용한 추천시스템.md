---
title: "Apache Spark를 이용한 추천시스템"
escerpt: "Apache Spark를 이용한 추천시스템"

categories:
  - Recommand
tags:
  - [AI, Recommand, Spark]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2024-02-04
last_modified_at: 2024-02-04

comments: true
  

---

## 1. Apache Spark 


- scaling it up
  - movielens 10만개 dataset이용한거를 scaling up해보자.

- 아파치 스파크(Apache Spark)
  - Java 8 / 11 에 적합한 환경이다.(Java16 설치하지말것.)
  - framework의 일종.
  - 컴퓨터 클러스터를 사용해 거대한 데이터 세트를 처리할 수 있게 한다.
  - 데이터가 늘수록 클러스터에 컴퓨터를 추가하고 추가하는 데에 한계는 없다.

### 1-1. install
1. Java 8 SDK 설치
  - [Java 8 설치경로](www.oracle.com/java/technologies/javase-jdk11-downloads.html)
  - 오라클 웹사이트에서 Java 8 SDK를 설치해야 하는데 Java 9 이상이 아니라 8 이어야만 한다. 
  - 현재 스파크는 Java 8에서만 구동되기 때문.
  - 설치시 글자공백이 없는 경로에 설치 필요
    - 주의) 그냥 단순 설치시 program file에 설치되기에 공백이 생김.

2. 환경변수 설정

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/9179a5e8-9d04-4f09-b0b3-a64919592124)

  -  Java_Home 환경 변수를 Java 8 JDK를 설치한 경로로 설정

3. scaling up
  - 프로그램을 속이고자 강의 자료의 scaling up 폴더에서 winutils.exe 파일을 복사한 다음 {c:＼winutils＼bin}에 붙여 넣자.

4. 하둡 환경변수 설정

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/bb4eb2ee-48f4-48af-b5f7-f3cc7dc41f11)

  - (윈도우) 스파크는 하둡(Hadoop) 분산 파일 시스템을 일부에 사용하기에 하둡이 PC에 설치됐다고 인지해야만 작동된다.
  - 하둡 홈 환경 변수를 {c:＼winutil} 로 설정하고 이제 컴퓨터를 재부팅해서 모든 환경 변수가 적용되도록 한다.

5. pyspark install

```
$ conda install pyspark
```

6. 주의할점
- 스파크를 설치했고, 스파크 홈 환경 변수도 설정했다면 아나콘다에 설치된 pyspark 패키지와 설치한 스파크의 버전이 일치하는지 확인할것.

## 2. Apache Spark 아키텍처

스파크가 중요한 이유는 거대한 데이터를 처리하는 일을 클러스터 전반에 아주 효율적이고 믿을 수 있게 분배해 주기 때문.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/5df5a018-b1a8-408e-9520-f5fc59368de7)

 
  - spark driver (script)는 스파크가 제공하는 API를 활용해 데이터를 처리할 방법을 정의.
  - 파이썬, 스칼라 혹은 자바로 쓰면 되고 보통 클러스터의 마스터 노드에 있는 드라이버 스크립트를 실행하면 cluster manager와 소통하며 클러스터에서 필요한 리소스를 할당
  - 스파크를 하둡 클러스터에서 실행중이라면 클러스터 매니저는 하둡의 얀(YARN)이다.
  - 스파크 자체의 클러스터 매니저도 있으니 대신써도 됨.
  - cluster manager는 executor 프로세스를 클러스터 전반에 골고루 분배하여 데이터가 처리한다.
  
  - spark가 하는 역할은 **데이터를 나누고 처리하는 가장 효율적인 방식을 찾는 일**을 모두 맡아서 처리한다.
  - 스파크는 최대한 많은 것들을 메모리에 담고 데이터가 저장된 노드에서 데이터를 처리하려 하며 DAG(directed acyclic graph, 유향 비순환 그래프)라는 것을 활용하여 가장 효율적인 방식으로 프로세스를 조직하는데 모든 구성 요소들은 필요시 소통 가능.
  - 이는 클러스터의 일부가 고장 났을 때 필요한 기능으로 탄력적인 기능이다.  
  - cluster manager가 단일 장애점처럼 보이지만, 설정을 통해 백업 매니저를 준비하면 필요한 상황에서 매니저 역할을 수행.

### 2-1. spark software architecture
소프트웨어 개발자 관점에서 architecutre를 볼때의 시점.

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/553d2225-3ff3-4606-a950-f5d7779e1f92)

  - 스파크는 모든 과제를 분배하는 SPARK CORE로 구성되며 그 위에 작업 시 필요한 다른 라이브러리들이 얹혀 있다.
  - Spark SQL :  DataSet라는 것을 정의하는데 이는 스파크에서 SQL 데이터베이스에서처럼 작업할 수 있게 하고 스파크의 표준 사용 방식이 되었음.
  - Spark Streaming :  실시간으로 데이터를 소화하고 들어오는 대로 처리하게 한다.
    - 또한 구조적 스트리밍(Structured Streaming)을 제공하는데 실시간 데이터를 SQL 데이터로써 다룰 수 있게 한다.
  - GraphX :  SNS 같은 그래프 데이터 구조에서 조직된 데이터를 분석하고 처리
  - MLLib : 스파크의 머신러닝 라이브러리로써 거대 데이터세트로부터 아주 간편하게 추천을 생성하게 하는 클래스들을 갖고 있음.

## 3. RDD
- RDD('탄력적 분산 데이터 집합') : Resilient, Distributed, dataset 

  - SQL 인터페이스를 대신 쓰는 게 아니라면 이는 스파크의 코어가 된다.
  - RDD는 처리하려는 데이터를 캡슐화하는 객체
  - spark driver script 작성시 RDD 상의 서로 함께 묶인 연산들을 정의하는 것이며, 마지막엔 원하는 형태의 결과를 얻게 된다.
    - RDD가 어디서 데이터를 로드할지 어떤 연산과 aggregation 함수를 수행해야 할지 어디에 아웃풋을 둘지 정의하는 것.
    - 텐서플로처럼 직접 실행하고 아웃풋을 요청하지 않으면 아무 일도 생기지 않는다. 
  - 단지 우리가 원하는 바를 충족시켜줄 그래프를 스파크가 구축한 뒤 그 그래프를 실행해 주는 것.

### 3-1. RDD 장점
- (프로그래밍 관점)이것이 바람직한 이유는 단지 RDD와 데이터를 적용할 연산에만 신경 쓰면 되기 때문
- RDD는 스파크가 클러스터에 걸쳐 연산을 분배하고 노드 장애를 다루는 복잡한 과정들을 모두 숨겨주기 때문에 컴퓨팅의 분배에 대해서 신경 쓸 것 없이 데이터를 어떻게 변형할지만 고민하면 되는 것.

### 3-2. evolution of the spark api

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/c507ff60-cf01-484f-acfc-9890926e84b8)

  - 스파크는 이후 RDD에 기반해 구축된 대체 API들을 도입
  - spark driver script가 하는 일에 대한 하위 레벨의 제어력을 갖고자 RDD를 사용해서 코딩할 수도 있지만,
  - Data Frame는 기저의 데이터들을 행 객체로 다룰 수 있게 하여 SQL 데이터베이스처럼 코딩할 수 있고 원한다면 RDD를 Data Frmae으로 변환할 수도 있다.

  - 스파크 2는 DataSet을 도입했는데 더 나은 타입 세이프를 지닌 데이터 프레임과도 같다. 
  - 즉, runtime이 아니라 compile 타임에 driver script의 에러를 더 많이 잡아낼 수 있다.


- 스파크는 모든 구성 요소들 간의 공통 언어로서 DataSet를 사용하기 때문에 개발을 쉽게 만들어준다.
  - 특징
    1. Data Frame <-> DataSet 서로 변환가능
    2. Data Frame은 row objectes(행 객체)의 데이터 세트와 같지만 R이나 파이썬에서는 작동하지 않는다.

- if. python에서 개발 중이라면 스파크의 공통 요소로서 데이터 프레임 객체를 써야 한다.
- 또한 DataSet는 좀 느린 편이라 일부 사람들은 파이썬 외의 경우에도 데이터 프레임을 쓰고는 한다.

### 3-3. python에서 spark 활용하는 방법
- 현재 python 예제는 RDD interface를 initial cleanup과 인풋 평점 데이터를 구조화하는데 쓰지만 다시 **데이터 프레임으로 변환**해야 한다.
- Spark의 MLLib에서 요구하는 input이기 때문.

- 이제 spark driver script를 보고 이를 활용해 추천해보자.
(Spark Matrix Factorization 및 ALS를 사용한 영화추천 > scalingup에서 실습)

  
---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}