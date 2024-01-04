---
title: "3D vision"
escerpt: "3D vision"

categories:
  - Vision
tags:
  - [AI, Vision]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-12-18
last_modified_at: 2023-12-18

comments: true
 

---




## 1. 3D vision?
- 우리가 사는 세계는 2차원의 어떤 이미지로 표현가능.실제론 3차원으로 구성. X,Y,Z좌표가 추가된 형태
- point cloud : 점들의 집합.

## 2. 3D representation: Point Cloud, Voxel, Mesh
- 3D 모델링시 고려해야할점.
  - 어떻게 컴퓨터에서 3D object를 표현하고 다룰것인지가 중요!!
  - representation을 어떻게 빠르고 자동적으로 construction할것인가?

#### Q) 서로다른 object representation할때는?
: 서로다른 method(voxel, mesh 등)을 선택하는게 더 좋다. 즉, 상황에 따라 적절한 representation사용하는게 좋다.( 내부에 대한 값들을 가지고 있을 필요가 없을땐 voxel을 굳이 쓸필요는 없다.)

- 즉, geometry에 대해서 representation을 어떻게 주느냐가 중요!!
- geometry(기하학)를 describing(묘사) 할때 여러가지 language가 있다.
|sementaics|syntax|
|---|---|
|values|data structures|
|operations|algorithms|
  - data structures위에서 동작하는 알고리즘들에 대한 문법을 가지고 있으며, 거기서 value가 어떻게 달라지느냐에 따라 모양새가 달라진다.
  - 가장중요한 건 data구조를 어떻게 정의하느냐!! 
  - 즉, data구조를 point cloud로 정의할지, mesh, voxel로 정의할지에 따라 모양새가 달라진다.

## 3. 3D object representation 예시
|Raw data|Solids|Surfaces|High-level structures|
|---|---|---|---|
|Point cloud|Voxles|Mesh|Scene graph|
|Range image|BSP tree|Subdivision|Skeleton|
|Polygon soup|CSG|Parametric|Skeleton|
||Sweep|Implicit|Application specific|

### 3-1-1. Point Cloud
  - 3D point samples의 unstructured set.
    - 고전적으로는 range finder, computre vision을 통해서 3D point cloud 를 reconstruction(복원)한것.
    - 최근에는 LiDAR를 이용하여 3mm속도를 스캔하여 거리를 재서 point cloud를 scan한다.
    - 3차원 vision의 가장 기초가 되는 representation.
### 3-1-2. Range Image
  - depth image의 pixel들을 mapping한 3차원 point set.
  - range scanner같은 걸로 얻을수 있음.
  - stereo match같은걸 해서 image로 부터 얻어낼수도 있음.
  - range image가 point cloud와 같이 있을때 teselation(공간을 완전히 메꾸는것)으로 조금더 dense하게 만들수 있고,  range surface를 이어가지고 구조도 복원해낼수 있음.
### 3-1-3. polygon soup
  - 여러가지 polygon 의 unstructured set.

#### Q) interactive(상호적인)한 모델링시스템을 어떻게 만들것인가?
: 이것은 3D vision에서 큰 이슈이다. 일반적으로 point cloud, mesh, voxel만 다루지만 해당분야도 관심을 가져보자.

### 3-2-1. Mesh
  - polygon 기반의, surface 기반의 data structured.
  - mesh는 polygon들의 여러가지 connected set. 그래서 triangles를 주로 사용한다. 그렇기에, 어떤 표면에 대해서 삼각형들의 집합으로 이어져 있다. 
  - mesh의 특징
    - 꼭 closed 되어 있지 않을수도 있다는것. 표면을 따라가서 순환되는 고리가 안만들어질 수도 있고,  오픈된 기능이 있을수도 있다는 것. 그래서 mesh로 만들었을때 조금더 삼차형 구조를 더 정밀하게 복원해내는 것처럼 보일수 있다.

### 3-2-2. subdivision surface
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

  - open3D
    - 3D에서의 openCV와 같은 역할을 하는 라이브러리
    - 3D dataset은 image와 달리 visualization이 어렵다. 그러나 open3D를 이용시 visualization, 3D machine learning, robotics에서 활용할수 있는 다양한 정보를 제공해준다.
    - https://github.com/isl-org/Open3D 보면 star가 5천개가 넘고 Fork도 1500개가 넘는다.

    - open3D의 history
    - 많은 아카데믹 페이퍼들이 Open3D를 써서 논문을 썼다.
      - Reconstruction paper, visual odometry paper, visualization & rendering paper, registration paper들이 open3D를 활용함.

    - open3D data structures
      - 포인트점들과 triangle mesh들로 연결이 되어있다. 또한 lineset, voxel grid , Octree, RGBD image 등 다양한 data structure를 지원한다.

  - COLMAP
    - structure from motion(SfM) 이나 Multi-view Stereo(MVS)에 대해서 general-purpose로 구현할수 있게 만들어놓은 GUI도 지원되고 command-line interface도 지원하는 library
    - 순서가 있는 비디오와 같은 이미지셋이거나 unordered image collection에 대해서도 reconstruction해주도록 구현되어 있는 library
    - COLMAP의 SfM pipeline을 통과시켜서 로마의 중심도시를 reconstruction하는 Sparse model이다. 
    - MVS pipeline을 통과시켜서 landmark를 dense model로 reconstruction한다. 즉 이미지만 가지고도 그럴듯하게 3차원으로 복원해주는 기능이 COLMAP에 있다.

    - COLMAP Tutorial
      - image-based로 3D reconstruction을 하는데 sparse representation을 scene으로부터 얻어내고 camera pose도 얻어낼수 있는 알고리즘을 제공한다.
      - 또한 output은 multi-view Stereo와 input으로 쓰여서 camera pose와 image를 넣어서 scene에 대한 dense representation을 복원할수도 있다

    - COLMAP SfM pipeline
      - illuminated 된 이미지 + visually overlap이 큰 이미지 + different view point인 이미지들의 셋을 받아서 먼저 correspondense search를 한다. 그래서 feature extraction과 matching을 수행하고 랜셋 같은거로 geometric verification을 통해서 model fitting을 한다. 
      - 그다음에 incremental reconstruction과정을 거치는데, 먼저 2~3장의 이미지로 reconstruction을 하고 한장씩 추가하는 방식으로 image registration을 하고 triangulation을 통해서 포인트를 복워하고 bundle adjustment를 통해서 이미지 전체에 대한 adjustment를 수행하고 bundle의 filter를 통해서 불필요한 잘못된 매치들을 제거해준다.  



#### Study surface registration
##### Convex optimization based
#### Learn how energy functions can be minimized

### Sparse convolution on the quantized point cloud representation in voxel grid
: 딥러닝 시대에서 어떻게 convolution을 point cloud에 적용하는지 알아보자.


## Implicit Funcion : NeRF
## Open3D
## COLMAP (SfM, MVS)
## 3D reconstruction 
: 여러장의 multi view image를 가지고 2차원 정보로부터 3차원을 복원해내는것을 말함.
- 대표적인 application으로 building Rome in A Day라고 해서 하루만에 로마를 reconstruction하자는 work가 있었다. 

- recall
  - fundamental matrix를 다시 reconstruction하거나 RANSAC 기반의 reconstruction하는 방식에 대해서 생각을 해보고 시작해보자.

- How? 어덯게 할까?
  - multi-view geometry라는 computer vision책을 일고 이해한후 코딩을 통하는 과정필요.

- 우리는 핵심만 배우자.
- multi-view geometry란?
  - 


## Human Reconsturction
## SLAM and LiDAR

# Generative models and graphics


---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}