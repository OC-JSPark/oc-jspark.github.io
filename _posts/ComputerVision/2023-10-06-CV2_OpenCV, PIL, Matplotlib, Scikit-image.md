---
title: "CV2 : OpenCV, PIL, Matplotlib, Scikit-image"
escerpt: "Computer Vision 기본이론 2"

categories:
  - Vision
tags:
  - [AI, Vision, OpenCV, PIL, Matplotlib, Scikit-image]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2023-10-06
last_modified_at: 2023-10-06

comments: true
 

---

# 4. OpenCV, PIL, Matplotlib, Scikit-image 
: 이미지 읽어들이고 처리하는 많이 쓰이는 라이브러리

## 4-1. OpenCV
- 컴퓨터 비전 을 목적으로 하는 오픈 소스 라이브러리
- 인텔 CPU 에서 사용하는 경우 속도 향상을 볼 수 있는 IPP (Intel Performance Primitives) 를 지원한다.
-  기존에 C++ 에서 사용할 수 있게 구현되었으나, OpenCV-python 을 통해 python 포팅도 되어 있다.
-  단점: GPU operation 에 대한 지원을, 명시적으로 python 과 연계하여 하지는 않는다.
- 설치: pip install opencv-python 
- [OpenCV documentation](https://docs.opencv.org/4.x/)
- [OpenCV python tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### 4-1-1. Opencv 예시

1. Image read and visualize
2. multiple image visualize, figure/axs in matplotlib
3. image cropping, masking and save
4. Image matching and visualize

## 4-2. matplotlib

- Python 과 numpy array를 기반으로 plotting 과 visualize를 목적으로 하는 라이브러리
- 주어진 데이터에 대해서 차트와 plot 을 편리하게 그려주는 데이터
시각화 패키지 이다.
- 이미지 시각화와 디버깅을 위해 사용한다.
- 설치 : pip install matplotlib
- [Matplotlib documentation](https://matplotlib.org/stable/index.html)
- [Mattploblit tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [wikipedia-Matplotlib](https://ko.wikipedia.org/wiki/Matplotlib)


## 4-3. PIL

- PIL (Python Imaging Library) 로서 컴퓨터 비전 보다는 **이미지 처리에 중점을 둔 라이브러리**.
  - 컴퓨터 비전은 이미지 자체를 분석해서 classification이나 object를 찾는 결과를 desicion하는 것이 목적
  - 이미지처리는 이미지를 넣었을때 이미지가 어떻게 바뀔지를 찾는 컴퓨터 graphics에 가까운 task 이다.

- 픽셀단위의 이미지 조작이나, 마스킹, 투명도 제어, 윤곽 보정 및 검출 등의 다양한 이미지 조작을 할 수 있다.
  - downsampling
  - upsampling

- 설치 : pip install pillow
- [Pillow (PIL forked) documentation](https://pillow.readthedocs.io/en/stable)

### 4-3-1. PIL 예시

1. Image read and visualize
2. image cropping, rotating, scaling
3. Image interpolation(upsampling, downsampling)

## 4-4. Scikit-image

- Scikit-image 는 Pillow (PIL) 과 마찬가지로, 이미지 조작과 필터링이 가능하다.
- numpy 를 기반으로 동작하기 때문에, 좀 더 numpy와의 호환성이 좋다.
- 설치 : pip install scikit-image

```
from skimage import data, io, filters

image = data.coins()
# ... or any other NumPy array!
edges = filters.sobel(image)
io.imshow(edges)
io.show()
```

- [Scikit-image Documentation](https://scikit-image.org)

---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}