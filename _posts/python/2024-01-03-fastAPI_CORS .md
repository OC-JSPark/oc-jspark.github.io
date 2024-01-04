---
title:  "About CORS error"
excerpt: " About CORS error"

categories:
  - Python
tags:
  - [Python]

toc: true
toc_sticky: true

breadcrumbs: true
 
date: 2024-01-03
last_modified_at: 2024-01-03
---



## 1. FastAPI에서 CORS 해결하기

```
from starlette.middleware.cors import CORSMiddleware

origins=[
  "*"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
```

  - 모든 URL에 허용해주기 위해 origin에 "*"을 넣음.
  - 만약 Frontend URL을 지정해주고 싶다면?
  ```
  origins = [
    "frontend-app.yourdomain.com",
    "frontend-app2.yourdomain.com:7000"
    " <ip-address>:<port> "
  ]
  ```
- FastAPI의 응답헤더에 Cross-Origin을 허용한다는 정보를 추가하여, 브라우저가 각기다른 URL로부터 받은 응답을 조합하여 웹페이지를 표현하는데 문제없도록 하는것.

## 2. CORS란?
: Cross-Origin Resource Sharing

- 브라우저에서는 보안적인 이유로 **cross-origin** HTTP요청들을 제한함. 그래서 **cross-origin**요청하기 위해선 서버의 동의가 필요.
- 만약 서버가 동의한다면 브라우저에서는 요청을 허락 / 동의하지 않으면 브라우저에서 거절
- 이러하 허락을 구하고 거절하는 메커니즘을 HTTP-header를 이용해서 가능함. 이를 CORS라고 부름.
- 즉, 브라우저에서 **cross-origin**요청을 안전하게 할수 있도록 하는 메커니즘. 

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
 

