---
title: "open-source LLM applications on CPUs using LLama 2, C Transformers, GGML, and LangChain"
escerpt: "open-source LLM applications on CPUs using LLama 2, C Transformers, GGML, and LangChain"

categories:
  - NLP
tags:
  - [AI, NLP]

toc: true
toc_sticky: true

breadcrumbs: true

date: 2024-04-07
last_modified_at: 2024-04-07

comments: true
 

---


## 1. architectue

![image](https://github.com/OC-JSPark/oc-jspark.github.io/assets/46878973/e868d537-99fe-4f5c-ad34-1b7e96e29d25)

  - Sentence-transformers 를 이용하여 document들을 embedding시킨 후 vector store에 저장.
  - 질의하게 되면 유사한 내용을 search 후 이를 prompt에 few-shot learning 방식으로 질문의 답을 맞추는 방식으로 동작.

* few-show learning(FSL)?
  - **극소량의 데이터** 만을 이용하여 새로운 작업이나 클래스를 빠르게 학습하도록 설계된 알고리즘.

  - config.json으로 yaml을 json으로 변경
  - db_build.py 에서

  ```
  # Import config vars
  with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

  위에꺼 대신에 아래로 변경(yaml -> json load 위해서!)

  def load_env_from_file(file_path):
    with open(file_path, 'r') as f:
      env = json.load(f)
    return env
  cfg = load_env_from_file('config/config.json')
  ```

  - 4bit, 8bit 양자로 학습한 모델이 있음.
    - 4bit : llama-2-7b-chat.ggmlv3.q4_1.bin : 4GB정도됨.
    - 45초만에 응답이됨.
      - source문서에서 결과를 가져왔는지와, answer에 대한 내용으로 정확하게 750유로 million을 찾아낸걸 알수잇음.
  - data에 input pdf가 들어감.
  - model : 실제 model download하는것.
    - download는 huggingface hub를 이용해서 download
      models > 해당 폴더에서 python 들어가서 진행

      ```
      from huggingface_hub import hf_hub_download
      ## model_download.txt에 있는 문서 치면됨.

      hf_hub_download(repo_id="TheBloke/Llama-2-7B-Chat-GGML", filename="Llama-2-7B-Chat.ggmlv3.q4_0.bin", cache_dir="./")
      ```

  - download 받은후 config > config.json 에 
  MODEL_BIN_PATH: 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'
  부분을 수정해줘야 한다.
  MODEL_BIN_PATH: "models/models--TheBloke--Llama-2-7B-Chat-GGML/snapshots/76xxxx"

  - src > llm.py
    - Llama2모델을 불러오는 기능
  - src > prompts.py
    - Llama2모델에 few-shot learning으로 context를 담아서 질문을 하는 파일.
  - src > utils.py
    - vector embedding을 시키는 내용들이 있음.
  - src > db_build.py
    - db를 만드는 파일
  - src > main.py
    - 최초 실행하는 파일
---


[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}