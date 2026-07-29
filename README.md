# ADV

XAI 학회 활동 — EEG-텍스트 정렬 모델(ELM)을 이용한 zero-shot EEG 이상 탐지 벤치마크.

## 구성

- `stage_a_elm_benchmark.py` — 사전학습된 ELM 체크포인트로 EEG를 normal/abnormal로 zero-shot 분류하고 정확도를 측정하는 벤치마크 스크립트.
- `ELM/` — [SamGijsen/ELM](https://github.com/SamGijsen/ELM) (ICML'25, *"EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping"*)을 vendor한 코드. EEG 인코더/텍스트 인코더 정의, 학습 스크립트(`run_DL.py`), 사전학습 체크포인트(`ELM/pretrained/5s`, `ELM/pretrained/60s`)가 포함되어 있습니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ELM/requirements.txt
pip install torch transformers
```

## 사용법

### 1. EEG 임베딩 생성

`stage_a_elm_benchmark.py`는 이미 만들어진 EEG 임베딩(`.h5`)을 입력으로 받습니다. 실제 TUAB 데이터셋 경로를 `ELM/pretrained/{5s,60s}/config_xy.yaml`의 `dataset.path`, `dataset.data_path` 등에 설정한 뒤 아래처럼 임베딩을 생성합니다.

```bash
torchrun ELM/run_DL.py -f <config_file>   # setting: GEN_EMB
```

임베딩 파일은 `embeddings`(또는 `features`) 데이터셋과 `pathology`(또는 `pat`) 라벨을 포함해야 합니다.

### 2. Zero-shot 벤치마크 실행

```bash
python stage_a_elm_benchmark.py \
  --embeddings path/to/embedding_dataset.h5 \
  --checkpoint ELM/pretrained/5s/model_0_checkpoint.pt \
  --config ELM/pretrained/5s/config_xy.yaml \
  --out stage_a_elm_result.json
```

결과는 마크다운 표 한 줄로 출력되고, 동일 내용이 `--out`에 지정한 JSON 파일에 저장됩니다.

## 출처

ELM 코드/사전학습 모델은 [SamGijsen/ELM](https://github.com/SamGijsen/ELM) 원본 저장소에서 가져왔습니다. 라이선스 및 인용 정보는 `ELM/README.md`를 참고하세요.
