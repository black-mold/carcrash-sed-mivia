# deeplearning_framework_pl


## Introduction
- Dataset
  -  [MIVIA](https://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/)
     -  지금은 다운로드 못받음
  -  AudioSet(Strong)https://github.com/black-mold/carcrash-sed-mivia/blob/main/README.md
     -  [AUDIOSET-Temporally-Strong Labels](https://research.google.com/audioset/download_strong.html): 데이터 많음, 라벨 품질 안좋음
  -  DESED
     -  [DCASE-task4](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline): 데이터 많음, 가정 내 발생하는 오디오가 target, 라벨 품질 좋음
  -  참고:
     - DCASE Challenge의 경우 label이 없는 데이터도 활용하여 학습하는 semi-supervised learning이 baseline임. 여기는 supervised learning만 구현됨.   
   
- Model
  - FDY-CRNN[https://github.com/frednam93/FDY-SED]: SOTA
    - pre-trained weight를 사용할 것을 권장
  - CRNN: 대충 만든 것 <- 사용 금지
- Loss
  - binary cross entropy
- Evaluation
  - 여기서는 util.py에 구현
  - 참고자료: [Metrics for Polyphonic Sound Event Detection](https://www.mdpi.com/2076-3417/6/6/162)
  - 참고그림![image](https://github.com/black-mold/carcrash-sed-mivia/assets/96871530/e54f71a1-ac2b-4050-beff-2b650924e5bc)


## Getting Started
- MIVIA 데이터셋을 다운받아서 `data/mivia_raw`에 놓기
  - ![image](https://github.com/black-mold/carcrash-sed-mivia/assets/96871530/4c966c4a-0d5b-423e-886f-2dad74ef0f81)


### 1. train
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/mivia_server.yaml --mode train
```


### 2. test
- Test 하기 전에 yaml 파일의 resum_checkpoint 위치에 모델 학습 결과를 입력할 것
  - 예시:
  - ![image](https://github.com/black-mold/carcrash-sed-mivia/assets/96871530/77ddefb5-456f-470b-ab35-970021382dd1)

```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/mivia_server.yaml --mode test
```
