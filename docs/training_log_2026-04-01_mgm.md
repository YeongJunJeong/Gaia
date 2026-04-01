# Gaia 학습 기록 — 2026년 4월 1일 (MGM 전이학습)

---

## 1. MGM 전이학습이란

MGM(Microbiome General Model)은 26만 개의 미생물 샘플로 사전학습된 모델이다.
이 모델의 가중치를 가져와서 우리 토양 데이터로 추가 학습시키는 것을 "전이학습"이라 한다.

비유하면:
- 처음부터 학습 = 영어를 하나도 모르는 아이에게 영어 가르치기
- 전이학습 = 영어를 잘하는 사람에게 토양 전문 용어만 추가로 가르치기

## 2. MGM 모델 정보

| 항목 | 값 |
|------|-----|
| 아키텍처 | GPT-2 (Transformer Decoder) |
| 파라미터 수 | 8,924,928 (약 890만) |
| 사전학습 데이터 | MGnify 263,302개 샘플 (모든 바이옴) |
| 어휘 크기 | 9,669종 미생물 |
| 레이어 | 8 |
| 어텐션 헤드 | 8 |
| 임베딩 차원 | 256 |
| 입력 길이 | 512 |
| 프레임워크 | Hugging Face Transformers |
| 출처 | pip install microformer-mgm |

## 3. 데이터 준비

우리 토양 데이터를 MGM 형식으로 변환했다.
- MGM은 미생물 이름에 `g__` 접두사를 사용 (예: `g__Mycobacterium`)
- 우리 데이터 2,646종 중 MGM 어휘에 있는 것: **1,961종 (71.3%)**
- 최소 5종 이상 매칭된 샘플만 사용: **933개**

## 4. 학습 결과

### 설정

| 항목 | 값 |
|------|-----|
| 학습 데이터 | 933개 토양 샘플 (MGM 토큰으로 변환) |
| 학습/검증 분할 | 90% / 10% |
| 에폭 | 10 |
| 배치 크기 | 8 |
| 학습률 | 5e-5 |
| 장비 | CPU (RTX 5060이 MGM의 PyTorch 2.0.1과 비호환) |
| 소요 시간 | 약 21분 |

### 에폭별 검증 손실

| 에폭 | 검증 손실 |
|------|----------|
| 1 | 3.31 |
| 2 | 1.35 |
| 3 | 1.12 |
| 4 | 1.08 |
| 5 | 1.06 |
| 6 | 1.05 |
| 7 | 1.05 |
| 8 | 1.04 |
| 9 | 1.04 |
| 10 | **1.04** |

### 모든 모델 비교

| 모델 | 학습 데이터 | 검증 손실 | 정답 확률 |
|------|-----------|----------|----------|
| 자체 모델 (3/31, CPU) | 데모 200개 | 5.07 | 0.6% |
| 자체 모델 (3/31, GPU) | 데모 200개 | 5.12 | 0.6% |
| 자체 모델 (3/31, 실제) | MGnify 100개 | 5.63 | 0.4% |
| 자체 모델 (4/1) | MGnify 1,000개 | 5.76 | 0.3% |
| **MGM 전이학습 (4/1)** | **MGnify 933개** | **1.04** | **35%** |

MGM 전이학습 모델이 자체 모델 대비 검증 손실 **5.5배 낮음**, 정답 확률 **100배 이상 높음**.

## 5. 추론 테스트 비교

### 테스트 1: 산성 토양 미생물

입력: Mycobacterium, Bryobacter, Acidothermus

| 자체 모델 (1,000개) | MGM 전이학습 |
|---|---|
| Gardnerella | **Candidatus_Solibacter** (산성 토양) |
| Burkholderiaceae | **Acidibacter** (산성 토양) |
| Fusicatenibacter | Pajaroellobacter (토양) |
| **Blautia (장내 미생물!)** | Haliangium (토양) |
| **Streptococcus (장내!)** | Gemmatimonas (토양) |

MGM: 장내 미생물 0개, 토양 미생물만 예측.

### 테스트 2: 유기물 분해 미생물 (방선균)

입력: Streptomyces, Micromonosporaceae, Nocardioidaceae

| 자체 모델 (1,000개) | MGM 전이학습 |
|---|---|
| Gemmatimonadaceae | **Glycomyces** (방선균) |
| **Pseudonocardia (방선균)** | **Actinopolymorpha** (방선균) |
| Phycisphaeraceae | **Pseudonocardia** (방선균) |
| Ktedonobacteraceae | **Nocardioides** (방선균) |
| Roseiarcus | **Actinophytocola** (방선균) |

자체 모델: 방선균 1개 맞춤.
MGM 전이학습: **방선균 5개 연속 예측** — 같은 그룹임을 완벽히 인식.

### 테스트 3: 질소 고정 세균

입력: Bradyrhizobium, Rhizobium

| MGM 전이학습 예측 | 의미 |
|---|---|
| Sphingomonas | 뿌리 근처 세균 |
| Labrys | 토양 세균 |
| **Mesorhizobium** | **질소 고정 세균! (같은 그룹)** |
| Sphingobium | 토양 세균 |
| Pseudomonas | 뿌리 성장 촉진 세균 |

질소 고정 세균을 넣자 같은 역할의 Mesorhizobium을 예측. 생태학적으로 정확.

### 테스트 4: 흔한 토양 미생물

입력: Candidatus_Solibacter, Acetobacteraceae, Burkholderiaceae

| MGM 전이학습 예측 | 의미 |
|---|---|
| Bryobacter | 산성 토양 세균 |
| Acidothermus | 유기물 분해 |
| Candidatus_Koribacter | 산성 토양 세균 |
| Acidibacter | 산성 토양 세균 |
| Haliangium | 포식 세균 |

모두 토양 미생물. 산성 토양 계열이 주로 예측됨 — 입력 미생물의 특성을 반영.

## 6. 핵심 발견

1. **전이학습이 압도적**: 같은 데이터(933개)인데 검증 손실 5.76 → 1.04
2. **장내 미생물 오염 해결**: 자체 모델은 Blautia, Streptococcus 같은 장내 미생물이 섞였으나, MGM 전이학습 모델은 토양 미생물만 예측
3. **같은 그룹 인식**: 방선균 입력 시 방선균 5종 연속 예측, 질소 고정균 입력 시 같은 역할의 미생물 예측
4. **CPU로도 21분 만에 학습 가능**: MGM이 890만 파라미터로 작은 모델이라 GPU 없이도 빠름

## 7. 기술적 이슈

- RTX 5060(sm_120)이 MGM 요구 PyTorch 2.0.1과 호환되지 않아 CPU로 학습
- MGM 패키지(microformer-mgm)가 torch==2.0.1을 요구하여 버전 충돌 발생
- accelerate 버전도 0.23.0으로 다운그레이드 필요
- 향후 GPU 활용을 위해 MGM 가중치를 최신 PyTorch로 마이그레이션하는 작업 필요

## 8. 다음 단계

1. 다운로드 중인 나머지 데이터로 재학습 (현재 ~2,000개 수집 중)
2. 임베딩 지도 재생성 (MGM 모델 기반)
3. 미세조정 실험 (pH 예측, 바이옴 분류)
4. GPU 호환성 해결 (PyTorch 버전 마이그레이션)

## 9. 파일 구조

```
checkpoints/
├── pretrain/              # 데모 CPU (3/31)
├── pretrain_gpu/          # 데모 GPU (3/31)
├── pretrain_real/         # 자체 모델 100개 (3/31)
├── pretrain_1k/           # 자체 모델 1,000개 (4/1)
└── mgm_soil/              # MGM 전이학습 (4/1)
    └── best/              # 최종 모델 (검증 손실 1.04)
```
