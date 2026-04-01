# Gaia 학습 기록 — 2026년 4월 1일

---

## 1. 데이터 수집 (병렬)

### 병렬 수집 스크립트 작성

어제 순차 수집(샘플당 18초)에서, 5개 워커를 동시에 돌리는 병렬 수집으로 개선했다.
200개마다 중간 저장하여 중단되더라도 데이터를 잃지 않도록 설계.

### 수집 결과

| 항목 | 값 |
|------|-----|
| 시도한 분석 수 | ~1,400개 |
| 성공 (BIOM 파일 있음) | **1,000개** |
| 실패 (BIOM 파일 없음) | ~400개 |
| 발견된 미생물 종류 | **2,646종** |
| 경과 시간 | 약 3시간 (5워커 병렬) |
| 성공률 | 약 60% |

실패한 분석들은 BIOM 파일(미생물 분석 결과 파일)이 없거나 다른 형식으로 결과를 저장한 경우.

### 어제 대비 비교

| | 3/31 (어제) | 4/1 (오늘) | 증가 |
|---|------------|-----------|------|
| 샘플 수 | 100개 | **1,000개** | 10배 |
| 미생물 종류 | 975종 | **2,646종** | 2.7배 |

---

## 2. 전처리

| 단계 | 결과 |
|------|------|
| 분류체계 통일 | 2,646종 유지 |
| 풍부도 정규화 (TSS) | 완료 |
| 스파시티 필터링 (1%) | 2,646종 → **1,427종** (1,219종 제거) |
| 메타데이터 정제 | 469/974 완전 |
| 토큰화 | 어휘 **1,432개**, 길이 512 |

---

## 3. 모델 학습

### 설정

| 항목 | 어제 (100개) | 오늘 (1,000개) |
|------|------------|--------------|
| 샘플 수 | 100 | **1,000** |
| 학습/검증/테스트 | 85/10/5 | **850/100/50** |
| 어휘 크기 | 656 | **1,432** |
| 임베딩 차원 | 256 | **512** |
| 레이어 | 6 | 6 |
| 어텐션 헤드 | 8 | 8 |
| 피드포워드 차원 | 1,024 | **2,048** |
| 파라미터 수 | 650만 | **2,600만** |
| 에폭 | 100 | 50 |
| 배치 크기 | 16 | 32 |
| 학습률 | 3e-4 | 3e-4 |
| GPU | RTX 5060 | RTX 5060 |
| 소요 시간 | 14분 | **12분** |

### 학습 결과 (주요 구간)

| 에폭 | 학습 손실 | 검증 손실 | 비고 |
|------|----------|----------|------|
| 1 | 7.6273 | 6.5534 | |
| 3 | 6.0094 | 5.8002 | |
| 4 | 5.9189 | **5.7816** | |
| 7 | 5.7754 | **5.7614** | 검증 최저 (best.pt) |
| 10 | 5.6654 | 5.7892 | |
| 20 | 5.2847 | 5.8210 | |
| 30 | 5.1075 | 5.8490 | |
| 50 | 4.9491 | 5.8799 | 최종 |

### 어제 vs 오늘 — 과적합 비교

| | 어제 (100개) | 오늘 (1,000개) |
|---|-------------|--------------|
| 최종 학습 손실 | 3.85 | 4.95 |
| 최종 검증 손실 | 7.08 | 5.88 |
| **격차** | **3.23** | **0.93** |

과적합이 크게 줄었다. 어제는 학습/검증 격차가 3.23으로 모델이 데이터를 "외운" 상태였지만, 오늘은 0.93으로 줄어 실제 패턴을 학습하기 시작했다.

---

## 4. 임베딩 지도

모델 내부의 미생물 임베딩을 UMAP으로 2차원으로 압축하여 시각화했다.

- 어제 (100개 학습): 역할별 군집화 불명확
- 오늘 (1,000개 학습): 역할별 군집화 시작. 산성 토양 미생물, 방선균 등이 모이기 시작

파일:
- `docs/embedding_map.png` — 어제 모델 (100개 학습)
- `docs/embedding_map_1k.png` — 오늘 모델 (1,000개 학습)

---

## 5. 추론 테스트

best.pt 모델(에폭 7)로 미생물 예측 테스트를 수행했다.

### 테스트 1: 산성 토양 미생물

**입력:** Mycobacterium, Bryobacter, Acidothermus

**예측:**
1. Gardnerella
2. Burkholderiaceae — 토양 세균
3. Fusicatenibacter
4. Blautia
5. Streptococcus
6. Peptoniphilaceae
7. Bacillus — 토양 세균
8. Pajaroellobacter — 토양 세균
9. Micropepsaceae — 토양 세균
10. Roseburia

### 테스트 2: 질소 순환 미생물

**입력:** Haliangium, Nitrosomonadaceae, Nitrospira

**예측:**
1. Pajaroellobacter — 토양 세균
2. Fastidiosipila
3. Ureaplasma
4. Staphylococcus
5. Caulobacteraceae — 빈영양 토양 세균
6. Blautia
7. Acidibacter — 산성 토양 세균
8. Pirellulaceae — Planctomycetes
9. Alistipes
10. Ktedonobacteraceae — 토양 방선균

### 테스트 3: 흔한 토양 미생물

**입력:** Candidatus_Solibacter, Acetobacteraceae, Burkholderiaceae

**예측:**
1. Fastidiosipila
2. Peptoniphilus
3. Acidothermus — 토양 세균
4. Agathobacter
5. Pajaroellobacter — 토양 세균
6. Polyangiaceae — 포식 세균
7. Steroidobacteraceae — 토양 세균
8. Bradyrhizobium — 질소 고정 세균
9. Candidatus_Koribacter — 산성 토양 세균
10. Blautia

### 테스트 4: 유기물 분해 미생물 (방선균)

**입력:** Streptomyces, Micromonosporaceae, Nocardioidaceae

**예측:**
1. Gemmatimonadaceae — 건조 토양 세균
2. **Pseudonocardia — 방선균 (같은 그룹!)**
3. Phycisphaeraceae
4. Ktedonobacteraceae — 토양 세균
5. Roseiarcus — 토양 세균
6. Pirellulaceae
7. Acidibacter — 산성 토양 세균
8. Rhodoplanes — 질소 순환 세균
9. Acetobacteraceae — 토양 세균
10. Rhodobacteraceae

### 어제 대비 개선점

1. **중복 없이 다양한 미생물 예측** — 어제는 같은 이름이 반복되었으나 오늘은 10종 모두 다름
2. **같은 그룹 인식** — 방선균(Streptomyces)을 넣자 같은 방선균(Pseudonocardia)을 예측
3. **토양 세균 비율 증가** — 예측 결과에 토양 미생물이 더 많이 포함

### 남은 한계

- Blautia, Streptococcus 같은 장내 미생물이 섞여 나옴 (학습 데이터에 비토양 샘플 포함 가능성)
- 데이터 5,000개 이상으로 늘리면 토양 미생물만 정확하게 예측할 것으로 기대

---

## 6. 파일 구조 (업데이트)

```
data/
├── raw/mgnify/
│   ├── mgnify_abundance.csv     # 1,000 샘플 × 2,646 종
│   └── mgnify_metadata.csv      # 974 샘플 메타데이터
├── processed_real/
│   ├── gaia-corpus-v1.pkl       # 토큰화 코퍼스 (1,000 샘플)
│   ├── gaia-abundance-v1.csv    # 정규화 풍부도
│   ├── gaia-metadata-v1.csv     # 표준화 메타데이터
│   └── tokenizer.json           # 어휘 1,432개

checkpoints/
├── pretrain_1k/
│   ├── best.pt                  # 에폭 7 (검증 손실 5.7614)
│   └── gaia-v0.1.pt             # 에폭 50 최종

docs/
├── embedding_map.png            # 임베딩 지도 (100개 모델)
├── embedding_map_1k.png         # 임베딩 지도 (1,000개 모델)
├── training_log_2026-03-31.md   # 어제 기록
└── training_log_2026-04-01.md   # 오늘 기록
```
