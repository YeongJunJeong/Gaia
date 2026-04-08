# Gaia Bernburg OOD 검증 — 2026년 4월 8일

---

## 1. 새 데이터 확보

**Bernburg 장기 농업시험** (raabmarie/Synthesis_Three_Years_Bernburg, GitHub)
- Westerfeld와 동일 저자/포맷, 다른 사이트
- 96 샘플 × 35,942 ASV (763 속으로 집계)
- 3년 시계열 (2019, 2020, 2021)
- 토양화학 페어링: pH, C[%], N[%], OM[%], NH4-N, NO3-N, P2O5, 미량원소
- 처리: `data/processed_real/bernburg_abundance.csv`, `bernburg_metadata.csv`

## 2. 검증 1 — Linear Probe (백본 동결, Bernburg 80/20)

| 작업 | Gaia | RF | 승자 |
|------|------|-----|------|
| 경운 분류 | 84.2% | **100.0%** | RF |
| 시비 분류 | **68.4%** | 57.9% | **Gaia** |
| pH 예측 | **R²=0.585** | 0.551 | **Gaia** |
| 총 탄소 (C%) | **R²=0.718** | 0.357 | **Gaia (~2배)** |
| 총 질소 (N%) | **R²=0.732** | 0.725 | **Gaia** |
| 유기물 (OM%) | **R²=0.717** | 0.497 | **Gaia** |

**Gaia 5/6 승.** 백본은 Bernburg를 한 번도 안 봤지만 사전학습된 표현이 새 사이트에서도 유효함을 입증.

## 3. 검증 2 — True Zero-Shot Cross-Site (Westerfeld 학습 → Bernburg 시험)

머리(head)도 Bernburg 데이터를 전혀 안 보고, Westerfeld만으로 학습한 뒤 Bernburg로 직행.

| 작업 | Gaia | RF |
|------|------|-----|
| 총 탄소 (C%) | **R²=0.291** | 0.199 |
| pH | **R²=0.393** | **−0.515** (실패) |
| 총 질소 (N%) | **R²=0.520** | 0.307 |

**Gaia 3/3 승.** 특히 pH에서 RF는 음수 R² (평균 예측보다 나쁨), Gaia는 의미있는 양수.

## 4. 해석

- 사이트 간 토양/기후/관리 방식이 달라서 절대 R²는 낮지만, **방향성이 일관되게 Gaia > RF**
- RF는 각 속을 독립적인 feature로 처리 → 사이트별 가중치가 새 사이트에서 망가짐
- Gaia는 self-supervised로 학습한 속 간 관계를 표현 → 사이트가 바뀌어도 의미있는 신호 유지
- **Foundation model 일반화의 가장 강력한 증거**

## 5. 논문 업데이트

- 초록에 Bernburg OOD 결과 추가
- §4.12 새 섹션 추가: 두 검증 regime + 표 2개 + 해석
- 총 20쪽 (figures 4개 유지)
- 커밋: `d58e822` — feat: Bernburg OOD validation
