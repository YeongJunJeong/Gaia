# Gaia Benchmarks

Standard benchmarks for evaluating soil microbiome foundation models.

## Tasks

| # | Task | Type | Metric | Data Source |
|---|------|------|--------|-------------|
| 1 | Biome Classification | Multi-class | ROC-AUC, F1 | MGnify |
| 2 | Soil Chemistry Prediction | Regression | R², RMSE | NEON |
| 3 | Tillage Classification | Multi-class | Accuracy, Kappa | Literature |
| 4 | Drought Stress Detection | Binary | Accuracy, F1 | Naylor et al. |
| 5 | Abundance Reconstruction | Reconstruction | Cosine Similarity | Test set |

## Baselines

| Model | Type |
|-------|------|
| Random Forest | Traditional ML |
| XGBoost | Traditional ML |
| SVM (L2) | Traditional ML |
| MGM (original) | Foundation Model |
| MGM (zero-shot) | Zero-shot |
| Gaia (zero-shot) | Zero-shot |
| Gaia (fine-tuned) | Transfer Learning |

## Running Benchmarks

```bash
python -m benchmarks.run_all --model-path checkpoints/pretrain/best.pt
```
