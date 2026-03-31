# Gaia Data Standardization Protocol

## Overview

This document defines the data standardization protocol for soil microbiome data used in the Gaia project. All contributed data must follow these standards to ensure consistency and reproducibility.

## Taxonomy Reference

- **Standard**: GTDB r220 (Genome Taxonomy Database)
- All taxonomic assignments must be mapped to GTDB nomenclature
- Resolution level: Genus (속)

## Abundance Normalization

Two normalization methods are supported:

### Option A: TSS (Total Sum Scaling)
- Relative abundance (proportions summing to 1.0)
- Simple, interpretable
- Subject to compositionality artifacts

### Option B: CLR (Centered Log-Ratio)
- Log-ratio transformation with geometric mean reference
- Addresses compositionality
- Requires pseudocount for zeros (default: 1e-6)

## Sparsity Filtering

- Remove genera present in < 0.1% of total samples
- Reduces noise and manages vocabulary size
- Expected vocabulary: ~3,000-5,000 soil-associated genera

## Metadata Requirements

### Required Fields

| Field | Format | Example |
|-------|--------|---------|
| `sample_id` | String | `MGYS00005432_run1` |
| `biome` | ENVO ontology term | `ENVO:00002259` (agricultural soil) |
| `latitude` | Float (-90 to 90) | `38.8951` |
| `longitude` | Float (-180 to 180) | `-77.0364` |
| `collection_date` | ISO 8601 | `2024-06-15` |

### Recommended Fields

| Field | Format | Description |
|-------|--------|-------------|
| `sequencing_platform` | String | Illumina MiSeq, NovaSeq, etc. |
| `extraction_kit` | String | DNA extraction method |
| `analysis_pipeline` | String | Bioinformatics pipeline used |
| `depth_cm` | Float | Soil sampling depth |
| `ph` | Float | Soil pH |
| `organic_carbon_pct` | Float | Soil organic carbon (%) |
| `total_nitrogen_pct` | Float | Total nitrogen (%) |
| `moisture_pct` | Float | Soil moisture (%) |
| `temperature_c` | Float | Soil temperature (C) |
| `land_use` | String | Cropland, forest, grassland, etc. |

## Biome Classification (ENVO Ontology)

| Biome | ENVO Term |
|-------|-----------|
| Agricultural soil | ENVO:00002259 |
| Forest soil | ENVO:01001198 |
| Grassland soil | ENVO:00005750 |
| Desert soil | ENVO:01001357 |
| Wetland soil | ENVO:00002044 |
| Permafrost | ENVO:01001526 |
| Rhizosphere | ENVO:01000999 |

## Quality Checklist

| Criterion | Threshold | Action if Failed |
|-----------|-----------|-----------------|
| Total reads | > 10,000 | Remove sample |
| Classified genera | > 20 | Remove sample |
| Metadata completeness | Biome + location required | Tag as "unknown" |
| Top-1 genus share | < 90% | Flag contamination, review |
| Sequencing platform info | Required | Tag as "batch correction not possible" |

## Corpus Format

### MGM-compatible tokenization:
1. Sort genera by abundance (descending)
2. Tokenize genus names using vocabulary index
3. Pad/truncate to sequence length 512
4. 99.99% of samples fit within 512 tokens

### File Formats

- **Corpus**: `gaia-corpus-v1.pkl` (Python pickle, pandas DataFrame)
- **Metadata**: `gaia-metadata-v1.csv` (CSV with headers)
- **Config**: `data_config.yaml` (YAML configuration)
