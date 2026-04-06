"""
Earth Microbiome Project (EMP) Data Collector.

EMP provides pre-processed OTU tables via Qiita/FTP.
Filters for soil samples only.
Saves to NEW file (never overwrites existing data).
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw/emp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_emp_biom():
    """Download EMP release1 OTU table from Qiita/FTP."""

    # EMP provides data through multiple sources
    # Try redbiom API first (Qiita)
    logger.info("Attempting to download EMP data...")

    # EMP closed-reference OTU table is available as BIOM
    # Using the processed summary from EMP website
    urls_to_try = [
        # EMP FTP - pre-processed tables
        "ftp://ftp.microbio.me/emp/release1/otu_tables/closed_ref_greengenes/emp_cr_gg_13_8.release1.biom",
        # Alternative: subset tables
        "ftp://ftp.microbio.me/emp/release1/otu_tables/closed_ref_greengenes/emp_cr_gg_13_8.release1.subset_2k.biom",
    ]

    # EMP sample metadata
    metadata_url = "ftp://ftp.microbio.me/emp/release1/mapping_files/emp_qiime_mapping_release1.tsv"

    # Try downloading metadata first (smaller)
    logger.info("Downloading EMP metadata...")
    try:
        import urllib.request
        meta_path = OUTPUT_DIR / "emp_metadata.tsv"
        if not meta_path.exists():
            urllib.request.urlretrieve(metadata_url, str(meta_path))
            logger.info(f"Metadata saved: {meta_path}")

        meta = pd.read_csv(meta_path, sep='\t', low_memory=False)
        logger.info(f"EMP metadata: {len(meta)} samples")

        # Filter soil samples
        soil_keywords = ['soil', 'rhizosphere', 'agricultural', 'forest floor', 'grassland']
        if 'empo_3' in meta.columns:
            soil_mask = meta['empo_3'].str.lower().str.contains('soil', na=False)
            soil_samples = meta[soil_mask]
            logger.info(f"Soil samples (empo_3): {len(soil_samples)}")
        elif 'env_biome' in meta.columns:
            soil_mask = meta['env_biome'].str.lower().apply(
                lambda x: any(k in str(x) for k in soil_keywords) if pd.notna(x) else False
            )
            soil_samples = meta[soil_mask]
            logger.info(f"Soil samples (env_biome): {len(soil_samples)}")
        else:
            # Try any column with soil keywords
            for col in meta.columns:
                try:
                    mask = meta[col].astype(str).str.lower().str.contains('soil', na=False)
                    if mask.sum() > 100:
                        soil_samples = meta[mask]
                        logger.info(f"Soil samples ({col}): {len(soil_samples)}")
                        break
                except:
                    continue

        # Save soil sample IDs
        soil_ids = soil_samples['#SampleID'].tolist() if '#SampleID' in soil_samples.columns else soil_samples.iloc[:, 0].tolist()
        logger.info(f"Soil sample IDs: {len(soil_ids)}")

        # Save soil metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        soil_samples.to_csv(OUTPUT_DIR / f"emp_soil_metadata_{timestamp}.csv", index=False)

        return soil_samples, soil_ids

    except Exception as e:
        logger.error(f"Failed to download EMP metadata: {e}")
        return None, None


def try_qiita_api(soil_ids):
    """Try to get OTU data from Qiita API for soil samples."""
    logger.info("Trying Qiita API for OTU data...")

    # Qiita study 10317 is the main EMP study
    # Try to get processed data
    try:
        # Qiita public API
        url = "https://qiita.ucsd.edu/api/v1/study/10317/samples/info"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            logger.info("Qiita API accessible")
            return resp.json()
    except:
        pass

    # Alternative: use redbiom
    try:
        url = "https://qiita.ucsd.edu/api/v1/study/10317/samples"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            logger.info("Qiita samples accessible")
    except:
        pass

    return None


def download_emp_subset():
    """Download a pre-processed EMP subset with genus-level data."""
    logger.info("Downloading EMP genus-level data...")

    # Try to get genus-level collapsed table
    try:
        import urllib.request

        # EMP provides genus-level tables
        genus_url = "ftp://ftp.microbio.me/emp/release1/otu_tables/collapsed_tables/emp_cr_gg_13_8.release1.L6.biom"
        genus_path = OUTPUT_DIR / "emp_genus.biom"

        if not genus_path.exists():
            logger.info("Downloading genus-level BIOM (this may take a while)...")
            urllib.request.urlretrieve(genus_url, str(genus_path))
            logger.info(f"Downloaded: {genus_path} ({genus_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Parse BIOM
        try:
            from biom import load_table
            table = load_table(str(genus_path))
            logger.info(f"BIOM table: {table.shape[0]} features x {table.shape[1]} samples")

            # Convert to DataFrame
            df = pd.DataFrame(
                table.to_dataframe().T  # samples as rows
            )
            df.index.name = 'sample_id'
            df = df.reset_index()

            # Extract genus names from taxonomy strings
            new_cols = {}
            for col in df.columns:
                if col == 'sample_id':
                    continue
                # EMP taxonomy: k__Bacteria;p__Proteobacteria;...;g__Pseudomonas
                parts = str(col).split(';')
                genus = None
                for p in parts:
                    if p.startswith('g__') and len(p) > 3:
                        genus = p[3:]
                        break
                if genus:
                    if genus not in new_cols:
                        new_cols[genus] = df[col].values.copy()
                    else:
                        new_cols[genus] += df[col].values

            genus_df = pd.DataFrame(new_cols)
            genus_df.insert(0, 'sample_id', df['sample_id'])

            logger.info(f"Genus table: {genus_df.shape[0]} samples x {genus_df.shape[1]-1} genera")
            return genus_df

        except ImportError:
            logger.warning("biom-format not installed, trying JSON parse")
            import json
            with open(genus_path) as f:
                biom_data = json.load(f)
            logger.info(f"BIOM JSON loaded")
            return None

    except Exception as e:
        logger.error(f"Failed to download genus table: {e}")
        return None


def main():
    logger.info("=== EMP Data Collection ===")

    # Step 1: Get metadata and soil sample list
    soil_meta, soil_ids = download_emp_biom()

    # Step 2: Try to get genus-level OTU table
    genus_df = download_emp_subset()

    if genus_df is not None and soil_ids is not None:
        # Filter to soil samples only
        soil_genus = genus_df[genus_df['sample_id'].isin(soil_ids)]
        logger.info(f"Soil genus table: {len(soil_genus)} samples")

        # Save (with timestamp, never overwrite)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"emp_soil_genus_{timestamp}.csv"
        soil_genus.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
