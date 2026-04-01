"""
MGnify Parallel Data Collector for Gaia Project.

5개 워커로 동시에 BIOM 파일을 다운로드하여 수집 속도를 ~5배 높인다.
순차 처리 대비: 25시간 → 약 5시간 (5,000개 기준)

Usage:
    python data/scripts/collect_mgnify_parallel.py --max-samples 5000 --workers 5
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "data/configs/mgnify.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_soil_studies(config: dict) -> list[dict]:
    """Fetch studies associated with soil biomes."""
    base_url = config["api_base_url"]
    studies = []

    for lineage in config["biome_lineages"]:
        url = f"{base_url}/biomes/{lineage}/studies"
        while url:
            try:
                resp = requests.get(
                    url, params={"page_size": config["page_size"]}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("data", []):
                    studies.append(
                        {
                            "study_id": item["id"],
                            "biome_lineage": lineage,
                            "study_name": item["attributes"].get("study-name", ""),
                            "samples_count": item["attributes"].get(
                                "samples-count", 0
                            ),
                        }
                    )

                url = data.get("links", {}).get("next")
                time.sleep(0.5)
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                break

    seen = set()
    unique = []
    for s in studies:
        if s["study_id"] not in seen:
            seen.add(s["study_id"])
            unique.append(s)

    logger.info(f"Found {len(unique)} soil-related studies")
    return unique


def fetch_analyses_for_study(study_id: str, config: dict) -> list[dict]:
    """Fetch analyses for a given study."""
    base_url = config["api_base_url"]
    url = f"{base_url}/studies/{study_id}/analyses"
    analyses = []

    while url:
        try:
            resp = requests.get(
                url, params={"page_size": config["page_size"]}, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                sample_data = (
                    item.get("relationships", {})
                    .get("sample", {})
                    .get("data", {})
                )
                analyses.append(
                    {
                        "analysis_id": item["id"],
                        "study_id": study_id,
                        "sample_id": sample_data.get("id", "")
                        if sample_data
                        else "",
                        "pipeline_version": item["attributes"].get(
                            "pipeline-version", ""
                        ),
                    }
                )

            url = data.get("links", {}).get("next")
            time.sleep(0.3)
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch analyses for {study_id}: {e}")
            break

    return analyses


def fetch_biom_taxonomy(analysis_id: str, base_url: str) -> dict[str, float]:
    """Download BIOM file and extract genus-level taxonomy."""
    try:
        resp = requests.get(
            f"{base_url}/analyses/{analysis_id}/downloads", timeout=30
        )
        resp.raise_for_status()
        downloads = resp.json().get("data", [])
    except requests.RequestException:
        return {}

    biom_url = None
    for dl in downloads:
        alias = dl.get("attributes", {}).get("alias", "")
        if "SSU_OTU_TABLE_JSON" in alias:
            biom_url = dl["links"]["self"]
            break

    if not biom_url:
        return {}

    try:
        resp = requests.get(biom_url, timeout=60)
        resp.raise_for_status()
        biom_data = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return {}

    rows = biom_data.get("rows", [])
    data_entries = biom_data.get("data", [])

    row_counts = defaultdict(float)
    for entry in data_entries:
        if len(entry) >= 3:
            row_counts[entry[0]] += entry[2]

    genus_counts = {}
    for i, row in enumerate(rows):
        raw_taxonomy = row.get("metadata", {}).get("taxonomy", [])
        count = row_counts.get(i, 0)
        if count <= 0:
            continue

        if isinstance(raw_taxonomy, str):
            levels = [t.strip() for t in raw_taxonomy.split(";")]
        else:
            levels = list(raw_taxonomy)

        genus = None
        for level in levels:
            if level and level.startswith("g__") and len(level) > 3:
                genus = level[3:]
                break
        if genus is None:
            for level in levels:
                if level and level.startswith("f__") and len(level) > 3:
                    genus = level[3:]
                    break

        if genus and genus.strip():
            genus = genus.strip()
            genus_counts[genus] = genus_counts.get(genus, 0) + count

    return genus_counts


def fetch_sample_metadata(sample_id: str, base_url: str) -> dict:
    """Fetch metadata for a sample."""
    try:
        resp = requests.get(f"{base_url}/samples/{sample_id}", timeout=30)
        resp.raise_for_status()
        attrs = resp.json().get("data", {}).get("attributes", {})
        return {
            "sample_id": sample_id,
            "sample_name": attrs.get("sample-name", ""),
            "latitude": attrs.get("latitude"),
            "longitude": attrs.get("longitude"),
            "collection_date": attrs.get("collection-date", ""),
            "biome": attrs.get("environment-biome", ""),
            "feature": attrs.get("environment-feature", ""),
            "material": attrs.get("environment-material", ""),
        }
    except requests.RequestException:
        return {"sample_id": sample_id}


def process_one_analysis(analysis: dict, base_url: str) -> dict | None:
    """하나의 analysis에서 BIOM + 메타데이터를 가져오는 함수 (워커가 실행)."""
    analysis_id = analysis["analysis_id"]
    sample_id = analysis["sample_id"]

    genus_counts = fetch_biom_taxonomy(analysis_id, base_url)
    if not genus_counts:
        return None

    metadata = fetch_sample_metadata(sample_id, base_url)
    metadata["study_id"] = analysis["study_id"]
    metadata["pipeline_version"] = analysis["pipeline_version"]

    return {
        "abundance": {"sample_id": sample_id, "analysis_id": analysis_id, **genus_counts},
        "metadata": metadata,
    }


def save_checkpoint(
    abundance_records: list,
    metadata_records: list,
    output_dir: Path,
    config: dict,
):
    """중간 저장 — 혹시 중단되더라도 데이터를 잃지 않도록."""
    if abundance_records:
        df = pd.DataFrame(abundance_records).fillna(0)
        df.to_csv(output_dir / config["abundance_file"], index=False)

    if metadata_records:
        df = pd.DataFrame(metadata_records)
        df.to_csv(output_dir / config["metadata_file"], index=False)


def collect_all(
    config: dict,
    output_dir: Path,
    max_samples: int = 5000,
    workers: int = 5,
    checkpoint_every: int = 200,
):
    """병렬 수집 메인 함수."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = config["api_base_url"]

    # Step 1: 연구 목록 가져오기
    logger.info("Step 1: Finding soil studies...")
    studies = fetch_soil_studies(config)
    pd.DataFrame(studies).to_csv(output_dir / "studies.csv", index=False)

    # Step 2: 분석 목록 가져오기
    logger.info("Step 2: Finding analyses...")
    all_analyses = []
    for study in tqdm(studies, desc="Studies"):
        analyses = fetch_analyses_for_study(study["study_id"], config)
        all_analyses.extend(analyses)
        if len(all_analyses) >= max_samples:
            all_analyses = all_analyses[:max_samples]
            break

    logger.info(f"Found {len(all_analyses)} analyses to process")

    # Step 3: 병렬로 BIOM 다운로드
    logger.info(f"Step 3: Downloading with {workers} parallel workers...")
    abundance_records = []
    metadata_records = []
    seen_samples = set()
    n_success = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one_analysis, analysis, base_url): analysis
            for analysis in all_analyses
        }

        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()

                if result:
                    abundance_records.append(result["abundance"])
                    sample_id = result["metadata"]["sample_id"]
                    if sample_id not in seen_samples:
                        metadata_records.append(result["metadata"])
                        seen_samples.add(sample_id)
                    n_success += 1
                else:
                    n_failed += 1

                pbar.update(1)
                pbar.set_postfix(ok=n_success, fail=n_failed)

                # 중간 저장
                if n_success > 0 and n_success % checkpoint_every == 0:
                    save_checkpoint(
                        abundance_records, metadata_records, output_dir, config
                    )
                    logger.info(
                        f"Checkpoint: {n_success} samples saved"
                    )

    # Step 4: 최종 저장
    logger.info("Step 4: Saving final results...")
    save_checkpoint(abundance_records, metadata_records, output_dir, config)

    if abundance_records:
        n_genera = len(abundance_records[0]) - 2
        logger.info(f"Saved: {len(abundance_records)} samples")

    logger.info(
        f"Done! {n_success} succeeded, {n_failed} failed "
        f"out of {len(all_analyses)} total"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Parallel collect soil microbiome data from MGnify"
    )
    parser.add_argument(
        "--config", default="data/configs/mgnify.yaml"
    )
    parser.add_argument(
        "--max-samples", type=int, default=5000,
        help="Maximum samples to collect",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of parallel workers (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=200,
        help="Save checkpoint every N samples",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])

    collect_all(
        config,
        output_dir,
        max_samples=args.max_samples,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
