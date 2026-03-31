"""
MGnify Data Collector for Gaia Project.

Collects genus-level taxonomic abundance tables from soil biome samples
via the MGnify REST API.

Source: https://www.ebi.ac.uk/metagenomics/api/v1
Target: 5,000-15,000 soil-related samples
Biomes: agricultural soil, forest soil, grassland soil, desert soil,
        permafrost, rhizosphere
"""

import argparse
import json
import logging
import time
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
        page = 1

        while url:
            try:
                resp = requests.get(
                    url,
                    params={"page_size": config["page_size"]},
                    timeout=30,
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
                page += 1
                time.sleep(0.5)  # Rate limiting

            except requests.RequestException as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                break

    logger.info(f"Found {len(studies)} soil-related studies")
    return studies


def fetch_analyses_for_study(
    study_id: str, config: dict
) -> list[dict]:
    """Fetch analyses (runs) for a given study."""
    base_url = config["api_base_url"]
    url = f"{base_url}/studies/{study_id}/analyses"
    analyses = []

    while url:
        try:
            resp = requests.get(
                url,
                params={"page_size": config["page_size"]},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                analyses.append(
                    {
                        "analysis_id": item["id"],
                        "study_id": study_id,
                        "sample_id": item["relationships"]
                        .get("sample", {})
                        .get("data", {})
                        .get("id", ""),
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


def fetch_taxonomy(analysis_id: str, config: dict) -> dict[str, float]:
    """Fetch genus-level taxonomy for an analysis."""
    base_url = config["api_base_url"]
    url = f"{base_url}/analyses/{analysis_id}/taxonomy/ssu"
    taxonomy = {}

    while url:
        try:
            resp = requests.get(
                url,
                params={"page_size": config["page_size"]},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                attrs = item.get("attributes", {})
                lineage = attrs.get("lineage", "")
                count = attrs.get("count", 0)

                # Extract genus level from lineage
                parts = lineage.split(";")
                if len(parts) >= 6:  # Has genus level
                    genus = parts[5].strip()
                    if genus and genus != "":
                        taxonomy[genus] = taxonomy.get(genus, 0) + count

            url = data.get("links", {}).get("next")
            time.sleep(0.3)

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch taxonomy for {analysis_id}: {e}")
            break

    return taxonomy


def fetch_sample_metadata(sample_id: str, config: dict) -> dict:
    """Fetch metadata for a sample."""
    base_url = config["api_base_url"]
    url = f"{base_url}/samples/{sample_id}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        attrs = data.get("data", {}).get("attributes", {})
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
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch metadata for {sample_id}: {e}")
        return {"sample_id": sample_id}


def collect_all(config: dict, output_dir: Path, max_samples: int | None = None):
    """Main collection pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get all soil studies
    logger.info("Step 1: Fetching soil-related studies...")
    studies = fetch_soil_studies(config)

    # Save study list
    studies_df = pd.DataFrame(studies)
    studies_df.to_csv(output_dir / "studies.csv", index=False)
    logger.info(f"Saved {len(studies)} studies")

    # Step 2: Get analyses for each study
    logger.info("Step 2: Fetching analyses for each study...")
    all_analyses = []
    for study in tqdm(studies, desc="Studies"):
        analyses = fetch_analyses_for_study(study["study_id"], config)
        all_analyses.extend(analyses)

        if max_samples and len(all_analyses) >= max_samples:
            all_analyses = all_analyses[:max_samples]
            break

    logger.info(f"Found {len(all_analyses)} analyses")

    # Step 3: Fetch taxonomy and metadata for each analysis
    logger.info("Step 3: Fetching taxonomy and metadata...")
    abundance_records = []
    metadata_records = []
    seen_samples = set()

    for analysis in tqdm(all_analyses, desc="Analyses"):
        analysis_id = analysis["analysis_id"]
        sample_id = analysis["sample_id"]

        # Fetch taxonomy
        taxonomy = fetch_taxonomy(analysis_id, config)
        if taxonomy:
            record = {"sample_id": sample_id, "analysis_id": analysis_id}
            record.update(taxonomy)
            abundance_records.append(record)

        # Fetch metadata (deduplicate by sample_id)
        if sample_id and sample_id not in seen_samples:
            metadata = fetch_sample_metadata(sample_id, config)
            metadata["study_id"] = analysis["study_id"]
            metadata["pipeline_version"] = analysis["pipeline_version"]
            metadata_records.append(metadata)
            seen_samples.add(sample_id)

        time.sleep(0.2)  # Rate limiting

    # Step 4: Save results
    logger.info("Step 4: Saving results...")
    if abundance_records:
        abundance_df = pd.DataFrame(abundance_records).fillna(0)
        abundance_df.to_csv(
            output_dir / config["abundance_file"], index=False
        )
        logger.info(
            f"Saved abundance table: {abundance_df.shape[0]} samples, "
            f"{abundance_df.shape[1] - 2} genera"
        )

    if metadata_records:
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(
            output_dir / config["metadata_file"], index=False
        )
        logger.info(f"Saved metadata: {metadata_df.shape[0]} samples")

    return abundance_records, metadata_records


def main():
    parser = argparse.ArgumentParser(
        description="Collect soil microbiome data from MGnify"
    )
    parser.add_argument(
        "--config",
        default="data/configs/mgnify.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to collect (for testing)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    collect_all(config, output_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
