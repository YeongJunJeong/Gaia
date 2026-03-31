"""
NEON Data Collector for Gaia Project.

Collects paired soil microbiome + environmental data from the
National Ecological Observatory Network (NEON).

Data Products:
  - DP1.10107.001: Soil microbe metagenome sequencing
  - DP1.10086.001: Soil chemical properties (pH, organic C, total N)
  - DP1.00094.001: Soil temperature and moisture
  - DP1.00006.001: Precipitation and air temperature

Value: Only large-scale public source with paired microbiome + environmental data.
"""

import argparse
import logging
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


def load_config(config_path: str = "data/configs/neon.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_neon_sites(config: dict) -> list[dict]:
    """Get all terrestrial NEON sites."""
    url = f"{config['api_base_url']}/sites"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    sites = []
    for site in data.get("data", []):
        if site.get("siteType") == config["site_type"]:
            sites.append(
                {
                    "site_code": site["siteCode"],
                    "site_name": site.get("siteName", ""),
                    "state": site.get("stateCode", ""),
                    "latitude": site.get("siteLatitude"),
                    "longitude": site.get("siteLongitude"),
                    "domain": site.get("domainCode", ""),
                }
            )

    logger.info(f"Found {len(sites)} terrestrial NEON sites")
    return sites


def get_available_data(
    site_code: str, product_id: str, config: dict
) -> list[dict]:
    """Get available data files for a site and product."""
    url = (
        f"{config['api_base_url']}/data/{product_id}/{site_code}"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        files = []
        for month_data in data.get("data", {}).get("files", []):
            files.append(
                {
                    "name": month_data.get("name", ""),
                    "url": month_data.get("url", ""),
                    "size": month_data.get("size", 0),
                }
            )
        return files

    except requests.RequestException as e:
        logger.debug(f"No data for {site_code}/{product_id}: {e}")
        return []


def download_product_data(
    sites: list[dict],
    product_id: str,
    product_name: str,
    config: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """Download and combine data for a product across all sites."""
    all_frames = []

    for site in tqdm(sites, desc=f"Downloading {product_name}"):
        site_code = site["site_code"]
        files = get_available_data(site_code, product_id, config)

        for file_info in files:
            file_url = file_info["url"]
            if not file_url or not file_info["name"].endswith(".csv"):
                continue

            try:
                df = pd.read_csv(file_url)
                df["siteID"] = site_code
                all_frames.append(df)
            except Exception as e:
                logger.debug(f"Failed to read {file_url}: {e}")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        output_path = output_dir / f"neon_{product_name}.csv"
        combined.to_csv(output_path, index=False)
        logger.info(
            f"Saved {product_name}: {combined.shape[0]} rows, "
            f"{combined.shape[1]} columns"
        )
        return combined

    logger.warning(f"No data collected for {product_name}")
    return pd.DataFrame()


def create_paired_dataset(
    microbe_df: pd.DataFrame,
    chemical_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Create paired dataset by matching microbiome data with
    environmental measurements by site and date.
    """
    if microbe_df.empty or chemical_df.empty:
        logger.warning("Cannot create paired dataset: missing data")
        return pd.DataFrame()

    # Standardize date columns for joining
    for df in [microbe_df, chemical_df, physical_df]:
        if "collectDate" in df.columns:
            df["collect_month"] = pd.to_datetime(
                df["collectDate"]
            ).dt.to_period("M")

    # Join on site + month
    paired = microbe_df.merge(
        chemical_df[
            [
                "siteID",
                "collect_month",
                "soilInWaterpH",
                "organicCPercent",
                "nitrogenPercent",
            ]
        ].drop_duplicates(),
        on=["siteID", "collect_month"],
        how="inner",
    )

    if not physical_df.empty and "collect_month" in physical_df.columns:
        paired = paired.merge(
            physical_df[
                [
                    "siteID",
                    "collect_month",
                    "soilMoisture",
                    "soilTemp",
                ]
            ].drop_duplicates(),
            on=["siteID", "collect_month"],
            how="left",
        )

    output_path = output_dir / "neon_paired.csv"
    paired.to_csv(output_path, index=False)
    logger.info(f"Created paired dataset: {paired.shape[0]} samples")
    return paired


def collect_all(config: dict, output_dir: Path):
    """Main NEON collection pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get NEON sites
    logger.info("Step 1: Fetching NEON terrestrial sites...")
    sites = get_neon_sites(config)
    sites_df = pd.DataFrame(sites)
    sites_df.to_csv(output_dir / "neon_sites.csv", index=False)

    # Step 2: Download each data product
    products = config["data_products"]

    logger.info("Step 2: Downloading soil microbiome data...")
    microbe_df = download_product_data(
        sites,
        products["soil_microbe"]["id"],
        "soil_microbe",
        config,
        output_dir,
    )

    logger.info("Step 3: Downloading soil chemical data...")
    chemical_df = download_product_data(
        sites,
        products["soil_chemical"]["id"],
        "soil_chemical",
        config,
        output_dir,
    )

    logger.info("Step 4: Downloading soil physical data...")
    physical_df = download_product_data(
        sites,
        products["soil_physical"]["id"],
        "soil_physical",
        config,
        output_dir,
    )

    # Step 3: Create paired dataset
    logger.info("Step 5: Creating paired dataset...")
    create_paired_dataset(microbe_df, chemical_df, physical_df, output_dir)

    logger.info("NEON data collection complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Collect paired soil data from NEON"
    )
    parser.add_argument(
        "--config",
        default="data/configs/neon.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    collect_all(config, output_dir)


if __name__ == "__main__":
    main()
