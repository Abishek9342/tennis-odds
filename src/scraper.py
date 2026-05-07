"""Stage 1: Download and normalise ATP tennis data from tennis-data.co.uk."""

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "http://www.tennis-data.co.uk"

COLUMN_MAP = {
    # Date
    "Date": "date",
    # Tournament info
    "Tournament": "tournament",
    "Series": "tournament",  # older files
    "Surface": "surface",
    "Court": "court",
    "Round": "round",
    "Best of": "best_of",
    # Players
    "Winner": "winner",
    "Loser": "loser",
    # Rankings
    "WRank": "w_rank",
    "LRank": "l_rank",
    "WPts": "w_pts",
    "LPts": "l_pts",
    # Score
    "Wsets": "w_sets",
    "Lsets": "l_sets",
    # Bet365
    "B365W": "b365w",
    "B365L": "b365l",
    # Pinnacle
    "PSW": "psw",
    "PSL": "psl",
    "PinnacleSP_W": "psw",
    "PinnacleSP_L": "psl",
    # Best/max odds
    "MaxW": "maxw",
    "MaxL": "maxl",
    # Average odds
    "AvgW": "avgw",
    "AvgL": "avgl",
}

KEEP_COLS = [
    "date", "tournament", "surface", "round", "best_of",
    "winner", "loser",
    "w_rank", "l_rank", "w_pts", "l_pts",
    "w_sets", "l_sets",
    "b365w", "b365l",
    "psw", "psl",
    "maxw", "maxl",
    "avgw", "avgl",
]


def discover_urls(base: str = BASE_URL) -> list[tuple[int, str]]:
    """Fetch alldata.php and return [(year, url)] for ATP files only, oldest first."""
    resp = requests.get(
        f"{base}/alldata.php",
        headers={"User-Agent": "Mozilla/5.0 (research/data-collection)"},
        timeout=30,
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # ATP files: "{year}/{year}.xls(x)" — WTA files have "w" suffix in dir
        if not (href.endswith(".xls") or href.endswith(".xlsx")):
            continue
        parts = href.split("/")
        if len(parts) != 2:
            continue
        year_dir = parts[0]
        # Skip WTA (directories end with 'w', e.g. "2024w")
        if year_dir.endswith("w"):
            continue
        try:
            year = int(year_dir)
        except ValueError:
            continue
        full_url = f"{base}/{href}"
        results.append((year, full_url))

    results.sort(key=lambda x: x[0])
    return results


def download_all(dest: Path, force: bool = False) -> list[Path]:
    """Download all ATP files into dest/raw/, skipping existing files."""
    raw_dir = dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = discover_urls()
    paths = []

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research/data-collection)"

    for year, url in tqdm(urls, desc="Downloading"):
        filename = url.split("/")[-1]
        local = raw_dir / f"{year}_{filename}"

        if local.exists() and not force:
            paths.append(local)
            continue

        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        local.write_bytes(resp.content)
        paths.append(local)
        time.sleep(1.0)

    return paths


def _read_excel(path: Path, year: int) -> pd.DataFrame:
    engine = "xlrd" if path.suffix == ".xls" else "openpyxl"
    df = pd.read_excel(path, engine=engine)
    # Some files have duplicate column names — deduplicate by keeping first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    df["year"] = year
    return df


NUMERIC_COLS = {
    "w_rank", "l_rank", "w_pts", "l_pts", "w_sets", "l_sets",
    "b365w", "b365l", "psw", "psl", "maxw", "maxl", "avgw", "avgl", "best_of",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    # After renaming, drop any duplicate column names (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    present = [c for c in KEEP_COLS if c in df.columns]
    df = df[present + ["year"]].copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Coerce numeric columns — some files use 'NR' or other strings
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_normalise(paths: list[Path], out: Path) -> pd.DataFrame:
    """Read all downloaded files, normalise schema, save to Parquet."""
    frames = []
    for p in paths:
        year = int(p.stem.split("_")[0])
        try:
            df = _read_excel(p, year)
            frames.append(_normalise(df))
        except Exception as e:
            print(f"  Warning: could not read {p.name}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date", "winner", "loser"])
    combined = combined.sort_values("date").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    print(f"Saved {len(combined):,} rows → {out}")
    return combined
