#!/usr/bin/env python3
"""
Merge MIDAS Open BADC-CSV daily files (one per year) into a single CSV.

Works for:
  * uk-daily-temperature-obs  → max_air_temp, min_air_temp
  * uk-daily-rain-obs         → prcp_amt
  * uk-daily-weather-obs      → snow_depth, fresh_snow, hail, thunder, etc.

Usage:
  1. Put this script in the folder that contains your downloaded CSV files
     (or anywhere — you just need to know the path to them).
  2. Edit the INPUT_DIR below to point to the folder with the yearly CSVs.
     The script searches all subfolders, so it's fine to point at a top-level
     folder if you downloaded the whole tree (qc-version-1 etc.).
  3. Run:  python3 merge_midas.py
  4. It will produce ONE file:  midas_merged_daily.csv

Requires only the Python standard library and pandas:
  pip install pandas
"""

from pathlib import Path
import sys
import pandas as pd

# ----------------------------------------------------------------------------
# EDIT THIS LINE: path to the folder containing your downloaded CSV files
# (pointing at a parent folder is fine — script will find all CSVs underneath)
# ----------------------------------------------------------------------------
INPUT_DIR  = Path("/Users/dgk28/Library/Mobile Documents/com~apple~CloudDocs/Persoenliches/26_05_Cambridge Weather & Climate analysis/data/ground_temperature_CUBG_NIABfrom2019")
OUTPUT_CSV = Path("midas_merged_daily_groundtemp.csv")


def read_badc_csv(path: Path) -> pd.DataFrame:
    """Read one BADC-CSV file. The data sits between a `data` line
    and an `end data` line. The line immediately after `data` is the
    column-name header."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Find the "data" marker line
    start = None
    end = None
    for i, line in enumerate(lines):
        token = line.strip().lower().split(",")[0]
        if token == "data" and start is None:
            start = i + 1  # header row is immediately after
        elif token == "end data":
            end = i
            break

    if start is None:
        # Some files don't have the BADC wrapper — fall back to plain CSV
        return pd.read_csv(path, low_memory=False)

    if end is None:
        end = len(lines)

    # The first line after "data" is the column header
    header_line = lines[start].strip()
    data_lines = lines[start + 1 : end]
    if not data_lines:
        return pd.DataFrame()

    from io import StringIO
    block = header_line + "\n" + "".join(data_lines)
    return pd.read_csv(StringIO(block), low_memory=False)


def main() -> None:
    if not INPUT_DIR.exists():
        sys.exit(f"ERROR: folder {INPUT_DIR!s} does not exist. "
                 f"Edit INPUT_DIR at the top of the script.")

    files = sorted(INPUT_DIR.rglob("*.csv"))
    # Skip station metadata files (they're not yearly data)
    files = [p for p in files if "station-metadata" not in p.name.lower()]
    if not files:
        sys.exit(f"ERROR: no CSV files found under {INPUT_DIR!s}.")

    print(f"Found {len(files)} CSV files. Reading...")

    frames = []
    for i, p in enumerate(files, 1):
        try:
            df = read_badc_csv(p)
            if df.empty:
                continue
            df["__source_file"] = p.name
            frames.append(df)
        except Exception as e:
            print(f"  ! Skipping {p.name}: {e}")
        if i % 20 == 0 or i == len(files):
            print(f"  ... {i}/{len(files)} files read")

    if not frames:
        sys.exit("ERROR: no readable data found.")

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Parse the observation timestamp column to a real date (column name
    # varies a bit between dataset types: ob_end_time, ob_date, etc.)
    time_cols = [c for c in merged.columns
                 if c.lower() in {"ob_end_time", "ob_date", "ob_time"}]
    if time_cols:
        tc = time_cols[0]
        merged[tc] = pd.to_datetime(merged[tc], errors="coerce")
        # Some files have multiple QC versions for the same day — keep the
        # latest one (highest version_num if present)
        if "version_num" in merged.columns:
            merged = (merged.sort_values([tc, "version_num"])
                            .drop_duplicates(subset=[tc], keep="last"))
        else:
            merged = merged.drop_duplicates(subset=[tc], keep="last")
        merged = merged.sort_values(tc).reset_index(drop=True)
        # Add a plain ISO date column for convenience
        merged.insert(0, "date", merged[tc].dt.date)

    print(f"\nMerged: {len(merged):,} rows, {len(merged.columns)} columns.")
    print(f"Columns: {list(merged.columns)}")
    if "date" in merged.columns:
        print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Wrote {OUTPUT_CSV.resolve()}")
    print(f"   Size: {OUTPUT_CSV.stat().st_size / 1024:.0f} KB")
    print("\nUpload this single file to the chat and we can pick up from there.")


if __name__ == "__main__":
    main()
