import pandas as pd
import numpy as np
from pathlib import Path
import zipfile, sys

def preprocess_cicids(zip_path, out_path):
    print(f"Extracting {zip_path}...")
    extract_dir = Path("data/raw/cicids_extracted")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    csvs = list(extract_dir.rglob("*.csv"))
    print(f"Found {len(csvs)} CSV files:")
    for c in sorted(csvs):
        print(f"  {c.name}")

    dfs = []
    for csv in sorted(csvs):
        try:
            df = pd.read_csv(csv, low_memory=False)
            df["source_file"] = csv.name
            dfs.append(df)
            print(f"  Loaded {csv.name}: {len(df)} rows")
        except Exception as e:
            print(f"  SKIP {csv.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = combined.columns.str.strip()
    print(f"Total rows: {len(combined)}")

    src_col   = next((c for c in combined.columns if "src" in c.lower() and "ip" in c.lower()), None)
    dst_col   = next((c for c in combined.columns if "dst" in c.lower() and "ip" in c.lower()), None)
    label_col = next((c for c in combined.columns if "label" in c.lower()), None)
    len_col   = next((c for c in combined.columns if "total" in c.lower() and ("length" in c.lower() or "fwd" in c.lower())), None)

    print(f"src={src_col} dst={dst_col} label={label_col} bytes={len_col}")
    print("Labels:", combined[label_col].value_counts().to_dict())

    out = pd.DataFrame()
    out["src_ip"]    = combined[src_col].astype(str)
    out["dst_ip"]    = combined[dst_col].astype(str)
    out["bytes"]     = pd.to_numeric(combined[len_col], errors="coerce").fillna(0).astype(int)
    out["label"]     = (combined[label_col].str.strip() != "BENIGN").astype(int)
    out["timestamp"] = np.linspace(0, len(out) * 2, len(out))

    out = out[out["src_ip"].str.match(r"^\d+\.\d+\.\d+\.\d+$")]
    out = out[out["dst_ip"].str.match(r"^\d+\.\d+\.\d+\.\d+$")]
    out = out.reset_index(drop=True)

    print(f"After cleaning: {len(out)} rows")
    print(f"Anomalies: {out['label'].sum()} ({out['label'].mean()*100:.1f}%)")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

preprocess_cicids("data/raw/MachineLearningCSV.zip", "data/raw/cicids2017_processed.csv")
