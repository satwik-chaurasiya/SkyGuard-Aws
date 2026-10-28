"""
data_manager.py — Privacy-first data manager for SkyGuard ChatBot.
All user data stays LOCAL, is never transmitted, and is wiped on exit or request.
"""

import os
import shutil
import atexit
import hashlib
import json
from datetime import datetime
from pathlib import Path

# Isolated scratch directory — never shared, never synced
DATA_DIR = Path(__file__).parent / ".scratch_data"
METADATA_FILE = DATA_DIR / "_metadata.json"


class DataManager:
    """Manages user-uploaded datasets with guaranteed cleanup."""

    def __init__(self):
        self._datasets: dict[str, dict] = {}  # id -> {path, name, loaded_at, rows, cols}
        self._ensure_dir()
        # Register cleanup so data is wiped even on crash / Ctrl+C
        atexit.register(self.purge_all)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _ensure_dir(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest(self, src_path: str) -> dict:
        """Copy a user file into the scratch directory and return metadata."""
        import pandas as pd

        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {src_path}")

        ext = src.suffix.lower()
        if ext not in (".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"):
            raise ValueError(f"Unsupported file type: {ext}. "
                             f"Supported: csv, xlsx, xls, json, parquet, tsv")

        # Unique ID from content hash (no duplicates)
        file_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        dest = DATA_DIR / f"{file_hash}{ext}"
        shutil.copy2(src, dest)

        # Load into pandas to get shape info
        df = self._read_file(dest)
        meta = {
            "id": file_hash,
            "name": src.name,
            "path": str(dest),
            "loaded_at": datetime.now().isoformat(),
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        self._datasets[file_hash] = meta
        self._save_metadata()
        return meta

    def _read_file(self, path: Path):
        """Read a data file into a pandas DataFrame."""
        import pandas as pd

        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Cannot read {ext}")

    def get_dataframe(self, dataset_id: str):
        """Return the pandas DataFrame for a loaded dataset."""
        import pandas as pd

        meta = self._datasets.get(dataset_id)
        if not meta:
            raise KeyError(f"Dataset '{dataset_id}' not found. Load it first.")
        return self._read_file(Path(meta["path"]))

    # ------------------------------------------------------------------
    # Listing / info
    # ------------------------------------------------------------------
    def list_datasets(self) -> list[dict]:
        return list(self._datasets.values())

    def dataset_summary(self, dataset_id: str) -> str:
        """Return a human-readable summary of a dataset."""
        import pandas as pd

        df = self.get_dataframe(dataset_id)
        meta = self._datasets[dataset_id]
        lines = [
            f"📊 Dataset: {meta['name']}",
            f"   Rows: {meta['rows']:,}  |  Columns: {meta['cols']}",
            f"   Loaded at: {meta['loaded_at']}",
            f"   Columns: {', '.join(meta['columns'])}",
            "",
            "--- Numeric Summary ---",
            df.describe().to_string(),
            "",
            "--- Missing Values ---",
            df.isnull().sum().to_string(),
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete_dataset(self, dataset_id: str):
        """Securely delete a single dataset."""
        meta = self._datasets.pop(dataset_id, None)
        if meta:
            p = Path(meta["path"])
            if p.exists():
                # Overwrite with zeros before unlinking (basic secure delete)
                size = p.stat().st_size
                with open(p, "wb") as f:
                    f.write(b"\x00" * size)
                p.unlink()
        self._save_metadata()

    def purge_all(self):
        """Wipe every file in the scratch directory — called on exit."""
        if DATA_DIR.exists():
            for f in DATA_DIR.iterdir():
                try:
                    if f.is_file():
                        # Overwrite then delete
                        size = f.stat().st_size
                        with open(f, "wb") as fh:
                            fh.write(b"\x00" * size)
                        f.unlink()
                except Exception:
                    pass
            try:
                shutil.rmtree(DATA_DIR, ignore_errors=True)
            except Exception:
                pass
        self._datasets.clear()

    # ------------------------------------------------------------------
    # Metadata persistence (within session only — wiped on exit)
    # ------------------------------------------------------------------
    def _save_metadata(self):
        self._ensure_dir()
        METADATA_FILE.write_text(json.dumps(self._datasets, indent=2))
