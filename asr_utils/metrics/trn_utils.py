"""
Functions for working with doing evaluation with .trn files.
Adapted from SAPC-template.
"""

from pathlib import Path
import csv
import re
import subprocess
import os
import shutil
from typing import Dict, List


def load_trn_map(path: Path) -> dict[str, str]:
    """Reads .trn file (hyp, utt_id), and turns it into a dictionary with the id as the key.

    Args:
        path: Path to .trn file.

    Returns:
        dict[str, str]: Dictionary with utterance id as the key, hyp as value.
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            hyp = re.match(r"^(.*)\s+\(([^()]+)\)$", line)
            if hyp:
                out[hyp.group(2)] = hyp.group(1)
    return out


def load_manifest_ids(csv_path: str) -> List[str]:
    """Read manifest CSV and return ordered list of IDs."""
    ids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
    return ids


def load_predictions(csv_path: str, hyp_col: str) -> Dict[str, str]:
    """Load predictions (id -> raw_hyp_text) from CSV file."""
    preds = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["id"]
            text = row.get(hyp_col, "")
            preds[uid] = text
    return preds


def write_trn(texts, ids, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for text, uid in zip(texts, ids):
            f.write(f"{text} ({uid})\n")


def write_subset_trn(trn_map, utt_ids, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for uid in utt_ids:
            f.write(f"{trn_map[uid]} ({uid})\n")


def run_sclite(ref_trn: str, hyp_trn: str, sgml_out: str) -> bool:
    """
    Run sclite alignment: ref vs hyp -> SGML.
    Returns True on success.
    """
    try:
        result = subprocess.run(
            [
                "sclite",
                "-r",
                ref_trn,
                "trn",
                "-h",
                hyp_trn,
                "trn",
                "-i",
                "wsj",
                "-o",
                "all",
                "sgml",
            ],
            capture_output=True,
            timeout=600,
        )
        # sclite writes <hyp_trn>.sgml
        generated_sgml = hyp_trn + ".sgml"
        if os.path.isfile(generated_sgml):
            shutil.move(generated_sgml, sgml_out)
            # Cleanup other sclite artifacts
            for ext in [".sys", ".raw", ".pra"]:
                artifact = hyp_trn + ext
                if os.path.isfile(artifact):
                    os.remove(artifact)
            return True
        return False
    except Exception as e:
        print(f"sclite error: {e}")
        return False
