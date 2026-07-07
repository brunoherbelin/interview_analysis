#!/usr/bin/env python3
"""
SenticNet knowledge base loader.

By default the offline `senticnet` pip package (SenticNet 6, ~200k concepts) is
used. When `config.senticnet.download` is true, the newer full SenticNet release
is fetched from `config.senticnet.url` (a zip containing `senticnet.py`) into
`config.senticnet.path`, parsed once, and cached — so re-running the analysis on
any machine reproduces the same knowledge base without keeping the raw file in the
repo. `ConceptAffect` (src/text/opinions.py) reads whichever source this returns.

The downloaded `senticnet.py` is a large `senticnet['concept'] = [...]` literal
whose few emoji entries are not valid Python, so it is parsed line-by-line (bad
rows skipped) rather than imported.
"""

import ast
import re
import pickle
import urllib.request
from functools import lru_cache
from pathlib import Path

from src.config import config, logger

DEFAULT_URL = "https://sentic.net/senticnet.zip"
# senticnet['concept'] = ['introspection','temper','attitude','sensitivity',
#                         'primary_emotion','secondary_emotion','polarity_label',
#                         'polarity_value', semantics...]
_LINE = re.compile(r"^senticnet\[(.+?)\]\s*=\s*(\[.*\])\s*$")


def _download_senticnet_py(dest: Path, url: str) -> Path:
    """Fetch the SenticNet zip and extract `senticnet.py` to `dest`."""
    import zipfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / "_senticnet_download.zip"
    logger.info(f"Downloading SenticNet from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(tmp) as z:
        member = next((m for m in z.namelist() if m.endswith("senticnet.py")), None)
        if not member:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"senticnet.py not found inside {url}")
        with z.open(member) as src, open(dest, "wb") as out:
            out.write(src.read())
    tmp.unlink(missing_ok=True)
    logger.info(f"Extracted SenticNet knowledge base to {dest}")
    return dest


def _parse_senticnet_py(py_path: Path) -> dict:
    """Parse `senticnet['concept'] = [...]` lines, skipping malformed rows."""
    import warnings

    data = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # stray escapes in semantics strings
        with open(py_path, encoding="utf-8") as f:
            for line in f:
                m = _LINE.match(line.rstrip("\n"))
                if not m:
                    continue
                try:
                    data[ast.literal_eval(m.group(1))] = ast.literal_eval(m.group(2))
                except Exception:
                    continue
    return data


@lru_cache(maxsize=1)
def load_concept_data():
    """Return the downloaded SenticNet as {concept: [...]}, or None to use the pip package."""
    if not config.get("senticnet.download", False):
        return None

    path = Path(config.get("senticnet.path", "data/senticnet")).expanduser()
    cache = path / "senticnet.parsed.pkl"
    if cache.exists():
        try:
            return pickle.load(open(cache, "rb"))
        except Exception:
            logger.warning(f"Could not read SenticNet cache {cache}; rebuilding.")

    try:
        py_path = path / "senticnet.py"
        if not py_path.exists():
            _download_senticnet_py(py_path, config.get("senticnet.url", DEFAULT_URL))
        data = _parse_senticnet_py(py_path)
    except Exception as e:
        logger.warning(f"SenticNet download/parse failed ({e}); "
                       f"falling back to bundled senticnet package.")
        return None

    if not data:
        logger.warning("Parsed SenticNet is empty; falling back to bundled package.")
        return None

    try:
        pickle.dump(data, open(cache, "wb"))
    except Exception:
        pass
    logger.info(f"SenticNet: using downloaded knowledge base ({len(data)} concepts).")
    return data
