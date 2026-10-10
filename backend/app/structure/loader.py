"""Structure loader for retrieving and verifying local or remote macromolecular structures."""

import os
import urllib.request
from pathlib import Path
from typing import Union, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
STRUCTURES_DIR = BASE_DIR / "data" / "structures"
STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)


class StructureLoadError(Exception):
    """Raised when a macromolecular structure cannot be found, downloaded, or loaded."""
    pass


def normalize_pdb_id(identifier: str) -> str:
    """Extract standard 4-character uppercase PDB ID from filename or string."""
    clean = str(identifier).strip()
    clean = clean.replace("pdb_", "").replace("PDB_", "")
    stem = Path(clean).stem
    if len(stem) >= 4:
        return stem[-4:].upper()
    return stem.upper()


def resolve_structure_path(
    identifier_or_path: Union[str, Path],
    allow_download: bool = True
) -> Path:
    """Resolve file path for a structure, searching local cache or fetching from RCSB PDB.

    Args:
        identifier_or_path: PDB ID (e.g. '1EJO') or direct path to a .cif/.mmcif/.pdb file.
        allow_download: Whether to fetch from RCSB PDB if not found locally.

    Returns:
        Path to the verified existing structure file.

    Raises:
        StructureLoadError: If file cannot be found or downloaded.
    """
    path = Path(identifier_or_path)

    # 1. Direct file path check
    if path.exists() and path.is_file():
        if path.stat().st_size == 0:
            raise StructureLoadError(f"Structure file at '{path}' is empty.")
        return path

    # 2. Check local data/structures/ cache
    pdb_id = normalize_pdb_id(str(identifier_or_path))
    if len(pdb_id) != 4:
        raise StructureLoadError(f"Invalid PDB ID format: '{identifier_or_path}'")

    for ext in [".cif", ".mmcif", ".pdb"]:
        candidate = STRUCTURES_DIR / f"{pdb_id}{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    # 3. Remote download fallback if enabled
    if not allow_download:
        raise StructureLoadError(f"Structure '{pdb_id}' not found locally and download is disabled.")

    download_dest = STRUCTURES_DIR / f"{pdb_id}.cif"
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    try:
        urllib.request.urlretrieve(url, download_dest)
        if not download_dest.exists() or download_dest.stat().st_size == 0:
            if download_dest.exists():
                download_dest.unlink()
            raise StructureLoadError(f"Downloaded structure '{pdb_id}' is empty or failed.")
        return download_dest
    except Exception as exc:
        if download_dest.exists():
            download_dest.unlink()
        raise StructureLoadError(f"Failed to retrieve structure '{pdb_id}' from RCSB PDB: {exc}") from exc
