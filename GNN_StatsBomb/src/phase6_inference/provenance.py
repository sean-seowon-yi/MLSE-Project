"""
Embedding artifact provenance and consistency checks.

This module records lightweight metadata at embedding-generation time and
validates it before downstream diagnostics/analysis that combine:
  - embeddings on disk
  - a checkpoint loaded into the model
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json


MANIFEST_FILENAME = "embedding_manifest.json"


def _norm_path(p: Path) -> str:
    return str(p.resolve()).lower()


def build_manifest(
    *,
    checkpoint_path: Path,
    embedding_output_dir: Path,
    graphs_filename: str,
    split_context_edges: bool,
    tag: str,
    inference_split: str = "all",
) -> Dict[str, object]:
    try:
        ckpt_stat = checkpoint_path.stat()
    except OSError:
        ckpt_stat = None
    emb_file = embedding_output_dir / "player_embeddings.npy"
    try:
        emb_stat = emb_file.stat() if emb_file.exists() else None
    except OSError:
        emb_stat = None

    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_mtime_ns": int(ckpt_stat.st_mtime_ns) if ckpt_stat else -1,
        "checkpoint_size_bytes": int(ckpt_stat.st_size) if ckpt_stat else -1,
        "embedding_file": str(emb_file.resolve()) if emb_file.exists() else "",
        "embedding_mtime_ns": int(emb_stat.st_mtime_ns) if emb_stat else -1,
        "graphs_filename": graphs_filename,
        "split_context_edges": bool(split_context_edges),
        "tag": tag or "",
        "inference_split": inference_split,
    }


def write_manifest(output_dir: Path, manifest: Dict[str, object]) -> None:
    path = output_dir / MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(output_dir: Path) -> Optional[Dict[str, object]]:
    path = output_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    graphs_filename: str,
    split_context_edges: bool,
) -> Tuple[bool, str]:
    """
    Validate that embeddings in `output_dir` are compatible with the model/checkpoint.

    Returns:
      (True, "ok") on success
      (False, reason) on mismatch/missing/stale artifacts
    """
    if not checkpoint_path.exists():
        return False, f"Checkpoint not found: {checkpoint_path}"

    manifest = load_manifest(output_dir)
    if manifest is None:
        return (
            False,
            f"Missing {MANIFEST_FILENAME} in {output_dir}. "
            "Run --mode inference to regenerate embeddings with provenance.",
        )

    manifest_ckpt = str(manifest.get("checkpoint_path", ""))
    expected_ckpt = str(checkpoint_path.resolve())
    if _norm_path(Path(manifest_ckpt)) != _norm_path(Path(expected_ckpt)):
        return (
            False,
            "Embedding/checkpoint mismatch: manifest checkpoint path differs "
            f"(manifest={manifest_ckpt}, expected={expected_ckpt}).",
        )

    manifest_graphs = str(manifest.get("graphs_filename", ""))
    if manifest_graphs != graphs_filename:
        return (
            False,
            "Embedding/graph mismatch: manifest graphs filename differs "
            f"(manifest={manifest_graphs}, expected={graphs_filename}).",
        )

    manifest_split = bool(manifest.get("split_context_edges", False))
    if manifest_split != bool(split_context_edges):
        return (
            False,
            "Embedding graph-architecture mismatch: split_context_edges differs "
            f"(manifest={manifest_split}, expected={split_context_edges}).",
        )

    emb_mtime_ns = int(manifest.get("embedding_mtime_ns", -1))
    try:
        ckpt_mtime_ns = int(checkpoint_path.stat().st_mtime_ns)
    except OSError:
        return False, f"Cannot stat checkpoint: {checkpoint_path}"
    if emb_mtime_ns >= 0 and ckpt_mtime_ns > emb_mtime_ns:
        return (
            False,
            "Checkpoint is newer than embeddings. Re-run --mode inference "
            "so embeddings match the current checkpoint.",
        )

    return True, "ok"
