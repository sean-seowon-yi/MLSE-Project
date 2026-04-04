"""Look up player IDs by (partial) name.

Supports accent-insensitive matching (e.g. ``mbappe`` matches ``Mbappé``) and
optional fuzzy matching for typos and reorderings (via rapidfuzz).

Usage
-----
    python find_player.py messi
    python find_player.py "de bruyne"
    python find_player.py mbappe
    python find_player.py --tag split_ctx ronaldo
    python find_player.py --all putellas
    python find_player.py --no-fuzzy bruyne
    python find_player.py --min-score 65 bruine

By default searches ``embeddings/baseline/player_info.parquet``.
Use ``--tag <tag>`` for another pipeline, or ``--all`` for raw event metadata.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz, process


def _fold(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace, soften punctuation."""
    if not isinstance(s, str):
        s = str(s)
    decomposed = unicodedata.normalize("NFD", s)
    without_marks = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )
    # Hyphens / apostrophes (incl. right single quote) → space for "Jean Claude" etc.
    without_marks = re.sub(r"[-'\u2019]", " ", without_marks)
    without_marks = re.sub(r"\s+", " ", without_marks).strip().lower()
    return without_marks


def _substring_matches(df: pd.DataFrame, q_fold: str) -> Set[int]:
    """Integer row positions (0..n-1, iloc) where folded name contains *q_fold*."""
    if not q_fold:
        return set()
    pat = re.escape(q_fold)
    mask = df["_fold"].str.contains(pat, regex=True, na=False)
    return {i for i, ok in enumerate(mask) if ok}


def _fuzzy_matches(
    folded_names: List[str],
    q_fold: str,
    limit: int,
    score_cutoff: float,
) -> List[Tuple[int, float]]:
    """List of (position in *folded_names*, score) from rapidfuzz."""
    if not q_fold or len(q_fold) < 2:
        return []
    results = process.extract(
        q_fold,
        folded_names,
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=score_cutoff,
    )
    out: List[Tuple[int, float]] = []
    for _choice, score, pos in results:
        out.append((int(pos), float(score)))
    return out


def find_players(
    df: pd.DataFrame,
    query: str,
    *,
    fuzzy_limit: int = 30,
    min_score: float = 72.0,
    no_fuzzy: bool = False,
) -> pd.DataFrame:
    """Return unique players matching *query*, sorted: substring first, then fuzzy score."""
    q_fold = _fold(query)
    if len(q_fold) < 2:
        raise ValueError("Query must be at least 2 characters after normalisation.")

    work = df.copy()
    work["_fold"] = work["player_name"].map(_fold)
    folded_list = work["_fold"].tolist()

    sub_ilocs = _substring_matches(work, q_fold)
    fuzzy_scores: Dict[int, float] = {}
    if not no_fuzzy:
        for pos, sc in _fuzzy_matches(
            folded_list, q_fold, limit=fuzzy_limit, score_cutoff=min_score,
        ):
            fuzzy_scores[pos] = max(fuzzy_scores.get(pos, 0.0), sc)

    all_ilocs = sub_ilocs | set(fuzzy_scores.keys())

    if not all_ilocs:
        return pd.DataFrame()

    def sort_key(iloc: int) -> Tuple[int, float, str]:
        in_sub = iloc in sub_ilocs
        tier = 0 if in_sub else 1
        sc = 100.0 if in_sub else fuzzy_scores.get(iloc, 0.0)
        name = work.iloc[iloc]["player_name"]
        return (tier, -sc, str(name))

    ordered = sorted(all_ilocs, key=sort_key)
    out = work.iloc[ordered].copy()
    out["match"] = [
        "substring" if i in sub_ilocs else f"fuzzy ({fuzzy_scores.get(i, 0):.0f})"
        for i in ordered
    ]
    out = out.drop(columns=["_fold"])
    return out.drop_duplicates(subset=["player_id"], keep="first")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find player IDs by name (accent-insensitive + fuzzy).",
    )
    parser.add_argument(
        "name", type=str,
        help="Full or partial player name.",
    )
    parser.add_argument(
        "--tag", type=str, default="baseline",
        help="Pipeline tag for embeddings (default: baseline).",
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        dest="search_all",
        help="Search raw event metadata (all players).",
    )
    parser.add_argument(
        "--no-fuzzy", action="store_true", default=False,
        help="Only substring matching (after accent normalisation).",
    )
    parser.add_argument(
        "--min-score", type=float, default=72.0,
        metavar="N",
        help="Minimum rapidfuzz WRatio for fuzzy matches (default: 72).",
    )
    parser.add_argument(
        "--fuzzy-limit", type=int, default=30,
        help="Max fuzzy candidates to consider (default: 30).",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    if args.search_all:
        meta_path = base / "processed_data" / "event_metadata.parquet"
        if not meta_path.exists():
            print(f"Metadata not found: {meta_path}")
            print("Run  python main.py --mode prepare  first.")
            sys.exit(1)
        df = pd.read_parquet(
            meta_path,
            columns=["player_id", "player_name", "position_name", "team_name"],
        )
        df = df.dropna(subset=["player_id"])
        df["player_id"] = df["player_id"].astype(int)
        counts = df.groupby("player_id").size().rename("n_events")
        df = df.drop_duplicates(subset=["player_id"])
        df = df.merge(counts, left_on="player_id", right_index=True)
    else:
        emb_dir = base / "embeddings" / args.tag
        info_path = emb_dir / "player_info.parquet"
        if not info_path.exists():
            print(f"Player info not found: {info_path}")
            print(f"Run  python main.py --mode inference --tag {args.tag}  first,")
            print("or use --all to search the raw event metadata.")
            sys.exit(1)
        df = pd.read_parquet(info_path)

    try:
        hits = find_players(
            df,
            args.name,
            fuzzy_limit=args.fuzzy_limit,
            min_score=args.min_score,
            no_fuzzy=args.no_fuzzy,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    if hits.empty:
        print(f'No players found matching "{args.name}".')
        print("Try a shorter substring, lower --min-score, or omit --no-fuzzy.")
        sys.exit(0)

    show_cols = ["player_id", "player_name", "match"]
    if "position_name" in hits.columns:
        show_cols.append("position_name")
    if "team_name" in hits.columns:
        show_cols.append("team_name")
    if "n_possessions" in hits.columns:
        show_cols.append("n_possessions")
    elif "n_events" in hits.columns:
        show_cols.append("n_events")

    print(f'\nMatches for "{args.name}" ({len(hits)} found):\n')
    print(hits[show_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
