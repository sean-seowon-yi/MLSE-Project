"""
Test analysis: sample 10 male + 10 female players, find each player's
top-1 same-gender substitute via GNN cosine similarity, and compare
their FIFA stats side-by-side.

Usage (standalone):
    python test_fifa_comparison.py

Can also be called programmatically via ``run_comparison()``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ── Default paths (used by standalone execution) ──────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GNN_DIR = PROJECT_ROOT / "GNN_StatsBomb"
FIFA_DIR = PROJECT_ROOT / "FIFA_data"

_DEFAULT_EMBEDDINGS_DIR = GNN_DIR / "embeddings" / "baseline"
_DEFAULT_PROCESSED_DIR = GNN_DIR / "processed_data"
_DEFAULT_STATSBOMB_BASE = PROJECT_ROOT / "StatsBomb" / "data"
_DEFAULT_MALE_FIFA_CSV = FIFA_DIR / "statsbomb_male_players_fifa.csv"
_DEFAULT_FEMALE_FIFA_CSV = FIFA_DIR / "statsbomb_female_players_fifa.csv"
_DEFAULT_OUTPUT_FILE = GNN_DIR / "evaluations" / "baseline" / "fifa_comparison" / "test_fifa_comparison.txt"

SEED = None

FAMOUS_MALE_IDS = [
    5503,   # Lionel Messi
    5207,   # Cristiano Ronaldo
    3009,   # Kylian Mbappé
    5463,   # Luka Modrić
    30714,  # Jude Bellingham
    10955,  # Harry Kane
]

FAMOUS_FEMALE_IDS = [
    15284,  # Aitana Bonmatí
    10178,  # Lucy Bronze
    47521,  # Alessia Russo
    10143,  # Alexia Putellas
    15555,  # Lauren Hemp
    4961,   # Sam Kerr
]

POSITION_GROUPS = {
    "Goalkeeper": ["Goalkeeper"],
    "Defender": [
        "Center Back", "Left Back", "Left Center Back", "Left Wing Back",
        "Right Back", "Right Center Back", "Right Wing Back",
    ],
    "Midfielder": [
        "Center Attacking Midfield", "Center Defensive Midfield",
        "Center Midfield", "Left Attacking Midfield", "Left Center Midfield",
        "Left Defensive Midfield", "Left Midfield", "Right Attacking Midfield",
        "Right Center Midfield", "Right Defensive Midfield", "Right Midfield",
    ],
    "Forward": [
        "Center Forward", "Left Center Forward", "Left Wing",
        "Right Center Forward", "Right Wing", "Secondary Striker",
    ],
}

POS_TO_GROUP: Dict[str, str] = {}
for grp, positions in POSITION_GROUPS.items():
    for pos in positions:
        POS_TO_GROUP[pos] = grp

MAIN_STATS = [
    "fifa_pace", "fifa_shooting", "fifa_passing",
    "fifa_dribbling", "fifa_defending", "fifa_physic",
]

GK_STATS = [
    "fifa_goalkeeping_diving", "fifa_goalkeeping_handling",
    "fifa_goalkeeping_kicking", "fifa_goalkeeping_positioning",
    "fifa_goalkeeping_reflexes",
]

DETAIL_STATS = [
    "fifa_attacking_crossing", "fifa_attacking_finishing",
    "fifa_attacking_heading_accuracy", "fifa_attacking_short_passing",
    "fifa_attacking_volleys",
    "fifa_skill_dribbling", "fifa_skill_curve", "fifa_skill_fk_accuracy",
    "fifa_skill_long_passing", "fifa_skill_ball_control",
    "fifa_movement_acceleration", "fifa_movement_sprint_speed",
    "fifa_movement_agility", "fifa_movement_reactions",
    "fifa_movement_balance",
    "fifa_power_shot_power", "fifa_power_jumping", "fifa_power_stamina",
    "fifa_power_strength", "fifa_power_long_shots",
    "fifa_mentality_aggression", "fifa_mentality_interceptions",
    "fifa_mentality_positioning", "fifa_mentality_vision",
    "fifa_mentality_penalties", "fifa_mentality_composure",
    "fifa_defending_marking_awareness", "fifa_defending_standing_tackle",
    "fifa_defending_sliding_tackle",
    "fifa_goalkeeping_diving", "fifa_goalkeeping_handling",
    "fifa_goalkeeping_kicking", "fifa_goalkeeping_positioning",
    "fifa_goalkeeping_reflexes",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def build_gender_map(
    processed_dir: Optional[Path] = None,
    statsbomb_base: Optional[Path] = None,
) -> Dict[int, str]:
    """Build player_id -> gender from event metadata + competitions.json."""
    import json
    p_dir = processed_dir or _DEFAULT_PROCESSED_DIR
    sb_base = statsbomb_base or _DEFAULT_STATSBOMB_BASE
    meta = pd.read_parquet(
        p_dir / "event_metadata.parquet",
        columns=["player_id", "competition_id"],
    )
    with open(sb_base / "competitions.json", encoding="utf-8") as f:
        comps = json.load(f)
    comp_gender = {
        c["competition_id"]: c["competition_gender"]
        for c in comps if c.get("match_available_360")
    }
    meta["gender"] = meta["competition_id"].map(comp_gender)
    return (
        meta.dropna(subset=["gender"])
        .groupby("player_id")["gender"]
        .first()
        .to_dict()
    )


def load_fifa(path: Path) -> pd.DataFrame:
    """Load a FIFA CSV and deduplicate: keep the row with the highest overall
    per unique sb_player_id."""
    df = pd.read_csv(path)
    df["fifa_overall"] = pd.to_numeric(df["fifa_overall"], errors="coerce")
    df = df.sort_values("fifa_overall", ascending=False).drop_duplicates(
        subset="sb_player_id", keep="first",
    )
    return df.reset_index(drop=True)


def stratified_sample(
    player_ids: np.ndarray,
    player_info: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> List[int]:
    """Sample *n* players, stratified across position groups."""
    info_sub = player_info[player_info["player_id"].isin(player_ids)].copy()
    info_sub["group"] = info_sub["position_name"].map(POS_TO_GROUP).fillna("Other")

    groups = [g for g in ["Goalkeeper", "Defender", "Midfielder", "Forward"]
              if g in info_sub["group"].values]
    if not groups:
        idx = rng.choice(len(player_ids), size=min(n, len(player_ids)), replace=False)
        return player_ids[idx].tolist()

    per_group = max(1, n // len(groups))
    remainder = n - per_group * len(groups)
    sampled: List[int] = []

    for i, g in enumerate(groups):
        pool = info_sub[info_sub["group"] == g]["player_id"].values
        k = per_group + (1 if i < remainder else 0)
        k = min(k, len(pool))
        if k > 0:
            chosen = rng.choice(pool, size=k, replace=False)
            sampled.extend(chosen.tolist())

    while len(sampled) < n:
        remaining = [p for p in player_ids if p not in sampled]
        if not remaining:
            break
        extra = rng.choice(remaining, size=min(n - len(sampled), len(remaining)),
                           replace=False)
        sampled.extend(extra.tolist())

    return sampled[:n]


def _sample_with_famous(
    overlap: np.ndarray,
    player_info: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    famous_ids: List[int],
    fifa_pids: set,
    n_famous: int = 2,
) -> List[int]:
    """Sample *n* players, guaranteeing up to *n_famous* well-known players."""
    overlap_set = set(int(p) for p in overlap)
    eligible_famous = [
        pid for pid in famous_ids
        if pid in overlap_set and pid in fifa_pids
    ]
    chosen_famous = eligible_famous[:min(n_famous, len(eligible_famous))]

    remaining_n = n - len(chosen_famous)
    remaining_pool = np.array([p for p in overlap if int(p) not in set(chosen_famous)])

    if remaining_n > 0 and len(remaining_pool) > 0:
        extra = stratified_sample(remaining_pool, player_info, remaining_n, rng)
        return chosen_famous + [p for p in extra if p not in set(chosen_famous)]

    return chosen_famous[:n]


def find_top1_same_gender_with_fifa(
    query_pid: int,
    sims_row: np.ndarray,
    pids: np.ndarray,
    query_idx: int,
    gender_map: Dict[int, str],
    fifa_pids: set,
) -> Optional[Tuple[int, float]]:
    """Return (player_id, similarity) for the best same-gender candidate
    that also has FIFA data, or None."""
    query_gender = gender_map.get(int(query_pid))
    scores = sims_row.copy()
    scores[query_idx] = -np.inf

    order = np.argsort(scores)[::-1]
    for idx in order:
        cand_pid = int(pids[idx])
        if scores[idx] == -np.inf:
            break
        if gender_map.get(cand_pid) != query_gender:
            continue
        if cand_pid not in fifa_pids:
            continue
        return cand_pid, float(scores[idx])
    return None


def stat_val(val) -> Optional[float]:
    """Safely convert a FIFA stat value to float."""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def format_pair(
    query_row: pd.Series,
    sub_row: pd.Series,
    cosine_sim: float,
    pair_num: int,
    gender: str,
) -> Tuple[List[str], Dict]:
    """Format a single query-substitute comparison. Returns (lines, summary_dict)."""
    lines: List[str] = []
    lines.append(f"{'='*80}")
    lines.append(f"  PAIR {pair_num} ({gender.upper()})")
    lines.append(f"{'='*80}")

    q_name = _safe_label(str(query_row.get("fifa_long_name", query_row.get("sb_player_name", "?"))))
    s_name = _safe_label(str(sub_row.get("fifa_long_name", sub_row.get("sb_player_name", "?"))))
    q_pos = query_row.get("fifa_player_positions", query_row.get("sb_position", "?"))
    s_pos = sub_row.get("fifa_player_positions", sub_row.get("sb_position", "?"))
    q_club = query_row.get("fifa_club_name", "?")
    s_club = sub_row.get("fifa_club_name", "?")
    q_ovr = stat_val(query_row.get("fifa_overall"))
    s_ovr = stat_val(sub_row.get("fifa_overall"))
    q_age = stat_val(query_row.get("fifa_age"))
    s_age = stat_val(sub_row.get("fifa_age"))
    q_sb_pos = str(query_row.get("sb_position", ""))
    s_sb_pos = str(sub_row.get("sb_position", ""))

    lines.append(f"  {'':30s} {'QUERY':>20s}   {'SUBSTITUTE':>20s}   {'DIFF':>8s}")
    lines.append(f"  {'-'*30} {'-'*20}   {'-'*20}   {'-'*8}")
    lines.append(f"  {'Name':30s} {str(q_name):>20s}   {str(s_name):>20s}")
    lines.append(f"  {'FIFA Position':30s} {str(q_pos):>20s}   {str(s_pos):>20s}")
    lines.append(f"  {'SB Position':30s} {str(q_sb_pos):>20s}   {str(s_sb_pos):>20s}")
    lines.append(f"  {'Club':30s} {str(q_club):>20s}   {str(s_club):>20s}")
    lines.append(f"  {'Age':30s} {_fmt(q_age):>20s}   {_fmt(s_age):>20s}   {_diff(q_age, s_age):>8s}")
    lines.append(f"  {'Overall':30s} {_fmt(q_ovr):>20s}   {_fmt(s_ovr):>20s}   {_diff(q_ovr, s_ovr):>8s}")
    lines.append(f"  {'Cosine Similarity':30s} {cosine_sim:>20.4f}")
    lines.append("")

    lines.append(f"  --- Main Stats ---")
    main_diffs = []
    for stat in MAIN_STATS:
        label = stat.replace("fifa_", "").title()
        qv = stat_val(query_row.get(stat))
        sv = stat_val(sub_row.get(stat))
        d = _diff(qv, sv)
        if qv is not None and sv is not None:
            main_diffs.append(abs(qv - sv))
        lines.append(f"  {label:30s} {_fmt(qv):>20s}   {_fmt(sv):>20s}   {d:>8s}")

    mean_main_diff = np.mean(main_diffs) if main_diffs else None
    lines.append(f"  {'Mean Abs Diff (main 6)':30s} {'':>20s}   {'':>20s}   {_fmt(mean_main_diff):>8s}")
    lines.append("")

    lines.append(f"  --- Detailed Sub-Attributes ---")
    detail_diffs = []
    for stat in DETAIL_STATS:
        label = stat.replace("fifa_", "").replace("_", " ").title()
        qv = stat_val(query_row.get(stat))
        sv = stat_val(sub_row.get(stat))
        d = _diff(qv, sv)
        if qv is not None and sv is not None:
            detail_diffs.append(abs(qv - sv))
        lines.append(f"  {label:30s} {_fmt(qv):>20s}   {_fmt(sv):>20s}   {d:>8s}")

    mean_detail_diff = np.mean(detail_diffs) if detail_diffs else None
    lines.append(f"  {'Mean Abs Diff (detail)':30s} {'':>20s}   {'':>20s}   {_fmt(mean_detail_diff):>8s}")
    lines.append("")

    q_grp = POS_TO_GROUP.get(q_sb_pos, "Other")
    s_grp = POS_TO_GROUP.get(s_sb_pos, "Other")
    pos_match = q_grp == s_grp

    q_main_stats = {s: stat_val(query_row.get(s)) for s in MAIN_STATS}
    s_main_stats = {s: stat_val(sub_row.get(s)) for s in MAIN_STATS}
    q_gk_stats = {s: stat_val(query_row.get(s)) for s in GK_STATS}
    s_gk_stats = {s: stat_val(sub_row.get(s)) for s in GK_STATS}
    q_detail_stats = {s: stat_val(query_row.get(s)) for s in DETAIL_STATS}
    s_detail_stats = {s: stat_val(sub_row.get(s)) for s in DETAIL_STATS}
    is_gk_pair = q_grp == "Goalkeeper" or s_grp == "Goalkeeper"

    summary = {
        "query": str(q_name),
        "substitute": str(s_name),
        "gender": gender,
        "q_position": q_sb_pos,
        "s_position": s_sb_pos,
        "q_group": q_grp,
        "s_group": s_grp,
        "pos_group_match": pos_match,
        "cosine_sim": cosine_sim,
        "q_overall": q_ovr,
        "s_overall": s_ovr,
        "ovr_diff": abs(q_ovr - s_ovr) if q_ovr is not None and s_ovr is not None else None,
        "mean_main_diff": mean_main_diff,
        "mean_detail_diff": mean_detail_diff,
        "q_main_stats": q_main_stats,
        "s_main_stats": s_main_stats,
        "q_gk_stats": q_gk_stats,
        "s_gk_stats": s_gk_stats,
        "q_detail_stats": q_detail_stats,
        "s_detail_stats": s_detail_stats,
        "is_gk_pair": is_gk_pair,
    }

    return lines, summary


def _fmt(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "N/A"
    if val == int(val):
        return str(int(val))
    return f"{val:.1f}"


def _diff(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "N/A"
    if (isinstance(a, float) and (np.isnan(a) or np.isinf(a))) or \
       (isinstance(b, float) and (np.isnan(b) or np.isinf(b))):
        return "N/A"
    d = abs(a - b)
    return f"{d:.0f}" if d == int(d) else f"{d:.1f}"


# ── Visualization ──────────────────────────────────────────────────────────

_STAT_LABELS = {
    "fifa_pace": "Pace",
    "fifa_shooting": "Shooting",
    "fifa_passing": "Passing",
    "fifa_dribbling": "Dribbling",
    "fifa_defending": "Defending",
    "fifa_physic": "Physical",
}

_GK_STAT_LABELS = {
    "fifa_goalkeeping_diving": "Diving",
    "fifa_goalkeeping_handling": "Handling",
    "fifa_goalkeeping_kicking": "Kicking",
    "fifa_goalkeeping_positioning": "Positioning",
    "fifa_goalkeeping_reflexes": "Reflexes",
}

_GENDER_COLORS = {"male": "#3b82f6", "female": "#ec4899"}
_QUERY_COLOR = "#2563eb"
_SUB_COLOR = "#f97316"

# Paper figure: few large radars per page (readable axis labels in print).
_PAPER_RADARS_PER_PAGE = 4
_PAPER_NCOLS = 2


def _safe_label(name: str) -> str:
    """Keep only Latin-script characters, digits, and common punctuation.

    Strips Georgian, Cyrillic, Arabic, CJK, emoji, and other non-Latin
    scripts that appear in some StatsBomb player names.
    """
    out = []
    for ch in name:
        cp = ord(ch)
        if (cp <= 0x024F            # ASCII + Latin-1 Supplement + Latin Extended-A/B
                or 0x1E00 <= cp <= 0x1EFF   # Latin Extended Additional
                or 0x2000 <= cp <= 0x206F   # General Punctuation (dashes, quotes)
                or 0x0300 <= cp <= 0x036F   # Combining Diacritical Marks (accents)
           ):
            out.append(ch)
    return "".join(out).strip()


def _draw_single_radar(
    ax,
    s: Dict,
    idx: int,
    *,
    label_size: float,
    radial_label_size: float,
    title_size: float,
    legend_size: float,
    line_width: float,
    marker_size: float,
    title_pad: float,
) -> None:
    """Plot one query vs substitute radar on polar axes."""
    outfield_keys = list(_STAT_LABELS.keys())
    outfield_names = list(_STAT_LABELS.values())
    gk_keys = list(_GK_STAT_LABELS.keys())
    gk_names = list(_GK_STAT_LABELS.values())

    use_gk = s.get("is_gk_pair", False)
    if use_gk:
        stat_keys = gk_keys
        stat_names = gk_names
        q_src, s_src = s.get("q_gk_stats", {}), s.get("s_gk_stats", {})
    else:
        stat_keys = outfield_keys
        stat_names = outfield_names
        q_src, s_src = s["q_main_stats"], s["s_main_stats"]

    angles = np.linspace(0, 2 * np.pi, len(stat_keys), endpoint=False).tolist()
    angles += angles[:1]

    q_vals = [q_src.get(k) or 0 for k in stat_keys] + [q_src.get(stat_keys[0]) or 0]
    s_vals = [s_src.get(k) or 0 for k in stat_keys] + [s_src.get(stat_keys[0]) or 0]

    ax.plot(
        angles,
        q_vals,
        "o-",
        color=_QUERY_COLOR,
        linewidth=line_width,
        markersize=marker_size,
        label=_safe_label(s["query"]),
    )
    ax.fill(angles, q_vals, color=_QUERY_COLOR, alpha=0.12)
    ax.plot(
        angles,
        s_vals,
        "s-",
        color=_SUB_COLOR,
        linewidth=line_width,
        markersize=marker_size,
        label=_safe_label(s["substitute"]),
    )
    ax.fill(angles, s_vals, color=_SUB_COLOR, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stat_names, size=label_size)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], size=radial_label_size, color="grey")
    ax.tick_params(pad=10 if label_size >= 10 else 8)

    gender_tag = s["gender"][0].upper()
    gk_tag = " GK" if use_gk else ""
    ax.set_title(
        f"#{idx + 1} ({gender_tag}{gk_tag})  cos={s['cosine_sim']:.3f}",
        size=title_size,
        fontweight="bold",
        pad=title_pad,
    )
    ax.legend(
        loc="upper right",
        fontsize=legend_size,
        framealpha=0.9,
        bbox_to_anchor=(1.42, 1.12),
    )


def _plot_radar_grid(summaries: List[Dict], plot_dir: Path) -> Path:
    """Radar chart grid: query vs substitute stat profiles per pair.

    For goalkeeper pairs the radar uses GK-specific stats (Diving, Handling,
    Kicking, Positioning, Reflexes) instead of the outfield main-6, which are
    near-zero for keepers in FIFA data.
    """
    n = len(summaries)
    if n == 0:
        return plot_dir / "radar_comparison.png"

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows),
        subplot_kw={"polar": True},
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, s in enumerate(summaries):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i, col_i]
        _draw_single_radar(
            ax,
            s,
            idx,
            label_size=7,
            radial_label_size=6,
            title_size=9,
            legend_size=5.5,
            line_width=1.8,
            marker_size=4,
            title_pad=14,
        )

    for idx in range(n, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        axes[row_i, col_i].set_visible(False)

    fig.suptitle(
        "Query vs Substitute — FIFA Stat Profiles\n"
        "(GK pairs use Diving / Handling / Kicking / Positioning / Reflexes)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    path = plot_dir / "radar_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_radar_paper(summaries: List[Dict], plot_dir: Path) -> List[Path]:
    """Paper-friendly radars: at most four large plots per page, readable labels.

    Writes ``radar_comparison_paper.png`` (and ``radar_comparison_paper_p2.png``,
    …) when there are more than ``_PAPER_RADARS_PER_PAGE`` pairs.
    """
    n = len(summaries)
    if n == 0:
        return []

    paths: List[Path] = []
    ncols = _PAPER_NCOLS
    per_page = _PAPER_RADARS_PER_PAGE

    for page_start in range(0, n, per_page):
        chunk = summaries[page_start : page_start + per_page]
        k = len(chunk)
        # Single radar: one large panel; else 2 columns × up to 2 rows.
        use_ncols = 1 if k == 1 else ncols
        this_rows = (k + use_ncols - 1) // use_ncols
        fig_w = 7.5 * use_ncols
        fig_h = max(7.0 * this_rows, 7.0)

        fig, axes = plt.subplots(
            this_rows,
            use_ncols,
            figsize=(fig_w, fig_h),
            subplot_kw={"polar": True},
        )
        if this_rows == 1 and use_ncols == 1:
            axes = np.array([[axes]])
        elif this_rows == 1:
            axes = axes[np.newaxis, :]
        elif use_ncols == 1:
            axes = axes[:, np.newaxis]

        for j, s in enumerate(chunk):
            row_i, col_i = divmod(j, use_ncols)
            ax = axes[row_i, col_i]
            global_idx = page_start + j
            _draw_single_radar(
                ax,
                s,
                global_idx,
                label_size=12,
                radial_label_size=10,
                title_size=11,
                legend_size=10,
                line_width=2.4,
                marker_size=6,
                title_pad=18,
            )

        for j in range(k, this_rows * use_ncols):
            row_i, col_i = divmod(j, use_ncols)
            axes[row_i, col_i].set_visible(False)

        page_num = page_start // per_page + 1
        suffix = "" if page_num == 1 else f"_p{page_num}"
        fig.suptitle(
            "Query vs Substitute — FIFA profiles (paper view)\n"
            "(GK pairs: Diving / Handling / Kicking / Positioning / Reflexes)",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = plot_dir / f"radar_comparison_paper{suffix}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    return paths


def _plot_similarity_vs_diff(summaries: List[Dict], plot_dir: Path) -> Path:
    """Scatter: does higher GNN cosine similarity predict lower FIFA stat diff?"""
    valid = [s for s in summaries if s["mean_main_diff"] is not None]
    if len(valid) < 3:
        return plot_dir / "similarity_vs_stat_diff.png"

    cos_sims = np.array([s["cosine_sim"] for s in valid])
    stat_diffs = np.array([s["mean_main_diff"] for s in valid])
    genders = [s["gender"] for s in valid]

    fig, ax = plt.subplots(figsize=(7, 5))

    for gender in ("male", "female"):
        mask = [g == gender for g in genders]
        if not any(mask):
            continue
        c = np.array(cos_sims)[mask]
        d = np.array(stat_diffs)[mask]
        ax.scatter(c, d, s=70, alpha=0.75, color=_GENDER_COLORS[gender],
                   edgecolors="white", linewidth=0.6, label=gender.title(),
                   zorder=3)

    rho, p_val = spearmanr(cos_sims, stat_diffs)

    if len(cos_sims) >= 2:
        z = np.polyfit(cos_sims, stat_diffs, 1)
        x_line = np.linspace(cos_sims.min() - 0.01, cos_sims.max() + 0.01, 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="#94a3b8",
                linewidth=1.5, zorder=2)

    ax.set_xlabel("GNN Cosine Similarity", fontsize=11)
    ax.set_ylabel("Mean Main-6 FIFA Stat Difference", fontsize=11)
    ax.set_title(
        "Embedding Similarity vs FIFA Stat Distance\n"
        f"Spearman ρ = {rho:+.3f}  (p = {p_val:.3f})",
        fontsize=12, fontweight="bold",
    )

    rho_color = "#16a34a" if rho < -0.3 else "#eab308" if rho < 0 else "#dc2626"
    if rho < 0:
        verdict = "Negative trend: agreement with FIFA"
    else:
        verdict = "No negative trend: no agreement with FIFA"
    ax.annotate(
        verdict, xy=(0.5, 0.02), xycoords="axes fraction",
        ha="center", fontsize=9, fontstyle="italic", color=rho_color,
    )

    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "similarity_vs_stat_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stat_breakdown(summaries: List[Dict], plot_dir: Path) -> Path:
    """Horizontal bar chart: mean absolute difference per main stat."""
    valid = [s for s in summaries if s["mean_main_diff"] is not None]
    if not valid:
        return plot_dir / "stat_difference_breakdown.png"

    stat_keys = list(_STAT_LABELS.keys())
    stat_names = list(_STAT_LABELS.values())

    per_stat_diffs: Dict[str, List[float]] = {k: [] for k in stat_keys}
    for s in valid:
        for k in stat_keys:
            qv = s["q_main_stats"].get(k)
            sv = s["s_main_stats"].get(k)
            if qv is not None and sv is not None:
                per_stat_diffs[k].append(abs(qv - sv))

    means = [np.mean(per_stat_diffs[k]) if per_stat_diffs[k] else 0
             for k in stat_keys]
    stds = [np.std(per_stat_diffs[k]) if len(per_stat_diffs[k]) > 1 else 0
            for k in stat_keys]

    sorted_idx = np.argsort(means)[::-1]
    sorted_names = [stat_names[i] for i in sorted_idx]
    sorted_means = [means[i] for i in sorted_idx]
    sorted_stds = [stds[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#ef4444" if m > 15 else "#eab308" if m > 8 else "#22c55e"
              for m in sorted_means]
    bars = ax.barh(
        range(len(sorted_names)), sorted_means, xerr=sorted_stds,
        color=colors, edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.2, "capsize": 3, "color": "#64748b"},
        height=0.6,
    )

    for bar, val in zip(bars, sorted_means):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9, color="#475569")

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Difference (FIFA points)", fontsize=10)
    ax.set_title(
        f"Per-Attribute Accuracy ({len(valid)} pairs)\n"
        "Green ≤ 8 | Yellow ≤ 15 | Red > 15",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, max(sorted_means) * 1.25 + 1)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = plot_dir / "stat_difference_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_evaluation_summary(summaries: List[Dict], plot_dir: Path) -> Path:
    """Pair-level quality overview: dumbbell chart + stat-diff heatmap."""
    valid = [s for s in summaries
             if s.get("q_overall") is not None and s.get("s_overall") is not None]
    if not valid:
        return plot_dir / "evaluation_summary.png"

    n = len(valid)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, max(5, 0.5 * n + 2)),
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    # ── Panel 1: dumbbell chart (query vs substitute overall rating) ──
    labels = []
    for i, s in enumerate(valid):
        gender_tag = s["gender"][0].upper()
        gk_tag = " GK" if s.get("is_gk_pair") else ""
        pos_icon = "\u2713" if s["pos_group_match"] else "\u2717"
        labels.append(
            f"#{i+1} ({gender_tag}{gk_tag}) {pos_icon}  cos={s['cosine_sim']:.3f}"
        )

    y_pos = np.arange(n)
    q_ovrs = [s["q_overall"] or 0 for s in valid]
    s_ovrs = [s["s_overall"] or 0 for s in valid]

    for i in range(n):
        color = "#22c55e" if valid[i]["pos_group_match"] else "#ef4444"
        ax1.plot([q_ovrs[i], s_ovrs[i]], [i, i], "-", color=color,
                 linewidth=2, alpha=0.6, zorder=1)
    ax1.scatter(q_ovrs, y_pos, s=60, color=_QUERY_COLOR, zorder=3,
                edgecolors="white", linewidth=0.7, label="Query")
    ax1.scatter(s_ovrs, y_pos, s=60, color=_SUB_COLOR, zorder=3,
                edgecolors="white", linewidth=0.7, marker="s",
                label="Substitute")

    for i in range(n):
        diff = abs(q_ovrs[i] - s_ovrs[i])
        mid = (q_ovrs[i] + s_ovrs[i]) / 2
        ax1.text(mid, i + 0.28, f"\u0394{diff:.0f}", ha="center",
                 fontsize=7, color="#64748b")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("FIFA Overall Rating", fontsize=10)
    ax1.set_title("Query vs Substitute Ratings\n"
                   "(\u2713 = position match, \u2717 = mismatch,"
                   " green = match, red = mismatch)",
                   fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.8)
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: heatmap of per-pair stat differences ──
    stat_keys = list(_STAT_LABELS.keys())
    stat_names = list(_STAT_LABELS.values())
    gk_keys = list(_GK_STAT_LABELS.keys())
    gk_names = list(_GK_STAT_LABELS.values())

    has_gk = any(s.get("is_gk_pair") for s in valid)
    if has_gk:
        col_names = stat_names + [f"GK:{n}" for n in gk_names]
        total_cols = len(stat_keys) + len(gk_keys)
    else:
        col_names = stat_names
        total_cols = len(stat_keys)

    matrix = np.full((n, total_cols), np.nan)
    row_labels = []
    for i, s in enumerate(valid):
        q_name = _safe_label(s["query"])
        s_name = _safe_label(s["substitute"])
        row_labels.append(f"#{i+1} {q_name} vs {s_name}")

        is_gk = s.get("is_gk_pair", False)
        if not is_gk:
            for j, k in enumerate(stat_keys):
                qv = s["q_main_stats"].get(k)
                sv = s["s_main_stats"].get(k)
                if qv is not None and sv is not None:
                    matrix[i, j] = abs(qv - sv)
        if has_gk and is_gk:
            for j, k in enumerate(gk_keys):
                qv = s.get("q_gk_stats", {}).get(k)
                sv = s.get("s_gk_stats", {}).get(k)
                if qv is not None and sv is not None:
                    matrix[i, len(stat_keys) + j] = abs(qv - sv)

    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 30
    im = ax2.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                    vmin=0, vmax=max(vmax, 1), interpolation="nearest")

    for i in range(n):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                ax2.text(j, i, "–", ha="center", va="center",
                         fontsize=7, color="#94a3b8")
            else:
                text_color = "white" if val > vmax * 0.65 else "#1e293b"
                ax2.text(j, i, f"{val:.0f}", ha="center", va="center",
                         fontsize=7, fontweight="bold", color=text_color)

    ax2.set_xticks(range(len(col_names)))
    ax2.set_xticklabels(col_names, fontsize=8, rotation=45, ha="right")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(row_labels, fontsize=6.5)
    ax2.set_title("Per-Pair Stat Difference (lower = better)\n"
                   "GK rows show '–' for outfield stats",
                   fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
    cbar.set_label("Absolute Difference", fontsize=9)

    fig.suptitle(
        "FIFA Comparison — Pair Quality Overview",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = plot_dir / "evaluation_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


_DETAIL_CATEGORIES = {
    "Attacking": [
        "fifa_attacking_crossing", "fifa_attacking_finishing",
        "fifa_attacking_heading_accuracy", "fifa_attacking_short_passing",
        "fifa_attacking_volleys",
    ],
    "Skill": [
        "fifa_skill_dribbling", "fifa_skill_curve", "fifa_skill_fk_accuracy",
        "fifa_skill_long_passing", "fifa_skill_ball_control",
    ],
    "Movement": [
        "fifa_movement_acceleration", "fifa_movement_sprint_speed",
        "fifa_movement_agility", "fifa_movement_reactions",
        "fifa_movement_balance",
    ],
    "Power": [
        "fifa_power_shot_power", "fifa_power_jumping", "fifa_power_stamina",
        "fifa_power_strength", "fifa_power_long_shots",
    ],
    "Mentality": [
        "fifa_mentality_aggression", "fifa_mentality_interceptions",
        "fifa_mentality_positioning", "fifa_mentality_vision",
        "fifa_mentality_penalties", "fifa_mentality_composure",
    ],
    "Defending": [
        "fifa_defending_marking_awareness", "fifa_defending_standing_tackle",
        "fifa_defending_sliding_tackle",
    ],
}

_CAT_COLORS = {
    "Attacking": "#ef4444", "Skill": "#f59e0b", "Movement": "#22c55e",
    "Power": "#3b82f6", "Mentality": "#8b5cf6", "Defending": "#64748b",
}


def _plot_category_breakdown(summaries: List[Dict], plot_dir: Path) -> Path:
    """Bar chart: mean |diff| per sub-attribute category across all pairs."""
    valid = [s for s in summaries
             if s.get("q_detail_stats") and not s.get("is_gk_pair")]
    if not valid:
        return plot_dir / "category_breakdown.png"

    cat_diffs: Dict[str, List[float]] = {c: [] for c in _DETAIL_CATEGORIES}
    for s in valid:
        q_d = s["q_detail_stats"]
        s_d = s["s_detail_stats"]
        for cat, keys in _DETAIL_CATEGORIES.items():
            diffs = []
            for k in keys:
                qv, sv = q_d.get(k), s_d.get(k)
                if qv is not None and sv is not None:
                    diffs.append(abs(qv - sv))
            if diffs:
                cat_diffs[cat].append(float(np.mean(diffs)))

    cats = list(_DETAIL_CATEGORIES.keys())
    means = [np.mean(cat_diffs[c]) if cat_diffs[c] else 0 for c in cats]
    stds = [np.std(cat_diffs[c]) if len(cat_diffs[c]) > 1 else 0 for c in cats]

    sorted_idx = np.argsort(means)[::-1]
    sorted_cats = [cats[i] for i in sorted_idx]
    sorted_means = [means[i] for i in sorted_idx]
    sorted_stds = [stds[i] for i in sorted_idx]
    sorted_colors = [_CAT_COLORS.get(c, "#94a3b8") for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sorted_cats))
    bars = ax.bar(x, sorted_means, yerr=sorted_stds, color=sorted_colors,
                  edgecolor="white", linewidth=0.5, width=0.6,
                  error_kw={"elinewidth": 1.2, "capsize": 4, "color": "#475569"})

    for bar, v in zip(bars, sorted_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_cats, fontsize=10)
    ax.set_ylabel("Mean Absolute Difference", fontsize=10)
    ax.set_title(f"Sub-Attribute Category Accuracy ({len(valid)} outfield pairs)\n"
                 "Lower = GNN substitute has closer FIFA sub-stats",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(sorted_means) * 1.3 + 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = plot_dir / "category_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_detail_heatmap(summaries: List[Dict], plot_dir: Path) -> Path:
    """Heatmap: per-pair |diff| for every sub-attribute, grouped by category."""
    valid = [s for s in summaries
             if s.get("q_detail_stats") and not s.get("is_gk_pair")]
    if not valid:
        return plot_dir / "detail_heatmap.png"

    col_keys: List[str] = []
    col_labels: List[str] = []
    cat_boundaries: List[int] = []
    cat_names: List[str] = []
    for cat, keys in _DETAIL_CATEGORIES.items():
        cat_boundaries.append(len(col_keys))
        cat_names.append(cat)
        for k in keys:
            col_keys.append(k)
            short = k.replace("fifa_", "").split("_", 1)[-1].replace("_", " ").title()
            col_labels.append(short)

    n = len(valid)
    n_cols = len(col_keys)
    matrix = np.full((n, n_cols), np.nan)
    row_labels = []

    for i, s in enumerate(valid):
        q_name = _safe_label(s["query"])
        s_name = _safe_label(s["substitute"])
        row_labels.append(f"#{i+1} {q_name} vs {s_name}")
        q_d, s_d = s["q_detail_stats"], s["s_detail_stats"]
        for j, k in enumerate(col_keys):
            qv, sv = q_d.get(k), s_d.get(k)
            if qv is not None and sv is not None:
                matrix[i, j] = abs(qv - sv)

    fig_w = max(16, 0.45 * n_cols + 4)
    fig_h = max(5.5, 0.55 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 30
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=max(vmax, 1), interpolation="nearest")

    for i in range(n):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=6, color="#94a3b8")
            else:
                text_color = "white" if val > vmax * 0.65 else "#1e293b"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, fontweight="bold", color=text_color)

    for bnd in cat_boundaries[1:]:
        ax.axvline(bnd - 0.5, color="white", linewidth=1.5)

    for ci, (bnd, cat_name) in enumerate(zip(cat_boundaries, cat_names)):
        next_bnd = cat_boundaries[ci + 1] if ci + 1 < len(cat_boundaries) else n_cols
        mid = (bnd + next_bnd - 1) / 2
        ax.text(mid, -1.2, cat_name, ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=_CAT_COLORS.get(cat_name, "#475569"))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=60, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("Sub-Attribute Differences per Pair (outfield only)\n"
                 "Green = small gap, Red = large gap",
                 fontsize=12, fontweight="bold", pad=45)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Absolute Difference", fontsize=9)

    fig.tight_layout()
    path = plot_dir / "detail_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_visualizations(
    summaries: List[Dict], plot_dir: Path,
) -> List[Path]:
    """Generate all FIFA comparison visualizations. Returns list of saved paths."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not summaries:
        return []

    paths: List[Path] = []
    print("[5/5] Generating visualizations …")

    print("  → Radar chart grid …")
    paths.append(_plot_radar_grid(summaries, plot_dir))

    print("  → Radar chart (paper layout, large labels) …")
    paths.extend(_plot_radar_paper(summaries, plot_dir))

    print("  → Similarity vs stat difference scatter …")
    paths.append(_plot_similarity_vs_diff(summaries, plot_dir))

    print("  → Per-stat difference breakdown …")
    paths.append(_plot_stat_breakdown(summaries, plot_dir))

    print("  → Evaluation summary dashboard …")
    paths.append(_plot_evaluation_summary(summaries, plot_dir))

    print("  → Sub-attribute category breakdown …")
    paths.append(_plot_category_breakdown(summaries, plot_dir))

    print("  → Sub-attribute detail heatmap …")
    paths.append(_plot_detail_heatmap(summaries, plot_dir))

    for p in paths:
        if p.exists():
            print(f"    Saved: {p}")

    return paths


# ── Public API ─────────────────────────────────────────────────────────────

def run_comparison(
    embeddings_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    statsbomb_base: Optional[Path] = None,
    male_fifa_csv: Optional[Path] = None,
    female_fifa_csv: Optional[Path] = None,
    output_file: Optional[Path] = None,
    seed: Optional[int] = SEED,
) -> None:
    """Run the full FIFA stat comparison test.

    All parameters default to the hardcoded project paths when ``None``.
    """
    emb_dir = embeddings_dir or _DEFAULT_EMBEDDINGS_DIR
    p_dir = processed_dir or _DEFAULT_PROCESSED_DIR
    sb_base = statsbomb_base or _DEFAULT_STATSBOMB_BASE
    m_csv = male_fifa_csv or _DEFAULT_MALE_FIFA_CSV
    f_csv = female_fifa_csv or _DEFAULT_FEMALE_FIFA_CSV
    out_file = output_file or _DEFAULT_OUTPUT_FILE

    print("=" * 80)
    print("  FIFA STAT COMPARISON TEST — GNN Player Similarity Validation")
    print("=" * 80)

    print("\n[1/5] Loading embeddings …")
    Z = np.load(emb_dir / "player_embeddings.npy")
    player_info = pd.read_parquet(emb_dir / "player_info.parquet")
    pids = player_info["player_id"].values
    print(f"  Embeddings: {Z.shape[0]} players, {Z.shape[1]}-dim")

    print("[2/5] Loading FIFA data …")
    fifa_m = load_fifa(m_csv)
    fifa_f = load_fifa(f_csv)
    fifa_all = pd.concat([fifa_m, fifa_f], ignore_index=True)
    fifa_lookup: Dict[int, pd.Series] = {
        int(row["sb_player_id"]): row
        for _, row in fifa_all.iterrows()
    }
    fifa_pids = set(fifa_lookup.keys())
    print(f"  FIFA unique: {len(fifa_m)} male, {len(fifa_f)} female")

    print("[3/5] Building gender map …")
    gender_map = build_gender_map(processed_dir=p_dir, statsbomb_base=sb_base)

    emb_pids = set(int(p) for p in pids)
    male_overlap = np.array(sorted(
        emb_pids & fifa_pids & {p for p, g in gender_map.items() if g == "male"}
    ))
    female_overlap = np.array(sorted(
        emb_pids & fifa_pids & {p for p, g in gender_map.items() if g == "female"}
    ))
    print(f"  Players with embedding + FIFA: {len(male_overlap)} male, {len(female_overlap)} female")

    if len(male_overlap) == 0 and len(female_overlap) == 0:
        print("  No overlapping players — skipping FIFA comparison.")
        return

    print("[4/5] Sampling players & finding substitutes …")
    rng = np.random.default_rng(seed)
    n_male = min(10, len(male_overlap))
    n_female = min(10, len(female_overlap))
    male_sample = _sample_with_famous(
        male_overlap, player_info, n_male, rng, FAMOUS_MALE_IDS, fifa_pids,
    ) if n_male else []
    female_sample = _sample_with_famous(
        female_overlap, player_info, n_female, rng, FAMOUS_FEMALE_IDS, fifa_pids,
    ) if n_female else []

    sim_matrix = cosine_similarity(Z)

    all_lines: List[str] = []
    summaries: List[Dict] = []

    all_lines.append("=" * 80)
    all_lines.append("  FIFA STAT COMPARISON TEST — GNN Player Similarity Validation")
    all_lines.append("=" * 80)
    all_lines.append(f"  Seed: {seed if seed is not None else 'random'}")
    all_lines.append(f"  Embeddings: {Z.shape[0]} players ({Z.shape[1]}-dim)")
    all_lines.append(f"  Overlap: {len(male_overlap)} male, {len(female_overlap)} female")
    all_lines.append("")

    pair_num = 0
    for label, sample in [("male", male_sample), ("female", female_sample)]:
        if not sample:
            continue
        all_lines.append("")
        all_lines.append(f"{'#'*80}")
        all_lines.append(f"  {label.upper()} PLAYERS ({len(sample)} sampled)")
        all_lines.append(f"{'#'*80}")

        for query_pid in tqdm(sample, desc=f"{label.title()} pairs"):
            pair_num += 1
            q_idx = int(np.where(pids == query_pid)[0][0])
            result = find_top1_same_gender_with_fifa(
                query_pid, sim_matrix[q_idx], pids, q_idx,
                gender_map, fifa_pids,
            )
            if result is None:
                all_lines.append(f"\n  PAIR {pair_num}: {query_pid} — no same-gender substitute with FIFA data\n")
                continue

            sub_pid, cos_sim = result
            q_row = fifa_lookup[int(query_pid)]
            s_row = fifa_lookup[sub_pid]

            pair_lines, summary = format_pair(q_row, s_row, cos_sim, pair_num, label)
            all_lines.extend(pair_lines)
            summaries.append(summary)

    all_lines.append("")
    all_lines.append("=" * 80)
    all_lines.append("  SUMMARY TABLE")
    all_lines.append("=" * 80)

    hdr = (f"  {'#':>3s}  {'Gender':6s}  {'Query':25s}  {'Substitute':25s}  "
           f"{'CosSim':>7s}  {'PosOK':>5s}  {'OvrDiff':>7s}  {'MainDiff':>8s}")
    all_lines.append(hdr)
    all_lines.append(f"  {'-'*3}  {'-'*6}  {'-'*25}  {'-'*25}  "
                     f"{'-'*7}  {'-'*5}  {'-'*7}  {'-'*8}")

    for i, s in enumerate(summaries, 1):
        q = s["query"][:25]
        sub = s["substitute"][:25]
        pos_ok = "Y" if s["pos_group_match"] else "N"
        ovr_d = _fmt(s["ovr_diff"])
        main_d = f"{s['mean_main_diff']:.1f}" if s["mean_main_diff"] is not None else "N/A"
        all_lines.append(
            f"  {i:>3d}  {s['gender']:6s}  {q:25s}  {sub:25s}  "
            f"{s['cosine_sim']:>7.4f}  {pos_ok:>5s}  {ovr_d:>7s}  {main_d:>8s}"
        )

    valid = [s for s in summaries if s["mean_main_diff"] is not None]
    if valid:
        avg_cos = np.mean([s["cosine_sim"] for s in valid])
        avg_ovr = np.mean([s["ovr_diff"] for s in valid if s["ovr_diff"] is not None])
        avg_main = np.mean([s["mean_main_diff"] for s in valid])
        avg_detail = np.mean([s["mean_detail_diff"] for s in valid
                              if s["mean_detail_diff"] is not None])
        pos_pct = 100 * sum(1 for s in valid if s["pos_group_match"]) / len(valid)

        all_lines.append("")
        all_lines.append(f"  --- Aggregates ({len(valid)} pairs) ---")
        all_lines.append(f"  Avg cosine similarity:      {avg_cos:.4f}")
        all_lines.append(f"  Avg overall rating diff:    {avg_ovr:.1f}")
        all_lines.append(f"  Avg main-6 stat diff:       {avg_main:.1f}")
        all_lines.append(f"  Avg detail stat diff:        {avg_detail:.1f}")
        all_lines.append(f"  Position group match:        {pos_pct:.0f}%")

    all_lines.append("")

    report = "\n".join(all_lines)
    print(report)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {out_file}")

    plot_dir = out_file.parent
    if summaries:
        saved_plots = generate_visualizations(summaries, plot_dir)
        print(f"\n  {len(saved_plots)} visualizations saved to {plot_dir}")


def main() -> None:
    """Standalone entry point using default paths."""
    run_comparison()


if __name__ == "__main__":
    main()
