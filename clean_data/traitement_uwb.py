"""
traitement_uwb.py
-----------------
Pipeline de nettoyage et de traitement des donnees CSV issues du systeme
de localisation UWB (DWM1001) pour le water-polo.

Ce script est concu pour s'adapter a tout fichier CSV respectant la
structure standard : time, nodeID, positionX, positionY, positionZ, quality.

Les parametres du terrain (dimensions, marges) et du traitement (seuils,
fenetres) sont configurables en ligne de commande ou via un fichier de
configuration JSON.

Orientation du terrain (cf. Figure 2 du cahier des charges) :
  - Axe X (0-25 m) : longueur du terrain, buts aux extremites X
  - Axe Y (0-26 m) : largeur du terrain + marges (terrain joue : 3-23 m)
  - But gauche : X ~ 0,  centre Y ~ 13 m
  - But droit  : X ~ 25, centre Y ~ 13 m

Zones officielles (depuis chaque but sur X) :
  - Zone des 2 m       : 0-2 m et 23-25 m
  - Zone attaque/def   : 2-6 m et 19-23 m
  - Zone de transition : 6-19 m (13 m au centre)

Usage basique :
    python traitement_uwb.py --input t1.csv --output t1_clean.xlsx

Usage avec fichier joueurs :
    python traitement_uwb.py --input t1.csv --output t1_clean.xlsx \\
        --players players.json

Usage pour traiter plusieurs fichiers :
    python traitement_uwb.py --input t1.csv t2.csv t3.csv --output-dir ./clean/
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Optional


# =========================================================================
# PERIODES DES MATCHS (date -> configuration)
# =========================================================================

MATCH_PERIODS = {
    "2026-01-13": {"n_periods": 3, "duration_s": 1200},  # 3 x 20 min
    "2026-01-20": {"n_periods": 4, "duration_s": 720},   # 4 x 12 min
    "2026-02-03": {"n_periods": 4, "duration_s": 720},   # 4 x 12 min
}


# =========================================================================
# CONFIGURATION
# =========================================================================

@dataclass
class Config:
    """Configuration du pipeline, modifiable par CLI ou fichier JSON."""

    # --- Limites du terrain (metres) ---
    pool_x_min: float = -0.5
    pool_x_max: float = 25.5
    pool_y_min: float = -0.5
    pool_y_max: float = 27.0
    pool_z_min: float = -3.0
    pool_z_max: float = 3.0

    # --- Seuils de traitement ---
    min_quality: int = 30
    max_speed: float = 1.3       # m/s
    teleport_passes: int = 3

    # --- Lissage ---
    median_window: int = 5

    # --- Reechantillonnage ---
    resample_freq_ms: int = 100  # 10 Hz
    max_interp_gap: float = 2.0  # secondes

    # --- Buts (axe X) et centre Y ---
    goal_left_x: float = 0.0    # but cote X minimum
    goal_right_x: float = 25.0  # but cote X maximum
    goal_center_y: float = 13.25  # centre Y du bassin (largeur 26m / 2)

    # --- Lignes officielles water-polo (depuis chaque but) ---
    line_2m: float = 2.0   # ligne des 2 m
    line_6m: float = 6.0   # ligne des 6 m

    # --- Etapes a activer ---
    do_quality_filter: bool = True
    do_bounds_filter: bool = True
    do_teleport_filter: bool = True
    do_smooth: bool = True
    do_resample: bool = True
    do_derived: bool = True

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)


# =========================================================================
# RAPPORT DE TRAITEMENT
# =========================================================================

@dataclass
class StepReport:
    name: str
    rows_before: int
    rows_after: int
    rows_removed: int = 0
    detail: str = ""

    def __post_init__(self):
        self.rows_removed = self.rows_before - self.rows_after


@dataclass
class PipelineReport:
    input_file: str
    output_file: str
    config: dict = field(default_factory=dict)
    steps: List[dict] = field(default_factory=list)
    nodes_summary: dict = field(default_factory=dict)
    total_rows_in: int = 0
    total_rows_out: int = 0

    def add_step(self, step: StepReport):
        self.steps.append(asdict(step))

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


# =========================================================================
# CHARGEMENT DU MAPPING JOUEURS
# =========================================================================

def load_players(filepath: str) -> dict:
    """
    Charge le fichier JSON de mapping nodeID -> joueur/equipe.

    Format attendu :
    {
      "2026-02-03": {
        "1bb3": {"bonnet": 10, "equipe": "INSEP"},
        "4103": {"bonnet": 13, "equipe": "INSEP"},
        ...
      }
    }
    Retourne un dict vide si le fichier n'existe pas.
    """
    if not filepath or not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_players(df: pd.DataFrame, players: dict) -> pd.DataFrame:
    """Ajoute les colonnes 'bonnet' et 'equipe' depuis le mapping joueurs."""
    if not players:
        return df

    date_str = str(df["date"].iloc[0])
    day_map = players.get(date_str, {})
    if not day_map:
        log(f"  Aucun mapping joueur trouve pour la date {date_str}")
        return df

    df["bonnet"] = df["nodeID"].map(lambda nid: day_map.get(nid, {}).get("bonnet"))
    df["equipe"] = df["nodeID"].map(lambda nid: day_map.get(nid, {}).get("equipe"))
    mapped = df["bonnet"].notna().sum()
    log(f"  Mapping joueurs applique : {mapped}/{len(df)} lignes enrichies")
    return df


# =========================================================================
# FONCTIONS DU PIPELINE
# =========================================================================

def log(msg):
    print(f"  {msg}")


def load_csv(filepath):
    if not os.path.exists(filepath):
        print(f"ERREUR : fichier introuvable : {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath)

    required = {"time", "nodeID", "positionX", "positionY", "positionZ", "quality"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERREUR : colonnes manquantes dans {filepath} : {missing}")
        print(f"  Colonnes trouvees : {list(df.columns)}")
        print(f"  Colonnes requises : {sorted(required)}")
        sys.exit(1)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["nodeID", "time"]).reset_index(drop=True)

    log(f"Fichier charge : {os.path.basename(filepath)}")
    log(f"  {len(df)} lignes, {df['nodeID'].nunique()} tags")
    return df


def step_drop_nan(df):
    """Etape 1 : suppression des lignes NaN."""
    before = len(df)
    df = df.dropna(subset=["positionX", "positionY", "positionZ"]).copy()
    report = StepReport("suppression_nan", before, len(df))
    log(f"[1/7] NaN supprimes : {report.rows_removed} ({100 * report.rows_removed / before:.1f}%)")
    return df, report


def step_filter_quality(df, cfg: Config):
    """Etape 2 : filtre par score de qualite."""
    before = len(df)
    df = df[df["quality"] >= cfg.min_quality].copy()
    report = StepReport("filtre_qualite", before, len(df), detail=f"seuil={cfg.min_quality}")
    log(f"[2/7] Qualite < {cfg.min_quality} : {report.rows_removed} supprimes "
        f"({100 * report.rows_removed / before:.1f}%)")
    return df, report


def step_filter_bounds(df, cfg: Config):
    """Etape 3 : filtre geographique."""
    before = len(df)
    mask = (
        (df["positionX"] >= cfg.pool_x_min) & (df["positionX"] <= cfg.pool_x_max) &
        (df["positionY"] >= cfg.pool_y_min) & (df["positionY"] <= cfg.pool_y_max) &
        (df["positionZ"] >= cfg.pool_z_min) & (df["positionZ"] <= cfg.pool_z_max)
    )
    df = df[mask].copy()
    report = StepReport("filtre_geographique", before, len(df),
                        detail=f"X=[{cfg.pool_x_min},{cfg.pool_x_max}] "
                               f"Y=[{cfg.pool_y_min},{cfg.pool_y_max}] "
                               f"Z=[{cfg.pool_z_min},{cfg.pool_z_max}]")
    log(f"[3/7] Hors piscine : {report.rows_removed} supprimes "
        f"({100 * report.rows_removed / max(before, 1):.2f}%)")
    return df, report


def step_remove_teleportations(df, cfg: Config):
    """Etape 4 : suppression iterative des teleportations."""
    before_total = len(df)
    total_removed = 0

    for p in range(cfg.teleport_passes):
        before_pass = len(df)
        parts = []

        for _, group in df.groupby("nodeID"):
            group = group.sort_values("time").copy()
            if len(group) < 2:
                parts.append(group)
                continue

            dx = group["positionX"].diff()
            dy = group["positionY"].diff()
            dt = group["time"].diff().dt.total_seconds()

            dist = np.sqrt(dx ** 2 + dy ** 2)
            speed = dist / dt

            keep = (speed <= cfg.max_speed) | speed.isna()
            parts.append(group[keep])

        df = pd.concat(parts).sort_values(["nodeID", "time"]).reset_index(drop=True)
        removed = before_pass - len(df)
        total_removed += removed

        if removed == 0:
            break

    report = StepReport("suppression_teleportations", before_total, len(df),
                        detail=f"seuil={cfg.max_speed}m/s, {p + 1} passes")
    log(f"[4/7] Teleportations ({p + 1} passes) : {total_removed} supprimes "
        f"({100 * total_removed / max(before_total, 1):.1f}%)")
    return df, report


def step_smooth_median(df, cfg: Config):
    """Etape 5 : lissage par filtre median glissant."""
    before = len(df)
    parts = []

    for node_id, group in df.groupby("nodeID"):
        group = group.sort_values("time").copy()
        if len(group) < cfg.median_window:
            parts.append(group)
            continue

        for col in ["positionX", "positionY", "positionZ"]:
            group[col] = group[col].rolling(
                window=cfg.median_window, center=True, min_periods=1
            ).median()

        parts.append(group)

    df = pd.concat(parts).sort_values(["nodeID", "time"]).reset_index(drop=True)
    report = StepReport("lissage_median", before, len(df), detail=f"fenetre={cfg.median_window}")
    log(f"[5/7] Lissage median (fenetre={cfg.median_window}) applique")
    return df, report


def step_resample_interpolate(df, cfg: Config):
    """Etape 6 : reechantillonnage a frequence fixe + interpolation."""
    before = len(df)
    freq = f"{cfg.resample_freq_ms}ms"
    hz = 1000 / cfg.resample_freq_ms
    parts = []

    for node_id, group in df.groupby("nodeID"):
        group = group.sort_values("time").copy()
        original_times = group["time"].values

        group = group.set_index("time")
        resampled = group[["positionX", "positionY", "positionZ"]].resample(freq).mean()

        resampled = resampled.interpolate(method="time", limit_direction="forward")

        for i in range(1, len(original_times)):
            gap = (original_times[i] - original_times[i - 1]) / np.timedelta64(1, "s")
            if gap > cfg.max_interp_gap:
                mask = (resampled.index > original_times[i - 1]) & \
                       (resampled.index < original_times[i])
                resampled.loc[mask] = np.nan

        resampled = resampled.dropna()
        resampled["nodeID"] = node_id
        resampled = resampled.reset_index().rename(columns={"index": "time"})
        parts.append(resampled)

    df = pd.concat(parts).sort_values(["nodeID", "time"]).reset_index(drop=True)
    added = len(df) - before
    report = StepReport("reechantillonnage", before, len(df),
                        detail=f"freq={hz:.0f}Hz, interp_max={cfg.max_interp_gap}s, "
                               f"delta={'+' if added >= 0 else ''}{added} pts")
    log(f"[6/7] Reechantillonnage a {hz:.0f} Hz : {len(df)} lignes finales "
        f"(delta: {'+' if added >= 0 else ''}{added})")
    return df, report


def step_add_derived_columns(df, cfg: Config, session_name: str = ""):
    """Etape 7 : colonnes cinematiques, contextuelles et temporelles."""
    parts = []

    for _, group in df.groupby("nodeID"):
        group = group.sort_values("time").copy()

        dx = group["positionX"].diff()
        dy = group["positionY"].diff()
        dt = group["time"].diff().dt.total_seconds()

        group["distance_step"] = np.sqrt(dx ** 2 + dy ** 2).round(4)
        group["distance_cumul"] = group["distance_step"].fillna(0).cumsum().round(4)
        group["speed"] = (group["distance_step"] / dt).round(4)
        group["acceleration"] = (group["speed"].diff() / dt).round(4)
        group["heading"] = np.degrees(np.arctan2(dy, dx)).round(2)

        parts.append(group)

    df = pd.concat(parts).sort_values(["nodeID", "time"]).reset_index(drop=True)

    # --- Zones officielles water-polo (axe X) ---
    # But gauche : X ~ 0  |  But droit : X ~ 25
    # Zones depuis chaque but : 2m, 6m, puis transition au centre
    x2_left = cfg.goal_left_x + cfg.line_2m    # 0 + 2 = 2
    x6_left = cfg.goal_left_x + cfg.line_6m    # 0 + 6 = 6
    x6_right = cfg.goal_right_x - cfg.line_6m  # 25 - 6 = 19
    x2_right = cfg.goal_right_x - cfg.line_2m  # 25 - 2 = 23

    df["zone"] = pd.cut(
        df["positionX"],
        bins=[cfg.pool_x_min, x2_left, x6_left, x6_right, x2_right, cfg.pool_x_max],
        labels=["2m_gauche", "ad_gauche", "transition", "ad_droite", "2m_droite"],
        include_lowest=True,
    )

    # --- Distances aux buts (buts sur axe X, centres en Y) ---
    df["dist_but_gauche"] = np.sqrt(
        (df["positionX"] - cfg.goal_left_x) ** 2 +
        (df["positionY"] - cfg.goal_center_y) ** 2
    ).round(4)
    df["dist_but_droit"] = np.sqrt(
        (df["positionX"] - cfg.goal_right_x) ** 2 +
        (df["positionY"] - cfg.goal_center_y) ** 2
    ).round(4)

    # --- Colonnes temporelles ---
    df["date"] = df["time"].dt.date
    df["elapsed_s"] = (df["time"] - df["time"].min()).dt.total_seconds().round(1)
    df["session"] = session_name

    # --- Periode de jeu ---
    date_str = str(df["date"].iloc[0])
    period_cfg = MATCH_PERIODS.get(date_str)
    if period_cfg:
        dur = period_cfg["duration_s"]
        n = period_cfg["n_periods"]
        df["period"] = (df["elapsed_s"] // dur + 1).clip(upper=n).astype(int)
        log(f"  Periodes detectees : {n} x {dur // 60} min pour le {date_str}")
    else:
        df["period"] = None
        log(f"  Aucune configuration de periodes pour le {date_str} — colonne period = None")

    report = StepReport("colonnes_derivees", len(df), len(df),
                        detail="speed, distance, acceleration, heading, zone, buts, date, elapsed_s, period")
    log(f"[7/7] Colonnes derivees : distance_step, distance_cumul, speed, acceleration, "
        f"heading, zone, dist_but_gauche, dist_but_droit, date, elapsed_s, session, period")
    return df, report


# =========================================================================
# PIPELINE PRINCIPAL
# =========================================================================

def run_pipeline(input_path, output_path, cfg: Config, players: dict = None):
    """Execute le pipeline complet sur un fichier."""
    header = f"{'=' * 60}\n  TRAITEMENT : {os.path.basename(input_path)}\n{'=' * 60}"
    print(f"\n{header}\n")

    report = PipelineReport(
        input_file=os.path.basename(input_path),
        output_file=os.path.basename(output_path),
        config=asdict(cfg),
    )

    df = load_csv(input_path)
    report.total_rows_in = len(df)

    df, step_r = step_drop_nan(df)
    report.add_step(step_r)

    if cfg.do_quality_filter:
        df, step_r = step_filter_quality(df, cfg)
        report.add_step(step_r)

    if cfg.do_bounds_filter:
        df, step_r = step_filter_bounds(df, cfg)
        report.add_step(step_r)

    if cfg.do_teleport_filter:
        df, step_r = step_remove_teleportations(df, cfg)
        report.add_step(step_r)

    if cfg.do_smooth:
        df, step_r = step_smooth_median(df, cfg)
        report.add_step(step_r)

    if cfg.do_resample:
        df, step_r = step_resample_interpolate(df, cfg)
        report.add_step(step_r)

    if cfg.do_derived:
        session_name = os.path.splitext(os.path.basename(input_path))[0]
        df, step_r = step_add_derived_columns(df, cfg, session_name)
        report.add_step(step_r)

    # Mapping joueurs (bonnet + equipe)
    if players:
        df = apply_players(df, players)

    # Resume par tag
    report.total_rows_out = len(df)
    nodes_summary = {}
    print()
    log("--- RESUME PAR TAG ---")
    for node in sorted(df["nodeID"].unique()):
        sub = df[df["nodeID"] == node]
        duration = (sub["time"].max() - sub["time"].min()).total_seconds()
        nodes_summary[node] = {"points": len(sub), "duration_s": round(duration, 1)}
        log(f"  {node}: {len(sub)} points, {duration:.0f}s")

    report.nodes_summary = nodes_summary

    # Sauvegarde Excel (quality supprimee : synthetique apres reechantillonnage)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.drop(columns=["quality"], errors="ignore").to_excel(output_path, index=False, engine="openpyxl")
    log(f"\nFichier sauvegarde : {output_path}")

    report_path = output_path.replace(".xlsx", "_rapport.json")
    report.save(report_path)
    log(f"Rapport sauvegarde : {report_path}")

    return df, report


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de traitement des donnees UWB water-polo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python traitement_uwb.py --input t1.csv --output t1_clean.xlsx
  python traitement_uwb.py --input t1.csv --output t1_clean.xlsx --players players.json
  python traitement_uwb.py --input t1.csv t2.csv --output-dir ./clean/
  python traitement_uwb.py --input t1.csv --output t1_clean.xlsx --quality 40 --speed 3.0
  python traitement_uwb.py --input t1.csv --output t1_clean.xlsx --config config.json
  python traitement_uwb.py --generate-config config.json
        """
    )

    parser.add_argument("--input", nargs="+", help="Un ou plusieurs fichiers CSV a traiter")
    parser.add_argument("--output", type=str, default=None,
                        help="Fichier de sortie .xlsx (pour un seul fichier en entree)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Repertoire de sortie (pour plusieurs fichiers)")
    parser.add_argument("--config", type=str, default=None,
                        help="Fichier JSON de configuration")
    parser.add_argument("--players", type=str, default=None,
                        help="Fichier JSON de mapping nodeID -> joueur/equipe")
    parser.add_argument("--generate-config", type=str, default=None,
                        help="Genere un fichier de configuration par defaut et quitte")

    parser.add_argument("--quality", type=int, default=None, help="Score qualite minimum")
    parser.add_argument("--speed", type=float, default=None, help="Vitesse max (m/s)")
    parser.add_argument("--median-window", type=int, default=None, help="Fenetre du filtre median")
    parser.add_argument("--max-gap", type=float, default=None, help="Gap max pour interpolation (s)")
    parser.add_argument("--freq", type=int, default=None, help="Frequence de reechantillonnage (ms)")
    parser.add_argument("--no-smooth", action="store_true", help="Desactiver le lissage")
    parser.add_argument("--no-interp", action="store_true", help="Desactiver le reechantillonnage")
    parser.add_argument("--no-quality", action="store_true", help="Desactiver le filtre qualite")
    parser.add_argument("--no-bounds", action="store_true", help="Desactiver le filtre geographique")
    parser.add_argument("--no-teleport", action="store_true", help="Desactiver le filtre teleportation")
    parser.add_argument("--pool-x", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--pool-y", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--pool-z", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))

    args = parser.parse_args()

    if args.generate_config:
        cfg = Config()
        cfg.to_json(args.generate_config)
        print(f"Configuration par defaut sauvegardee dans : {args.generate_config}")
        return

    if not args.input:
        parser.error("--input est requis (sauf avec --generate-config)")

    cfg = Config()
    if args.config:
        cfg = Config.from_json(args.config)
        print(f"Configuration chargee depuis : {args.config}")

    if args.quality is not None:
        cfg.min_quality = args.quality
    if args.speed is not None:
        cfg.max_speed = args.speed
    if args.median_window is not None:
        cfg.median_window = args.median_window
    if args.max_gap is not None:
        cfg.max_interp_gap = args.max_gap
    if args.freq is not None:
        cfg.resample_freq_ms = args.freq
    if args.no_smooth:
        cfg.do_smooth = False
    if args.no_interp:
        cfg.do_resample = False
    if args.no_quality:
        cfg.do_quality_filter = False
    if args.no_bounds:
        cfg.do_bounds_filter = False
    if args.no_teleport:
        cfg.do_teleport_filter = False
    if args.pool_x:
        cfg.pool_x_min, cfg.pool_x_max = args.pool_x
    if args.pool_y:
        cfg.pool_y_min, cfg.pool_y_max = args.pool_y
    if args.pool_z:
        cfg.pool_z_min, cfg.pool_z_max = args.pool_z

    players = load_players(args.players) if args.players else {}

    if len(args.input) == 1 and args.output:
        run_pipeline(args.input[0], args.output, cfg, players)
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for filepath in args.input:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(args.output_dir, f"{basename}_clean.xlsx")
            run_pipeline(filepath, out_path, cfg, players)
    elif len(args.input) == 1:
        basename = os.path.splitext(args.input[0])[0]
        run_pipeline(args.input[0], f"{basename}_clean.xlsx", cfg, players)
    else:
        for filepath in args.input:
            basename = os.path.splitext(filepath)[0]
            run_pipeline(filepath, f"{basename}_clean.xlsx", cfg, players)


if __name__ == "__main__":
    main()
