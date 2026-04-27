"""
diagnostic_uwb.py
-----------------
Programme de diagnostic des fichiers CSV issus du systeme de localisation
UWB (DWM1001) pour le water-polo.

Analyse un ou plusieurs fichiers CSV et produit un rapport textuel complet
couvrant : structure, completude, qualite, coherence spatiale, coherence
temporelle, et detection d'anomalies.

Usage :
    python diagnostic_uwb.py --input t1.csv
    python diagnostic_uwb.py --input t1.csv t2.csv
    python diagnostic_uwb.py --input t1.csv --output rapport.txt
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import timedelta


# ---------------------------------------------------------------------------
# Constantes du terrain (metres)
# ---------------------------------------------------------------------------
POOL_X_MIN = 0.0
POOL_X_MAX = 25.0
POOL_Y_MIN = 0.0
POOL_Y_MAX = 26.0
POOL_Z_MIN = -2.0
POOL_Z_MAX = 2.0
MAX_REALISTIC_SPEED = 2.5  # m/s pour un nageur de water-polo


def separator(title=""):
    """Retourne une ligne de separation formatee."""
    if title:
        return f"\n{'=' * 70}\n  {title}\n{'=' * 70}\n"
    return "-" * 70


def load_csv(filepath):
    """Charge un fichier CSV et effectue les conversions de base."""
    df = pd.read_csv(filepath)
    expected_cols = {"time", "nodeID", "positionX", "positionY", "positionZ", "quality"}
    actual_cols = set(df.columns)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing:
        print(f"  ERREUR : colonnes manquantes : {missing}")
        sys.exit(1)

    df["time"] = pd.to_datetime(df["time"])
    return df, extra


def diagnostic_structure(df, filepath, extra_cols):
    """Diagnostic de la structure generale du fichier."""
    lines = []
    lines.append(separator("1. STRUCTURE DU FICHIER"))
    lines.append(f"  Fichier          : {os.path.basename(filepath)}")
    lines.append(f"  Taille           : {os.path.getsize(filepath) / 1e6:.2f} Mo")
    lines.append(f"  Nombre de lignes : {len(df)}")
    lines.append(f"  Colonnes         : {list(df.columns)}")
    if extra_cols:
        lines.append(f"  Colonnes inattendues : {extra_cols}")

    lines.append("")
    lines.append("  Types de donnees :")
    for col in df.columns:
        lines.append(f"    {col:15s} : {df[col].dtype}")

    nodes = sorted(df["nodeID"].unique())
    lines.append(f"\n  Nombre de tags (nodeID) : {len(nodes)}")
    lines.append(f"  Liste des tags          : {nodes}")

    return "\n".join(lines)


def diagnostic_completude(df):
    """Diagnostic de la completude des donnees (NaN, lignes vides)."""
    lines = []
    lines.append(separator("2. COMPLETUDE DES DONNEES"))

    total = len(df)
    nan_x = df["positionX"].isna().sum()
    nan_y = df["positionY"].isna().sum()
    nan_z = df["positionZ"].isna().sum()
    nan_q = df["quality"].isna().sum()
    nan_any = df[["positionX", "positionY", "positionZ"]].isna().any(axis=1).sum()

    lines.append(f"  Lignes totales                    : {total}")
    lines.append(f"  Lignes avec positionX = NaN       : {nan_x} ({100 * nan_x / total:.1f}%)")
    lines.append(f"  Lignes avec positionY = NaN       : {nan_y} ({100 * nan_y / total:.1f}%)")
    lines.append(f"  Lignes avec positionZ = NaN       : {nan_z} ({100 * nan_z / total:.1f}%)")
    lines.append(f"  Lignes avec quality = NaN         : {nan_q} ({100 * nan_q / total:.1f}%)")
    lines.append(f"  Lignes avec au moins un NaN (XYZ) : {nan_any} ({100 * nan_any / total:.1f}%)")
    lines.append(f"  Lignes exploitables (XYZ valides) : {total - nan_any} ({100 * (total - nan_any) / total:.1f}%)")

    lines.append(f"\n  Detail par tag :")
    lines.append(f"  {'Tag':>8s}  {'Total':>8s}  {'NaN':>8s}  {'%NaN':>8s}  {'Valides':>8s}")
    lines.append(f"  {'-' * 48}")

    for node in sorted(df["nodeID"].unique()):
        sub = df[df["nodeID"] == node]
        n = len(sub)
        nan_n = sub["positionX"].isna().sum()
        lines.append(
            f"  {node:>8s}  {n:>8d}  {nan_n:>8d}  {100 * nan_n / n:>7.1f}%  {n - nan_n:>8d}"
        )

    return "\n".join(lines)


def diagnostic_qualite(df):
    """Diagnostic du score de qualite."""
    lines = []
    lines.append(separator("3. SCORE DE QUALITE"))

    valid = df.dropna(subset=["positionX"])
    q = valid["quality"]

    lines.append(f"  Sur les {len(valid)} mesures valides :")
    lines.append(f"    Min      : {q.min()}")
    lines.append(f"    Max      : {q.max()}")
    lines.append(f"    Moyenne  : {q.mean():.1f}")
    lines.append(f"    Mediane  : {q.median():.0f}")
    lines.append(f"    Ecart-type : {q.std():.1f}")

    seuils = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    lines.append(f"\n  Distribution par seuil :")
    lines.append(f"  {'Seuil':>8s}  {'< seuil':>10s}  {'% < seuil':>10s}  {'>= seuil':>10s}  {'% >= seuil':>10s}")
    lines.append(f"  {'-' * 56}")
    for s in seuils:
        below = (q < s).sum()
        above = (q >= s).sum()
        lines.append(
            f"  {s:>8d}  {below:>10d}  {100 * below / len(q):>9.1f}%  "
            f"{above:>10d}  {100 * above / len(q):>9.1f}%"
        )

    lines.append(f"\n  Score de qualite moyen par tag :")
    for node in sorted(valid["nodeID"].unique()):
        sub = valid[valid["nodeID"] == node]
        lines.append(f"    {node}: moyenne = {sub['quality'].mean():.1f}, mediane = {sub['quality'].median():.0f}")

    return "\n".join(lines)


def diagnostic_spatial(df):
    """Diagnostic de la coherence spatiale (bornes, outliers)."""
    lines = []
    lines.append(separator("4. COHERENCE SPATIALE"))

    valid = df.dropna(subset=["positionX"]).copy()

    lines.append(f"  Bornes observees (sur {len(valid)} mesures valides) :")
    lines.append(f"    positionX : [{valid['positionX'].min():.3f}, {valid['positionX'].max():.3f}]  (attendu: [{POOL_X_MIN}, {POOL_X_MAX}])")
    lines.append(f"    positionY : [{valid['positionY'].min():.3f}, {valid['positionY'].max():.3f}]  (attendu: [{POOL_Y_MIN}, {POOL_Y_MAX}])")
    lines.append(f"    positionZ : [{valid['positionZ'].min():.3f}, {valid['positionZ'].max():.3f}]  (attendu: [{POOL_Z_MIN}, {POOL_Z_MAX}])")

    # Points hors limites
    margin = 1.0
    out_x = ((valid["positionX"] < POOL_X_MIN - margin) | (valid["positionX"] > POOL_X_MAX + margin)).sum()
    out_y = ((valid["positionY"] < POOL_Y_MIN - margin) | (valid["positionY"] > POOL_Y_MAX + margin)).sum()
    out_z = ((valid["positionZ"] < POOL_Z_MIN - margin) | (valid["positionZ"] > POOL_Z_MAX + margin)).sum()
    out_any = (
        (valid["positionX"] < POOL_X_MIN - margin) | (valid["positionX"] > POOL_X_MAX + margin) |
        (valid["positionY"] < POOL_Y_MIN - margin) | (valid["positionY"] > POOL_Y_MAX + margin) |
        (valid["positionZ"] < POOL_Z_MIN - margin) | (valid["positionZ"] > POOL_Z_MAX + margin)
    ).sum()

    lines.append(f"\n  Points hors limites (marge de {margin}m) :")
    lines.append(f"    Hors X : {out_x}")
    lines.append(f"    Hors Y : {out_y}")
    lines.append(f"    Hors Z : {out_z}")
    lines.append(f"    Total (au moins un axe) : {out_any} ({100 * out_any / len(valid):.3f}%)")

    lines.append(f"\n  Detail par tag :")
    for node in sorted(valid["nodeID"].unique()):
        sub = valid[valid["nodeID"] == node]
        out = (
            (sub["positionX"] < POOL_X_MIN - margin) | (sub["positionX"] > POOL_X_MAX + margin) |
            (sub["positionY"] < POOL_Y_MIN - margin) | (sub["positionY"] > POOL_Y_MAX + margin) |
            (sub["positionZ"] < POOL_Z_MIN - margin) | (sub["positionZ"] > POOL_Z_MAX + margin)
        ).sum()
        if out > 0:
            lines.append(f"    {node}: {out} points hors limites")

    return "\n".join(lines)


def diagnostic_temporel(df):
    """Diagnostic de la coherence temporelle (frequence, gaps, doublons)."""
    lines = []
    lines.append(separator("5. COHERENCE TEMPORELLE"))

    time_min = df["time"].min()
    time_max = df["time"].max()
    duration = (time_max - time_min).total_seconds()

    lines.append(f"  Debut        : {time_min}")
    lines.append(f"  Fin          : {time_max}")
    lines.append(f"  Duree totale : {duration:.1f}s ({duration / 60:.1f} min)")

    # Doublons
    dupes = df.duplicated(subset=["nodeID", "time"]).sum()
    lines.append(f"\n  Doublons (nodeID + time identiques) : {dupes}")

    # Frequence par tag
    lines.append(f"\n  Frequence d'echantillonnage par tag (sur mesures valides) :")
    lines.append(
        f"  {'Tag':>8s}  {'Mesures':>8s}  {'dt median':>10s}  {'dt moyen':>10s}  "
        f"{'Freq med.':>10s}  {'Max gap':>10s}  {'Gaps>2s':>8s}"
    )
    lines.append(f"  {'-' * 76}")

    valid = df.dropna(subset=["positionX"])

    for node in sorted(valid["nodeID"].unique()):
        sub = valid[valid["nodeID"] == node].sort_values("time")
        if len(sub) < 2:
            continue
        dt = sub["time"].diff().dt.total_seconds().dropna()
        dt_pos = dt[dt > 0]
        if len(dt_pos) == 0:
            continue
        median_dt = dt_pos.median()
        mean_dt = dt_pos.mean()
        freq = 1.0 / median_dt if median_dt > 0 else 0
        max_gap = dt_pos.max()
        gaps_2s = (dt_pos > 2.0).sum()

        lines.append(
            f"  {node:>8s}  {len(sub):>8d}  {median_dt:>9.4f}s  {mean_dt:>9.4f}s  "
            f"{freq:>9.1f}Hz  {max_gap:>9.1f}s  {gaps_2s:>8d}"
        )

    # Distribution des gaps
    lines.append(f"\n  Distribution des trous de tracking (tous tags confondus, mesures valides) :")
    all_gaps = []
    for node in sorted(valid["nodeID"].unique()):
        sub = valid[valid["nodeID"] == node].sort_values("time")
        if len(sub) < 2:
            continue
        dt = sub["time"].diff().dt.total_seconds().dropna()
        all_gaps.extend(dt[dt > 0].tolist())

    all_gaps = np.array(all_gaps)
    gap_thresholds = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    for t in gap_thresholds:
        count = (all_gaps > t).sum()
        lines.append(f"    Gaps > {t:>5.1f}s : {count}")

    return "\n".join(lines)


def diagnostic_anomalies(df):
    """Detection d'anomalies : teleportations, vitesses impossibles."""
    lines = []
    lines.append(separator("6. DETECTION D'ANOMALIES (TELEPORTATIONS)"))

    valid = df.dropna(subset=["positionX"]).copy()

    lines.append(
        f"  Seuil de vitesse realiste : {MAX_REALISTIC_SPEED} m/s\n"
    )
    lines.append(
        f"  {'Tag':>8s}  {'v > 3':>8s}  {'v > 5':>8s}  {'v > 10':>8s}  "
        f"{'v > 50':>8s}  {'v max':>12s}"
    )
    lines.append(f"  {'-' * 60}")

    total_teleports = 0

    for node in sorted(valid["nodeID"].unique()):
        sub = valid[valid["nodeID"] == node].sort_values("time")
        if len(sub) < 2:
            continue

        dx = sub["positionX"].diff()
        dy = sub["positionY"].diff()
        dt = sub["time"].diff().dt.total_seconds()
        dist = np.sqrt(dx ** 2 + dy ** 2)
        speed = dist / dt

        v3 = (speed > 3).sum()
        v5 = (speed > 5).sum()
        v10 = (speed > 10).sum()
        v50 = (speed > 50).sum()
        vmax = speed.max()

        total_teleports += v5
        lines.append(
            f"  {node:>8s}  {v3:>8d}  {v5:>8d}  {v10:>8d}  "
            f"{v50:>8d}  {vmax:>10.1f} m/s"
        )

    lines.append(f"\n  Total de points avec vitesse > 5 m/s : {total_teleports}")
    lines.append(f"  Ces points correspondent a des artefacts UWB (reflexions,")
    lines.append(f"  multipath, perte momentanee d'ancres) et doivent etre supprimes.")

    return "\n".join(lines)


def diagnostic_resume(df, filepath):
    """Resume des problemes detectes et recommandations."""
    lines = []
    lines.append(separator("7. RESUME ET RECOMMANDATIONS"))

    total = len(df)
    nan_count = df["positionX"].isna().sum()
    valid = df.dropna(subset=["positionX"])

    low_q = (valid["quality"] < 30).sum()

    margin = 1.0
    out_bounds = (
        (valid["positionX"] < POOL_X_MIN - margin) | (valid["positionX"] > POOL_X_MAX + margin) |
        (valid["positionY"] < POOL_Y_MIN - margin) | (valid["positionY"] > POOL_Y_MAX + margin) |
        (valid["positionZ"] < POOL_Z_MIN - margin) | (valid["positionZ"] > POOL_Z_MAX + margin)
    ).sum()

    teleports = 0
    for node in sorted(valid["nodeID"].unique()):
        sub = valid[valid["nodeID"] == node].sort_values("time")
        if len(sub) < 2:
            continue
        dx = sub["positionX"].diff()
        dy = sub["positionY"].diff()
        dt = sub["time"].diff().dt.total_seconds()
        speed = np.sqrt(dx ** 2 + dy ** 2) / dt
        teleports += (speed > 5).sum()

    lines.append(f"  Fichier : {os.path.basename(filepath)}")
    lines.append(f"  Lignes totales : {total}")
    lines.append(f"")
    lines.append(f"  Problemes detectes :")
    lines.append(f"    [NaN]            {nan_count:>8d} lignes  ({100 * nan_count / total:.1f}%)")
    lines.append(f"    [Qualite < 30]   {low_q:>8d} lignes  ({100 * low_q / len(valid):.1f}% des valides)")
    lines.append(f"    [Hors piscine]   {out_bounds:>8d} lignes  ({100 * out_bounds / len(valid):.3f}% des valides)")
    lines.append(f"    [Teleportations] {teleports:>8d} lignes  ({100 * teleports / len(valid):.1f}% des valides)")

    total_problemes = nan_count + low_q + out_bounds + teleports
    lines.append(f"")
    lines.append(f"  Total (avec recoupements possibles) : ~{total_problemes} anomalies")

    lines.append(f"")
    lines.append(f"  Recommandations de traitement :")
    lines.append(f"    1. Supprimer les lignes avec position NaN")
    lines.append(f"    2. Filtrer les mesures avec quality < 30")
    lines.append(f"    3. Supprimer les points hors des limites de la piscine")
    lines.append(f"    4. Supprimer les teleportations (vitesse > seuil, passes iteratives)")
    lines.append(f"    5. Appliquer un lissage (filtre median glissant, fenetre ~5 pts)")
    lines.append(f"    6. Reechantillonner a 10 Hz avec interpolation des gaps < 2s")

    return "\n".join(lines)


def run_diagnostic(filepath, output_path=None):
    """Execute le diagnostic complet sur un fichier."""
    print(f"\nChargement de {filepath}...")
    df, extra_cols = load_csv(filepath)

    sections = [
        diagnostic_structure(df, filepath, extra_cols),
        diagnostic_completude(df),
        diagnostic_qualite(df),
        diagnostic_spatial(df),
        diagnostic_temporel(df),
        diagnostic_anomalies(df),
        diagnostic_resume(df, filepath),
    ]

    header = separator(f"DIAGNOSTIC UWB - {os.path.basename(filepath)}")
    report = header + "\n".join(sections) + "\n"

    print(report)

    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(report)
        print(f"  Rapport sauvegarde dans : {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic des fichiers CSV UWB (DWM1001) pour le water-polo"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Un ou plusieurs fichiers CSV a analyser"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Fichier texte de sortie pour le rapport (optionnel)"
    )
    args = parser.parse_args()

    # Vider le fichier de sortie s'il existe
    if args.output and os.path.exists(args.output):
        os.remove(args.output)

    for filepath in args.input:
        if not os.path.exists(filepath):
            print(f"ERREUR : fichier introuvable : {filepath}")
            continue
        run_diagnostic(filepath, args.output)


if __name__ == "__main__":
    main()