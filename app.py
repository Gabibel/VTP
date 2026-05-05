# ==============================================================================
# PROJET VTP — Interface d'analyse de water-polo
# ==============================================================================
# Projet EFREI Research Lab × Fédération Française de Natation
#
# Lancer avec : streamlit run app.py
# ==============================================================================

import sys
import os
import io
import json
import pickle
from datetime import datetime
from contextlib import redirect_stdout

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Rendre le pipeline importable depuis le sous-dossier clean_data/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_data"))
from traitement_uwb import (
    Config,
    step_drop_nan,
    step_filter_quality,
    step_filter_bounds,
    step_remove_teleportations,
    step_smooth_median,
    step_resample_interpolate,
    step_add_derived_columns,
)

# scipy est optionnel : utilisé pour lisser la heatmap (dégradé plus doux)
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ==============================================================================
# CONFIGURATION STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Projet VTP — Water-Polo",
    layout="wide",
)


# ==============================================================================
# CONSTANTES DU TERRAIN
# ==============================================================================
POOL_X_MAX  = 25
POOL_Y_PLAY = (3, 23)
GOAL_LEFT   = (0,  13.25)
GOAL_RIGHT  = (25, 13.25)
ZONE_LINES  = [2, 6, 19, 23]

COULEURS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C",
    "#E67E22", "#34495E", "#E91E63", "#00BCD4", "#8BC34A", "#FF5722",
]

SAUVEGARDE_DIR = "sauvegarde"
MATCHS_DIR     = os.path.join(SAUVEGARDE_DIR, "matchs")


# ==============================================================================
# BIBLIOTHÈQUE DE MATCHS — fonctions de gestion
# ==============================================================================

def nom_en_slug(nom):
    """Transforme un nom de match en nom de fichier valide (sans caractères spéciaux)."""
    return nom.replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def lister_matchs():
    """Retourne la liste des slugs de tous les matchs sauvegardés, triée alphabétiquement."""
    if not os.path.exists(MATCHS_DIR):
        return []
    return sorted([
        f.replace("_info.json", "")
        for f in os.listdir(MATCHS_DIR)
        if f.endswith("_info.json")
    ])


def sauvegarder_match(df, nom_match, meta_extra=None):
    """
    Sauvegarde un match dans la bibliothèque locale.

    Crée deux fichiers dans sauvegarde/matchs/ :
        {slug}.pkl       — le DataFrame complet
        {slug}_info.json — nom, joueurs, date d'import, structure du match…

    meta_extra : dict optionnel fusionné dans le JSON de métadonnées.
    Retourne le slug.
    """
    os.makedirs(MATCHS_DIR, exist_ok=True)
    slug = nom_en_slug(nom_match)

    with open(os.path.join(MATCHS_DIR, f"{slug}.pkl"), "wb") as f:
        pickle.dump(df, f)

    info = {
        "nom":         nom_match,
        "slug":        slug,
        "nb_lignes":   len(df),
        "nb_joueurs":  int(df["nodeID"].nunique()),
        "date_import": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    if meta_extra:
        info.update(meta_extra)

    with open(os.path.join(MATCHS_DIR, f"{slug}_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return slug


def charger_match(slug):
    """Charge le DataFrame d'un match depuis la bibliothèque. Retourne None si introuvable."""
    chemin = os.path.join(MATCHS_DIR, f"{slug}.pkl")
    if os.path.exists(chemin):
        with open(chemin, "rb") as f:
            return pickle.load(f)
    return None


def info_match(slug):
    """Retourne le dictionnaire de métadonnées d'un match sauvegardé."""
    chemin = os.path.join(MATCHS_DIR, f"{slug}_info.json")
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"nom": slug, "nb_lignes": "?", "nb_joueurs": "?", "date_import": "?"}


def supprimer_match(slug):
    """Supprime les deux fichiers d'un match de la bibliothèque."""
    for suffixe in [".pkl", "_info.json"]:
        chemin = os.path.join(MATCHS_DIR, f"{slug}{suffixe}")
        if os.path.exists(chemin):
            os.remove(chemin)


# ==============================================================================
# PERSISTANCE DE SESSION
# ==============================================================================

def sauvegarder_session():
    os.makedirs(SAUVEGARDE_DIR, exist_ok=True)
    meta = {
        "match_selec":     st.session_state.match_selec,
        "mapping_joueurs": st.session_state.mapping_joueurs,
        "actions":         st.session_state.actions,
    }
    with open(os.path.join(SAUVEGARDE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def charger_session():
    chemin_meta = os.path.join(SAUVEGARDE_DIR, "meta.json")
    if os.path.exists(chemin_meta):
        try:
            with open(chemin_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if st.session_state.match_selec is None:
                st.session_state.match_selec = meta.get("match_selec")
            if not st.session_state.mapping_joueurs:
                st.session_state.mapping_joueurs = meta.get("mapping_joueurs", {})
            if not st.session_state.actions:
                st.session_state.actions = meta.get("actions", [])
            if st.session_state.donnees is None and st.session_state.match_selec:
                slug = nom_en_slug(st.session_state.match_selec)
                df = charger_match(slug)
                if df is not None:
                    st.session_state.donnees = df
        except Exception:
            pass


# ==============================================================================
# DONNÉES PARTAGÉES ENTRE LES PAGES
# ==============================================================================
valeurs_defaut = {
    "page":            "Configuration & Données",
    "donnees":         None,
    "match_selec":     None,
    "mapping_joueurs": {},
    "actions":         [],
    "show_form":       False,
}
for cle, valeur in valeurs_defaut.items():
    if cle not in st.session_state:
        st.session_state[cle] = valeur

if "session_chargee" not in st.session_state:
    charger_session()
    st.session_state["session_chargee"] = True


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def dessiner_bassin(ax, titre=""):
    ax.set_facecolor("#87CEEB")
    zone_jeu = patches.Rectangle(
        (0, POOL_Y_PLAY[0]), POOL_X_MAX, POOL_Y_PLAY[1] - POOL_Y_PLAY[0],
        linewidth=2, edgecolor="white", facecolor="#1565C0", alpha=0.65,
    )
    ax.add_patch(zone_jeu)
    for x in ZONE_LINES:
        couleur_ligne = "red" if x in [2, 23] else "yellow"
        ax.axvline(x=x, color=couleur_ligne, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(x, POOL_Y_PLAY[1] + 0.4, f"{x}m",
                ha="center", va="bottom", color=couleur_ligne, fontsize=7)
    for (gx, gy) in [GOAL_LEFT, GOAL_RIGHT]:
        decal_x = -0.2 if gx == 0 else 0.0
        but = patches.Rectangle(
            (gx + decal_x, gy - 0.45), 0.3, 0.9,
            linewidth=2, edgecolor="black", facecolor="white", zorder=4,
        )
        ax.add_patch(but)
    ax.set_xlim(-1, 26)
    ax.set_ylim(0, 27)
    ax.set_aspect("equal")
    ax.set_xlabel("X — longueur (m)", fontsize=9)
    ax.set_ylabel("Y — largeur (m)", fontsize=9)
    if titre:
        ax.set_title(titre, fontsize=11)
    return ax


def executer_pipeline(df_brut, qualite_min=30, vitesse_max=1.3):
    """
    Applique les 7 étapes du pipeline de nettoyage.
    Le filtre médian utilise une fenêtre fixe de 5 points (valeur standard).
    """
    cfg = Config(min_quality=qualite_min, max_speed=vitesse_max, median_window=5)
    logs = []
    df = df_brut.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["nodeID", "time"]).reset_index(drop=True)

    sink = io.StringIO()

    def log_etape(nom, avant, apres):
        diff = avant - apres
        if diff > 0:
            logs.append(f"{nom} : {diff:,} lignes supprimees ({avant:,} -> {apres:,})")
        elif diff < 0:
            logs.append(f"{nom} : {-diff:,} lignes ajoutees ({avant:,} -> {apres:,})")
        else:
            logs.append(f"{nom} : aucune ligne supprimee ({avant:,})")

    with redirect_stdout(sink):
        avant = len(df); df, _ = step_drop_nan(df)
        log_etape("Suppression des NaN", avant, len(df))

        avant = len(df); df, _ = step_filter_quality(df, cfg)
        log_etape(f"Filtre qualite (>= {qualite_min})", avant, len(df))

        avant = len(df); df, _ = step_filter_bounds(df, cfg)
        log_etape("Filtre geographique (hors bassin)", avant, len(df))

        avant = len(df); df, _ = step_remove_teleportations(df, cfg)
        log_etape(f"Anti-teleportation (vitesse max {vitesse_max} m/s)", avant, len(df))

        df, _ = step_smooth_median(df, cfg)
        logs.append("Lissage median (fenetre = 5 points) : aucune suppression")

        avant = len(df); df, _ = step_resample_interpolate(df, cfg)
        log_etape("Reechantillonnage a 10 Hz", avant, len(df))

        df, _ = step_add_derived_columns(df, cfg, session_name="import")
        logs.append("Colonnes calculees : vitesse, distance, zones, periodes")

    return df, logs


def nom_joueur(node_id):
    info = st.session_state.mapping_joueurs.get(str(node_id), {})
    bonnet = info.get("bonnet", "")
    return f"Bonnet {bonnet}" if bonnet else str(node_id)


def couleur_joueur(node_ids_tries, node_id):
    try:
        idx = list(node_ids_tries).index(str(node_id))
    except ValueError:
        idx = 0
    return COULEURS[idx % len(COULEURS)]


def formater_temps(secondes):
    s = int(secondes)
    return f"{s // 60}:{s % 60:02d}"


def bloc_infos_match(cle_prefix=""):
    """
    Bloc de saisie des informations sur le match.

    Pose d'abord la question : ce fichier contient-il un seul temps ou le match complet ?
    Selon la réponse, les champs affichés changent.

    cle_prefix : préfixe unique pour éviter les conflits de clés Streamlit (ex: "csv_", "xlsx_")

    Retourne un dict avec toutes les métadonnées saisies.
    """
    st.markdown("**Informations sur le match**")

    date_match = st.date_input(
        "Date du match",
        value=None,
        key=f"{cle_prefix}date_match",
        help="Date à laquelle s'est déroulé le match ou l'entrainement.",
    )

    type_contenu = st.radio(
        "Ces fichiers contiennent :",
        ["Le match complet", "Un ou plusieurs temps (pas le match entier)"],
        horizontal=True,
        key=f"{cle_prefix}type_contenu",
    )

    if type_contenu == "Le match complet":
        # Match complet : on demande juste la structure
        col_nb, col_dur = st.columns(2)
        nb_temps = col_nb.number_input(
            "Nombre de temps dans ce match",
            min_value=1, max_value=8, value=4, step=1,
            key=f"{cle_prefix}nb_temps",
            help="Ex : 4 temps pour un match officiel.",
        )
        duree_temps_min = col_dur.number_input(
            "Duree d'un temps (min)",
            min_value=1, max_value=60, value=12, step=1,
            key=f"{cle_prefix}duree_temps",
            help="Ex : 12 min en elite, 8 min en U17.",
        )
        return {
            "date_match":            str(date_match) if date_match else None,
            "type_contenu":          "match_complet",
            "nb_temps_total":        int(nb_temps),
            "nb_temps_importes":     int(nb_temps),
            "duree_temps_min":       int(duree_temps_min),
        }

    else:
        # Import partiel : on demande la structure du match ET ce qu'on importe
        st.caption(
            "Indiquez la structure complete du match, puis combien de temps "
            "sont presents dans les fichiers que vous importez maintenant."
        )
        col_total, col_importe, col_dur = st.columns(3)
        nb_temps_total = col_total.number_input(
            "Nombre de temps au total dans le match",
            min_value=1, max_value=8, value=4, step=1,
            key=f"{cle_prefix}nb_temps_total",
            help="Nombre de periodes prevues dans ce match (ex : 4).",
        )
        nb_temps_importes = col_importe.number_input(
            "Nombre de temps que vous importez",
            min_value=1, max_value=8, value=1, step=1,
            key=f"{cle_prefix}nb_temps_importes",
            help="Combien de periodes sont presentes dans les fichiers ci-dessus.",
        )
        duree_temps_min = col_dur.number_input(
            "Duree d'un temps (min)",
            min_value=1, max_value=60, value=12, step=1,
            key=f"{cle_prefix}duree_temps",
            help="Ex : 12 min en elite, 8 min en U17.",
        )

        # Validation simple : on ne peut pas importer plus de temps qu'il n'en existe
        if nb_temps_importes > nb_temps_total:
            st.warning(
                f"Vous ne pouvez pas importer {int(nb_temps_importes)} temps "
                f"si le match n'en contient que {int(nb_temps_total)}."
            )

        return {
            "date_match":        str(date_match) if date_match else None,
            "type_contenu":      "partiel",
            "nb_temps_total":    int(nb_temps_total),
            "nb_temps_importes": int(nb_temps_importes),
            "duree_temps_min":   int(duree_temps_min),
        }


def resume_structure(info):
    """
    Construit une courte phrase décrivant la structure d'un match à partir de ses métadonnées.
    Ex : "4x12min — complet" ou "2/4 temps de 12min"
    """
    nb_total   = info.get("nb_temps_total")
    nb_importe = info.get("nb_temps_importes")
    duree      = info.get("duree_temps_min")
    type_c     = info.get("type_contenu", "")

    if not nb_total or not duree:
        return "—"

    if type_c == "match_complet":
        return f"{nb_total}x{duree}min (complet)"
    else:
        return f"{nb_importe}/{nb_total} temps de {duree}min"


# ==============================================================================
# BARRE LATÉRALE — Navigation principale
# ==============================================================================
with st.sidebar:
    st.markdown("## PROJET VTP")
    st.markdown("---")

    for nom_page in ["Configuration & Données", "Replay & Vue Globale", "Analyse Dynamique"]:
        actif = st.session_state.page == nom_page
        if st.button(
            nom_page, key=f"nav_{nom_page}", use_container_width=True,
            type="primary" if actif else "secondary",
        ):
            st.session_state.page = nom_page
            st.rerun()

    if st.session_state.match_selec:
        st.markdown("---")
        st.caption(f"Match actif : **{st.session_state.match_selec}**")
        if st.button("Décharger le match", use_container_width=True):
            st.session_state.donnees     = None
            st.session_state.match_selec = None
            sauvegarder_session()
            st.rerun()


# ==============================================================================
# PAGE 1 — CONFIGURATION & DONNÉES
# ==============================================================================
if st.session_state.page == "Configuration & Données":

    st.title("Configuration & Données")

    # ------------------------------------------------------------------
    # Section 1 : Importer un nouveau match
    # ------------------------------------------------------------------
    st.subheader("Importer un nouveau match")

    with st.container(border=True):
        nom_match = st.text_input(
            "Nom du match",
            placeholder="ex : Entrainement 03/02/2026  ou  Match INSEP vs Lyon",
            help="Ce nom identifiera le match dans la bibliothèque.",
        )

        fichiers = st.file_uploader(
            "Fichiers à importer",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            help="CSV = données brutes UWB (pipeline de nettoyage appliqué automatiquement). "
                 "XLSX = données déjà nettoyées (chargement direct). "
                 "Vous pouvez déposer plusieurs fichiers d'un même match.",
        )

        if fichiers:
            types = {f.name.rsplit(".", 1)[-1].lower() for f in fichiers}

            if types == {"csv"}:
                # ---- Fichiers CSV bruts : pipeline de nettoyage ----
                st.markdown("**Paramètres de nettoyage**")
                col_q, col_v = st.columns(2)
                qualite = col_q.slider(
                    "Qualité minimum (0–100)", 0, 100, 30,
                    help="Supprimer les mesures dont le signal UWB est trop faible.",
                )
                vitesse = col_v.slider(
                    "Vitesse max autorisée (m/s)", 0.5, 5.0, 1.3, step=0.1,
                    help="Supprimer les sauts de position physiquement impossibles.",
                )

                st.markdown("---")
                infos = bloc_infos_match(cle_prefix="csv_")

                nb_total_lignes = sum(len(pd.read_csv(f)) for f in fichiers)
                st.caption(
                    f"{len(fichiers)} fichier(s) CSV — "
                    f"environ {nb_total_lignes:,} lignes brutes au total"
                )

                # Avertissement si import partiel et nb_temps_importes > nb_temps_total
                ok_structure = (
                    infos["nb_temps_importes"] <= infos["nb_temps_total"]
                )

                if not nom_match:
                    st.warning("Entrez un nom pour le match avant de continuer.")
                elif not ok_structure:
                    st.error("Corrigez la structure du match avant de continuer.")
                elif st.button("Traiter et sauvegarder ce match", type="primary"):
                    for f in fichiers:
                        f.seek(0)
                    dfs = [pd.read_csv(f) for f in fichiers]
                    df_brut = pd.concat(dfs, ignore_index=True)

                    colonnes_requises = {
                        "time", "nodeID", "positionX", "positionY", "positionZ", "quality"
                    }
                    colonnes_absentes = colonnes_requises - set(df_brut.columns)
                    if colonnes_absentes:
                        st.error(f"Colonnes manquantes dans le CSV : {colonnes_absentes}")
                    else:
                        with st.spinner("Nettoyage en cours… (30–60 s pour les gros fichiers)"):
                            df_clean, logs = executer_pipeline(df_brut, qualite, vitesse)

                        sauvegarder_match(df_clean, nom_match, meta_extra=infos)
                        st.session_state.donnees     = df_clean
                        st.session_state.match_selec = nom_match
                        sauvegarder_session()

                        date_txt = infos["date_match"] if infos["date_match"] else "date non renseignée"
                        st.success(
                            f"Match **{nom_match}** sauvegardé — "
                            f"{len(df_clean):,} lignes, {df_clean['nodeID'].nunique()} joueurs"
                        )
                        st.info(
                            f"Date : {date_txt} | Structure : {resume_structure(infos)}"
                        )
                        with st.expander("Détail des étapes du pipeline"):
                            for msg in logs:
                                st.write(msg)

            elif types == {"xlsx"}:
                # ---- Fichiers XLSX déjà nettoyés : chargement direct ----
                st.caption(
                    f"{len(fichiers)} fichier(s) XLSX — chargement direct sans pipeline."
                )

                st.markdown("---")
                infos = bloc_infos_match(cle_prefix="xlsx_")

                ok_structure = (
                    infos["nb_temps_importes"] <= infos["nb_temps_total"]
                )

                if not nom_match:
                    st.warning("Entrez un nom pour le match avant de continuer.")
                elif not ok_structure:
                    st.error("Corrigez la structure du match avant de continuer.")
                elif st.button("Sauvegarder ce match", type="primary"):
                    for f in fichiers:
                        f.seek(0)
                    dfs = [pd.read_excel(f) for f in fichiers]
                    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

                    sauvegarder_match(df, nom_match, meta_extra=infos)
                    st.session_state.donnees     = df
                    st.session_state.match_selec = nom_match
                    sauvegarder_session()

                    date_txt = infos["date_match"] if infos["date_match"] else "date non renseignée"
                    st.success(
                        f"Match **{nom_match}** sauvegardé — "
                        f"{len(df):,} lignes, {df['nodeID'].nunique()} joueurs"
                    )
                    st.info(
                        f"Date : {date_txt} | Structure : {resume_structure(infos)}"
                    )

            else:
                st.warning("Mélange CSV et XLSX non supporté. Importez un seul type à la fois.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 2 : Bibliothèque des matchs sauvegardés
    # ------------------------------------------------------------------
    st.subheader("Charger un match")

    matchs = lister_matchs()

    if not matchs:
        st.info("Aucun match sauvegardé. Importez un match ci-dessus pour commencer.")
    else:
        col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([2.5, 1, 2, 2, 1.5])
        col_h1.markdown("**Match**")
        col_h2.markdown("**Joueurs**")
        col_h3.markdown("**Structure**")
        col_h4.markdown("**Importé le**")

        for slug in matchs:
            info  = info_match(slug)
            nom   = info.get("nom", slug)
            actif = st.session_state.match_selec == nom

            col_nom, col_j, col_str, col_d, col_btns = st.columns([2.5, 1, 2, 2, 1.5])

            prefix = "▶ " if actif else ""
            col_nom.markdown(f"**{prefix}{nom}**" if actif else nom)
            col_j.write(str(info.get("nb_joueurs", "?")))
            col_str.write(resume_structure(info))
            col_d.write(info.get("date_import", "?"))

            btn_charge, btn_suppr = col_btns.columns(2)

            if btn_charge.button(
                "Charger", key=f"charge_{slug}",
                type="primary" if actif else "secondary",
            ):
                df_charge = charger_match(slug)
                if df_charge is not None:
                    st.session_state.donnees     = df_charge
                    st.session_state.match_selec = nom
                    sauvegarder_session()
                    st.rerun()
                else:
                    st.error(f"Fichier introuvable pour '{nom}'.")

            if btn_suppr.button(
                "Suppr.", key=f"suppr_{slug}",
                help=f"Supprimer le match '{nom}'",
            ):
                supprimer_match(slug)
                if st.session_state.match_selec == nom:
                    st.session_state.donnees     = None
                    st.session_state.match_selec = None
                    sauvegarder_session()
                st.rerun()

    if st.session_state.donnees is not None:
        df = st.session_state.donnees
        st.caption(
            f"Match actif : **{st.session_state.match_selec}** — "
            f"{len(df):,} lignes, {df['nodeID'].nunique()} joueurs"
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 3 : Répartition des joueurs (Tag → Bonnet)
    # ------------------------------------------------------------------
    st.subheader("Répartition des joueurs")

    if st.session_state.donnees is not None:
        node_ids = sorted(st.session_state.donnees["nodeID"].unique().astype(str))
    else:
        node_ids = [f"Tag {i}" for i in range(1, 13)]

    for node in node_ids:
        if node not in st.session_state.mapping_joueurs:
            st.session_state.mapping_joueurs[node] = {"bonnet": "", "equipe": "INSEP"}

    col_h1, col_h2 = st.columns(2)
    col_h1.markdown("**Tag**")
    col_h2.markdown("**Bonnet**")

    for node in node_ids:
        col_tag, col_bonnet = st.columns(2)
        col_tag.write(str(node))
        nouveau_bonnet = col_bonnet.text_input(
            label=f"bonnet_{node}",
            value=st.session_state.mapping_joueurs[node]["bonnet"],
            key=f"input_{node}",
            label_visibility="collapsed",
            placeholder="N° bonnet",
        )
        st.session_state.mapping_joueurs[node]["bonnet"] = nouveau_bonnet

    st.markdown("")
    if st.button("Enregistrer la composition", type="primary"):
        if st.session_state.match_selec and st.session_state.donnees is not None:
            slug = nom_en_slug(st.session_state.match_selec)
            chemin_info = os.path.join(MATCHS_DIR, f"{slug}_info.json")
            if os.path.exists(chemin_info):
                with open(chemin_info, "r", encoding="utf-8") as f:
                    info = json.load(f)
                info["mapping_joueurs"] = st.session_state.mapping_joueurs
                with open(chemin_info, "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
        sauvegarder_session()
        st.success("Composition enregistrée !")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 4 : Visualisation du match (vue d'ensemble)
    # ------------------------------------------------------------------
    st.subheader("Visualisation du match")

    if st.session_state.donnees is None:
        st.info("Chargez un match pour afficher la visualisation.")
    else:
        df = st.session_state.donnees
        node_ids = sorted(df["nodeID"].unique().astype(str))

        fig_vue, ax_vue = plt.subplots(figsize=(14, 7))
        dessiner_bassin(ax_vue, f"Vue d'ensemble — {st.session_state.match_selec or 'match'}")

        for node in node_ids:
            data_node = df[df["nodeID"].astype(str) == node]
            nb = min(500, len(data_node))
            sample = data_node.sample(nb, random_state=42)
            couleur = couleur_joueur(node_ids, node)
            ax_vue.scatter(
                sample["positionX"], sample["positionY"],
                s=4, alpha=0.35, color=couleur, label=nom_joueur(node),
            )

        ax_vue.legend(loc="upper right", fontsize=8, markerscale=3, ncol=2, framealpha=0.9)
        st.pyplot(fig_vue)
        plt.close()


# ==============================================================================
# PAGE 2 — REPLAY & VUE GLOBALE
# ==============================================================================
elif st.session_state.page == "Replay & Vue Globale":

    st.title("Replay & Vue Globale")

    if st.session_state.donnees is None:
        st.warning("Aucune donnée chargée. Allez d'abord dans **Configuration & Données**.")
        st.stop()

    df = st.session_state.donnees
    node_ids = sorted(df["nodeID"].unique().astype(str))

    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtres")

        if "period" in df.columns:
            periodes = ["Toutes"] + [
                str(int(p)) for p in sorted(df["period"].dropna().unique())
            ]
        else:
            periodes = ["Toutes"]
        periode_choisie = st.selectbox("Période", periodes)

        st.markdown("**Slider temporel**")
        temps_max    = int(df["elapsed_s"].max()) if "elapsed_s" in df.columns else 120
        temps_actuel = st.slider("Temps (s)", 0, temps_max, min(60, temps_max))

    if periode_choisie != "Toutes" and "period" in df.columns:
        df_filtre = df[df["period"] == int(periode_choisie)]
    else:
        df_filtre = df

    st.subheader("Visualisation du match")
    col_bassin, col_video = st.columns(2)

    with col_bassin:
        if "elapsed_s" in df_filtre.columns:
            fenetre_t = df_filtre[
                (df_filtre["elapsed_s"] >= temps_actuel - 2) &
                (df_filtre["elapsed_s"] <= temps_actuel + 2)
            ]
            positions_joueurs = (
                fenetre_t.sort_values("elapsed_s")
                .groupby("nodeID").last().reset_index()
            )
            positions_joueurs["nodeID"] = positions_joueurs["nodeID"].astype(str)
        else:
            positions_joueurs = pd.DataFrame()

        fig_rep, ax_rep = plt.subplots(figsize=(6, 9))
        dessiner_bassin(ax_rep, f"t = {formater_temps(temps_actuel)}")

        for _, joueur in positions_joueurs.iterrows():
            nid = str(joueur["nodeID"])
            couleur = couleur_joueur(node_ids, nid)
            cercle = plt.Circle(
                (joueur["positionX"], joueur["positionY"]),
                radius=0.6, color=couleur, zorder=5,
            )
            ax_rep.add_patch(cercle)
            label = st.session_state.mapping_joueurs.get(nid, {}).get("bonnet", "") or nid
            ax_rep.text(
                joueur["positionX"], joueur["positionY"],
                str(label), ha="center", va="center",
                color="white", fontsize=7, fontweight="bold", zorder=6,
            )

        st.pyplot(fig_rep)
        plt.close()

    with col_video:
        st.markdown("#### Vidéo")
        st.info("Vidéo non disponible — fonctionnalité prévue.")
        if not positions_joueurs.empty and "speed" in positions_joueurs.columns:
            st.markdown(f"**Joueurs à t = {formater_temps(temps_actuel)} :**")
            for _, j in positions_joueurs.iterrows():
                nid = str(j["nodeID"])
                vitesse_kmh = j["speed"] * 3.6 if pd.notna(j.get("speed")) else 0
                zone = j.get("zone", "—")
                st.write(f"- **{nom_joueur(nid)}** — {vitesse_kmh:.1f} km/h — zone : {zone}")

    if st.button("Enregistrer cette séquence", type="primary"):
        st.session_state.show_form = True

    if st.session_state.show_form:
        with st.container(border=True):
            st.markdown("#### Nom de l'action")
            nom_action = st.text_input(
                "", placeholder="ex : But en contre-attaque", label_visibility="collapsed",
            )
            st.markdown("#### Temps de l'action")
            col_deb, col_fin = st.columns(2)
            debut_s = col_deb.number_input(
                "Début", value=max(0, temps_actuel - 5), min_value=0, max_value=temps_max,
            )
            fin_s = col_fin.number_input(
                "Fin", value=min(temps_max, temps_actuel + 5), min_value=0, max_value=temps_max,
            )
            st.markdown("#### Type d'action")
            type_action = st.selectbox(
                "Menu déroulant",
                ["But", "Erreur", "Phase de jeu", "Faute", "Arrêt", "Autre"],
                label_visibility="collapsed",
            )
            st.markdown("#### Commentaire")
            commentaire = st.text_area("", label_visibility="collapsed")

            col_ok, col_non = st.columns(2)
            if col_ok.button("Valider", type="primary"):
                st.session_state.actions.append({
                    "nom":         nom_action or f"Action {len(st.session_state.actions) + 1}",
                    "type":        type_action,
                    "debut_s":     int(debut_s),
                    "fin_s":       int(fin_s),
                    "commentaire": commentaire,
                })
                st.session_state.show_form = False
                sauvegarder_session()
                st.rerun()
            if col_non.button("Annuler"):
                st.session_state.show_form = False
                st.rerun()

    st.markdown("---")
    st.subheader("Bibliothèque d'actions")

    col_a, col_b, col_c, col_d, col_e = st.columns([2, 1.5, 1.5, 3, 1])
    col_a.markdown("**Action**")
    col_b.markdown("**Type**")
    col_c.markdown("**Temps**")
    col_d.markdown("**Commentaires**")

    if not st.session_state.actions:
        st.caption("Aucune action enregistrée — utilisez le bouton ci-dessus.")
    else:
        for i, action in enumerate(st.session_state.actions):
            col_a, col_b, col_c, col_d, col_e = st.columns([2, 1.5, 1.5, 3, 1])
            col_a.write(action["nom"])
            col_b.write(action["type"])
            col_c.write(formater_temps(action["debut_s"]))
            col_d.write(action.get("commentaire", "…") or "…")
            col_e.button("consulter", key=f"consult_{i}")


# ==============================================================================
# PAGE 3 — ANALYSE DYNAMIQUE
# ==============================================================================
elif st.session_state.page == "Analyse Dynamique":

    st.title("Analyse Dynamique")

    if st.session_state.donnees is None:
        st.warning("Aucune donnée chargée. Allez d'abord dans **Configuration & Données**.")
        st.stop()

    df = st.session_state.donnees
    node_ids = sorted(df["nodeID"].unique().astype(str))

    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtres")
        mode = st.radio("Type d'analyse", ["Individuel", "Tactique"])

    # ==================================================================
    # MODE INDIVIDUEL
    # ==================================================================
    if mode == "Individuel":

        st.subheader("Sélection du joueur")
        labels = {node: nom_joueur(node) for node in node_ids}
        label_choisi = st.selectbox(
            "Joueur", list(labels.values()), label_visibility="collapsed",
        )
        node_choisi = [k for k, v in labels.items() if v == label_choisi][0]

        df_joueur = df[df["nodeID"].astype(str) == node_choisi].copy()

        st.subheader("Performance")

        if "elapsed_s" in df_joueur.columns:
            duree_s  = df_joueur["elapsed_s"].max() - df_joueur["elapsed_s"].min()
            temps_eau = formater_temps(duree_s) + " min"
        else:
            temps_eau = "N/A"

        if "distance_cumul" in df_joueur.columns:
            distance = f"{df_joueur['distance_cumul'].max() / 1000:.1f} km"
        else:
            distance = "N/A"

        if "speed" in df_joueur.columns:
            vitesse_moy = f"{df_joueur['speed'].mean() * 3.6:.1f} km/h"
            vitesse_max = f"{df_joueur['speed'].max() * 3.6:.1f} km/h"
        else:
            vitesse_moy = vitesse_max = "N/A"

        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric("Temps dans l'eau", temps_eau)
        col_2.metric("Distance totale",  distance)
        col_3.metric("Vitesse moyenne",  vitesse_moy)
        col_4.metric("Vitesse max",      vitesse_max)

        st.subheader("Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(13, 6))
        dessiner_bassin(ax_heat, f"Heatmap de présence — {label_choisi}")

        x_vals = df_joueur["positionX"].dropna()
        y_vals = df_joueur["positionY"].dropna()

        if len(x_vals) > 20:
            heatmap_data, _, _ = np.histogram2d(
                x_vals, y_vals, bins=[50, 40], range=[[0, 25], [3, 23]],
            )
            if SCIPY_OK:
                heatmap_data = gaussian_filter(heatmap_data, sigma=1.5)
            palette = LinearSegmentedColormap.from_list(
                "vtp_heat", ["#2196F3", "#FFA726", "#EF5350"]
            )
            im = ax_heat.imshow(
                heatmap_data.T, extent=[0, 25, 3, 23],
                origin="lower", cmap=palette, alpha=0.75, aspect="auto",
            )
            plt.colorbar(im, ax=ax_heat, label="Densité de présence", shrink=0.7)
        else:
            ax_heat.text(12.5, 13, "Pas assez de données",
                         ha="center", color="white", fontsize=12)

        st.pyplot(fig_heat)
        plt.close()

        st.subheader("Courbe de vitesse")
        if "speed" in df_joueur.columns and "elapsed_s" in df_joueur.columns:
            df_vit = (
                df_joueur.dropna(subset=["speed", "elapsed_s"])
                .sort_values("elapsed_s").copy()
            )
            df_vit["tranche_10s"] = (df_vit["elapsed_s"] // 10) * 10
            vitesse_lissee = df_vit.groupby("tranche_10s")["speed"].mean().reset_index()
            vitesse_lissee["km_h"]    = vitesse_lissee["speed"] * 3.6
            vitesse_lissee["minutes"] = vitesse_lissee["tranche_10s"] / 60

            fig_vit, ax_vit = plt.subplots(figsize=(13, 4))
            ax_vit.plot(vitesse_lissee["minutes"], vitesse_lissee["km_h"],
                        color="#1565C0", linewidth=2, marker="o", markersize=3)
            ax_vit.fill_between(vitesse_lissee["minutes"], vitesse_lissee["km_h"],
                                 alpha=0.12, color="#1565C0")
            ax_vit.set_xlabel("Temps (min)")
            ax_vit.set_ylabel("Vitesse (km/h)")
            ax_vit.grid(axis="y", linestyle="--", alpha=0.4)
            ax_vit.set_facecolor("#F8F9FA")
            st.pyplot(fig_vit)
            plt.close()
        else:
            st.info("Colonne 'speed' non disponible (nettoyage requis).")

    # ==================================================================
    # MODE TACTIQUE
    # ==================================================================
    elif mode == "Tactique":

        st.subheader("Vue tactique — Répartition par zones")

        if "zone" not in df.columns:
            st.info("Les données de zone ne sont pas disponibles (nettoyage requis).")
        else:
            zones_ordre = ["2m_gauche", "ad_gauche", "transition", "ad_droite", "2m_droite"]
            counts = df.groupby(["nodeID", "zone"]).size().unstack(fill_value=0)
            zones_dispo = [z for z in zones_ordre if z in counts.columns]
            counts = counts[zones_dispo]
            pourcentages = counts.div(counts.sum(axis=1), axis=0) * 100
            pourcentages.index = [nom_joueur(str(n)) for n in pourcentages.index]

            couleurs_zones = {
                "2m_gauche":  "#E74C3C",
                "ad_gauche":  "#F39C12",
                "transition": "#3498DB",
                "ad_droite":  "#F39C12",
                "2m_droite":  "#E74C3C",
            }
            couleurs = [couleurs_zones.get(z, "#95A5A6") for z in zones_dispo]

            fig_zones, ax_zones = plt.subplots(figsize=(13, 5))
            pourcentages.plot(kind="bar", ax=ax_zones, stacked=True,
                              color=couleurs, edgecolor="white", linewidth=0.5)
            ax_zones.set_xlabel("Joueur", fontsize=10)
            ax_zones.set_ylabel("% du temps", fontsize=10)
            ax_zones.set_title("Temps passé dans chaque zone — tous les joueurs", fontsize=11)
            ax_zones.legend(title="Zone", loc="upper right", fontsize=8)
            ax_zones.tick_params(axis="x", rotation=45)
            ax_zones.set_facecolor("#F8F9FA")
            st.pyplot(fig_zones)
            plt.close()

        st.subheader("Trajectoires sur le bassin")
        fig_traj, ax_traj = plt.subplots(figsize=(13, 7))
        dessiner_bassin(ax_traj, "Trajectoires de tous les joueurs")

        for node in node_ids:
            data_node = (
                df[df["nodeID"].astype(str) == node]
                .sort_values("elapsed_s").iloc[::5]
            )
            couleur = couleur_joueur(node_ids, node)
            ax_traj.plot(
                data_node["positionX"], data_node["positionY"],
                color=couleur, alpha=0.45, linewidth=0.8, label=nom_joueur(node),
            )

        ax_traj.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
        st.pyplot(fig_traj)
        plt.close()