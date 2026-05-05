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
    page_icon="🏊",
    layout="wide",
)


# ==============================================================================
# CONSTANTES DU TERRAIN
# ==============================================================================
POOL_X_MAX   = 25          # longueur du bassin (mètres)
POOL_Y_PLAY  = (3, 23)     # Y_min, Y_max de la zone de jeu
GOAL_LEFT    = (0,  13.25) # coordonnées du but gauche
GOAL_RIGHT   = (25, 13.25) # coordonnées du but droit
ZONE_LINES   = [2, 6, 19, 23]  # lignes officielles sur l'axe X

# 12 couleurs distinctes, une par joueur
COULEURS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C",
    "#E67E22", "#34495E", "#E91E63", "#00BCD4", "#8BC34A", "#FF5722",
]

# Dossiers contenant les données par date de match
DOSSIERS_MATCHS = {
    "13/01/2026": "dwm_positions_130126",
    "20/01/2026": "dwm_positions_200126",
    "03/02/2026": "dwm_positions_030226",
}


# ==============================================================================
# DONNÉES PARTAGÉES ENTRE LES PAGES
# Les variables dans st.session_state persistent pendant toute la session.
# ==============================================================================
valeurs_defaut = {
    "page":            "Configuration & Données",
    "donnees":         None,   # DataFrame principal (données nettoyées)
    "match_selec":     None,   # Date du match sélectionné ("13/01/2026", …)
    "mapping_joueurs": {},     # nodeID → {"bonnet": str, "equipe": str}
    "actions":         [],     # Liste des actions enregistrées
    "show_form":       False,  # True = afficher le formulaire d'enregistrement
}
for cle, valeur in valeurs_defaut.items():
    if cle not in st.session_state:
        st.session_state[cle] = valeur


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def dessiner_bassin(ax, titre=""):
    """
    Dessine le bassin de water-polo sur un axe matplotlib.

    Le bassin est orienté ainsi (cf. cahier des charges) :
      - Axe X (0–25 m) : longueur, buts aux extrémités
      - Axe Y (3–23 m) : largeur de la zone jouée
    """
    ax.set_facecolor("#87CEEB")  # fond bleu ciel pour les zones hors bassin

    # Zone de jeu (rectangle bleu)
    zone_jeu = patches.Rectangle(
        (0, POOL_Y_PLAY[0]),
        POOL_X_MAX,
        POOL_Y_PLAY[1] - POOL_Y_PLAY[0],
        linewidth=2, edgecolor="white", facecolor="#1565C0", alpha=0.65,
    )
    ax.add_patch(zone_jeu)

    # Lignes de zones officielles (2 m, 6 m, 19 m, 23 m)
    for x in ZONE_LINES:
        couleur_ligne = "red" if x in [2, 23] else "yellow"
        ax.axvline(x=x, color=couleur_ligne, linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(x, POOL_Y_PLAY[1] + 0.4, f"{x}m",
                ha="center", va="bottom", color=couleur_ligne, fontsize=7)

    # Buts (petits rectangles blancs)
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


def executer_pipeline(df_brut, qualite_min=30, vitesse_max=1.3, fenetre=5):
    """
    Applique les 7 étapes du pipeline de nettoyage sur un DataFrame brut.

    Paramètres :
        df_brut     : DataFrame issu d'un fichier CSV (colonnes : time, nodeID,
                      positionX, positionY, positionZ, quality)
        qualite_min : score qualité minimum à conserver (0–100)
        vitesse_max : vitesse maximale tolérée entre deux points consécutifs (m/s)
        fenetre     : nombre de points pour le filtre médian glissant

    Retourne :
        df_clean : DataFrame nettoyé avec colonnes calculées (vitesse, zones…)
        logs     : liste de chaînes décrivant chaque étape
    """
    cfg = Config(min_quality=qualite_min, max_speed=vitesse_max, median_window=fenetre)
    logs = []
    df = df_brut.copy()

    # Conversion de la colonne "time" en datetime si elle est encore en texte
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # Tri obligatoire avant de traiter les séquences par joueur
    df = df.sort_values(["nodeID", "time"]).reset_index(drop=True)

    # Les fonctions du pipeline écrivent des prints : on les redirige pour ne
    # pas encombrer le terminal, et on récupère l'info via les compteurs.
    sink = io.StringIO()

    def log_etape(nom, avant, apres, detail=""):
        supprimes = avant - apres
        logs.append(f"✅ {nom} : {supprimes} lignes supprimées → {apres} restantes. {detail}")

    with redirect_stdout(sink):

        # Étape 1 — suppression des NaN
        avant = len(df)
        df, _ = step_drop_nan(df)
        log_etape("Suppression des NaN", avant, len(df))

        # Étape 2 — filtre qualité du signal UWB
        avant = len(df)
        df, _ = step_filter_quality(df, cfg)
        log_etape(f"Filtre qualité (≥ {qualite_min})", avant, len(df))

        # Étape 3 — filtre géographique (positions hors bassin)
        avant = len(df)
        df, _ = step_filter_bounds(df, cfg)
        log_etape("Filtre géographique (hors bassin)", avant, len(df))

        # Étape 4 — suppression des téléportations (sauts impossibles)
        avant = len(df)
        df, _ = step_remove_teleportations(df, cfg)
        log_etape(f"Anti-téléportation (vitesse max {vitesse_max} m/s)", avant, len(df))

        # Étape 5 — lissage médian glissant (réduit le bruit)
        df, _ = step_smooth_median(df, cfg)
        logs.append(f"✅ Lissage médian (fenêtre = {fenetre} points)")

        # Étape 6 — rééchantillonnage à 10 Hz + interpolation des lacunes
        avant = len(df)
        df, _ = step_resample_interpolate(df, cfg)
        logs.append(f"✅ Rééchantillonnage à 10 Hz : {len(df)} lignes finales")

        # Étape 7 — calcul des colonnes dérivées (vitesse, zones, périodes…)
        df, _ = step_add_derived_columns(df, cfg, session_name="import")
        logs.append("✅ Colonnes calculées : vitesse, distance, zones, périodes")

    return df, logs


def nom_joueur(node_id):
    """
    Retourne l'étiquette affichée pour un joueur.
    Si un numéro de bonnet est défini dans le mapping, retourne "Bonnet X".
    Sinon retourne le nodeID brut.
    """
    info = st.session_state.mapping_joueurs.get(str(node_id), {})
    bonnet = info.get("bonnet", "")
    return f"Bonnet {bonnet}" if bonnet else str(node_id)


def couleur_joueur(node_ids_tries, node_id):
    """Retourne une couleur fixe pour un joueur selon son rang dans la liste."""
    try:
        idx = list(node_ids_tries).index(str(node_id))
    except ValueError:
        idx = 0
    return COULEURS[idx % len(COULEURS)]


def formater_temps(secondes):
    """Convertit des secondes en chaîne 'mm:ss'."""
    s = int(secondes)
    return f"{s // 60}:{s % 60:02d}"


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


# ==============================================================================
# PAGE 1 — CONFIGURATION & DONNÉES
# ==============================================================================
if st.session_state.page == "Configuration & Données":

    st.title("Configuration & Données")

    # ------------------------------------------------------------------
    # Section : Importation des données
    # ------------------------------------------------------------------
    st.subheader("Importation des données")
    st.caption(
        "Importez un **CSV brut** (sortie du système UWB) "
        "ou un **XLSX nettoyé** (déjà traité par le pipeline)."
    )

    fichier = st.file_uploader(
        "Fichier à importer",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if fichier is not None:

        if fichier.name.endswith(".xlsx"):
            # Fichier déjà nettoyé : chargement direct sans pipeline
            df_charge = pd.read_excel(fichier)
            st.session_state.donnees = df_charge
            st.success(
                f"Fichier XLSX chargé : **{fichier.name}** "
                f"— {len(df_charge):,} lignes, {df_charge['nodeID'].nunique()} joueurs"
            )

        elif fichier.name.endswith(".csv"):
            df_brut = pd.read_csv(fichier)

            # Vérifier que les colonnes obligatoires sont présentes
            colonnes_requises = {"time", "nodeID", "positionX", "positionY", "positionZ", "quality"}
            colonnes_absentes = colonnes_requises - set(df_brut.columns)

            if colonnes_absentes:
                st.error(f"Colonnes manquantes dans le fichier : {colonnes_absentes}")
                st.info(f"Colonnes trouvées : {list(df_brut.columns)}")
            else:
                st.info(
                    f"Fichier CSV brut : **{fichier.name}** "
                    f"— {len(df_brut):,} lignes, {df_brut['nodeID'].nunique()} tags"
                )

                # Paramètres du pipeline exposés à l'utilisateur
                st.markdown("#### Paramètres de nettoyage")
                col_q, col_v, col_f = st.columns(3)
                qualite = col_q.slider(
                    "Qualité minimum (0–100)", 0, 100, 30,
                    help="Supprimer les mesures dont le signal UWB est trop faible.",
                )
                vitesse = col_v.slider(
                    "Vitesse max (m/s)", 0.5, 5.0, 1.3, step=0.1,
                    help="Supprimer les sauts de position impliquant une vitesse impossible.",
                )
                fenetre = col_f.slider(
                    "Fenêtre lissage (points)", 3, 11, 5, step=2,
                    help="Taille de la fenêtre du filtre médian glissant.",
                )

                if st.button("Lancer le nettoyage", type="primary"):
                    with st.spinner(
                        "Nettoyage en cours… "
                        "(peut prendre 30–60 s pour les fichiers de grande taille)"
                    ):
                        df_clean, logs = executer_pipeline(df_brut, qualite, vitesse, fenetre)

                    st.session_state.donnees = df_clean
                    st.success(f"Nettoyage terminé — {len(df_clean):,} lignes conservées")

                    with st.expander("Détail des étapes"):
                        for msg in logs:
                            st.write(msg)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section : Charger un match existant
    # ------------------------------------------------------------------
    st.subheader("Charger un match")
    st.caption(
        "Les matchs avec un fichier XLSX pré-nettoyé sont chargés automatiquement. "
        "Pour les autres, importez le CSV brut ci-dessus."
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    for col, (date, dossier) in zip([col_m1, col_m2, col_m3], DOSSIERS_MATCHS.items()):
        actif = st.session_state.match_selec == date
        with col:
            if st.button(
                date, key=f"btn_match_{date}", use_container_width=True,
                type="primary" if actif else "secondary",
            ):
                st.session_state.match_selec = date

                # Chercher un XLSX nettoyé dans le dossier du match
                if os.path.exists(dossier):
                    fichiers_xlsx = [f for f in os.listdir(dossier) if f.endswith("_clean_data.xlsx")]
                    if fichiers_xlsx:
                        chemin = os.path.join(dossier, fichiers_xlsx[0])
                        st.session_state.donnees = pd.read_excel(chemin)
                        st.rerun()

    if st.session_state.match_selec:
        st.info(f"Match sélectionné : **{st.session_state.match_selec}**")
    if st.session_state.donnees is not None:
        df = st.session_state.donnees
        st.caption(f"Données chargées : {len(df):,} lignes — {df['nodeID'].nunique()} joueurs")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section : Répartition des joueurs (Tag → Bonnet)
    # ------------------------------------------------------------------
    st.subheader("Répartition des joueurs")

    # Utiliser les nodeIDs des données chargées, ou des exemples par défaut
    if st.session_state.donnees is not None:
        node_ids = sorted(st.session_state.donnees["nodeID"].unique().astype(str))
    else:
        node_ids = [f"Tag {i}" for i in range(1, 13)]

    # Initialiser les entrées manquantes dans le mapping
    for node in node_ids:
        if node not in st.session_state.mapping_joueurs:
            st.session_state.mapping_joueurs[node] = {"bonnet": "", "equipe": "INSEP"}

    # En-têtes
    col_h1, col_h2 = st.columns(2)
    col_h1.markdown("**Tag**")
    col_h2.markdown("**Bonnet**")

    # Une ligne par joueur
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
        st.success("Composition enregistrée !")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section : Visualisation du match (vue d'ensemble)
    # ------------------------------------------------------------------
    st.subheader("Visualisation du match")

    if st.session_state.donnees is None:
        st.info("Importez un fichier ou chargez un match pour afficher la visualisation.")
    else:
        df = st.session_state.donnees
        node_ids = sorted(df["nodeID"].unique().astype(str))

        fig_vue, ax_vue = plt.subplots(figsize=(14, 7))
        dessiner_bassin(ax_vue, "Vue d'ensemble — toutes les positions du match")

        for node in node_ids:
            data_node = df[df["nodeID"].astype(str) == node]
            # Sous-échantillonnage : max 500 points par joueur pour ne pas surcharger
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

    # ------------------------------------------------------------------
    # Éléments supplémentaires dans la barre latérale (filtres + slider)
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtres")

        # Filtre par période de jeu
        if "period" in df.columns:
            periodes = ["Toutes"] + [
                str(int(p)) for p in sorted(df["period"].dropna().unique())
            ]
        else:
            periodes = ["Toutes"]
        periode_choisie = st.selectbox("Période", periodes)

        # Slider temporel
        st.markdown("**Slider temporel (période du match)**")
        temps_max = int(df["elapsed_s"].max()) if "elapsed_s" in df.columns else 120
        temps_actuel = st.slider("Temps (s)", 0, temps_max, min(60, temps_max))

    # Appliquer le filtre de période sur le DataFrame
    if periode_choisie != "Toutes" and "period" in df.columns:
        df_filtre = df[df["period"] == int(periode_choisie)]
    else:
        df_filtre = df

    # ------------------------------------------------------------------
    # Section : Visualisation du match
    # ------------------------------------------------------------------
    st.subheader("Visualisation du match")
    col_bassin, col_video = st.columns(2)

    with col_bassin:
        # Positions des joueurs dans une fenêtre de ±2 s autour du temps choisi
        if "elapsed_s" in df_filtre.columns:
            fenetre_t = df_filtre[
                (df_filtre["elapsed_s"] >= temps_actuel - 2) &
                (df_filtre["elapsed_s"] <= temps_actuel + 2)
            ]
            # Pour chaque joueur, garder uniquement la dernière position connue
            positions_joueurs = (
                fenetre_t
                .sort_values("elapsed_s")
                .groupby("nodeID")
                .last()
                .reset_index()
            )
            positions_joueurs["nodeID"] = positions_joueurs["nodeID"].astype(str)
        else:
            positions_joueurs = pd.DataFrame()

        # Dessin du bassin avec les joueurs positionnés
        fig_rep, ax_rep = plt.subplots(figsize=(6, 9))
        dessiner_bassin(ax_rep, f"t = {formater_temps(temps_actuel)}")

        for _, joueur in positions_joueurs.iterrows():
            nid = str(joueur["nodeID"])
            couleur = couleur_joueur(node_ids, nid)

            # Cercle coloré représentant le joueur
            cercle = plt.Circle(
                (joueur["positionX"], joueur["positionY"]),
                radius=0.6, color=couleur, zorder=5,
            )
            ax_rep.add_patch(cercle)

            # Numéro dans le cercle : bonnet si disponible, sinon nodeID
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

        # Résumé textuel des joueurs visibles au temps actuel
        if not positions_joueurs.empty and "speed" in positions_joueurs.columns:
            st.markdown(f"**Joueurs à t = {formater_temps(temps_actuel)} :**")
            for _, j in positions_joueurs.iterrows():
                nid = str(j["nodeID"])
                vitesse_kmh = j["speed"] * 3.6 if pd.notna(j.get("speed")) else 0
                zone = j.get("zone", "—")
                st.write(f"- **{nom_joueur(nid)}** — {vitesse_kmh:.1f} km/h — zone : {zone}")

    # ------------------------------------------------------------------
    # Section : Enregistrement d'une séquence
    # ------------------------------------------------------------------
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
                st.rerun()

            if col_non.button("Annuler"):
                st.session_state.show_form = False
                st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section : Bibliothèque d'actions
    # ------------------------------------------------------------------
    st.subheader("Bibliothèque d'actions")

    # En-têtes du tableau
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

    # ------------------------------------------------------------------
    # Filtre dans la barre latérale : Individuel / Tactique
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtres")
        mode = st.radio("Type d'analyse", ["Individuel", "Tactique"])

    # ==================================================================
    # MODE INDIVIDUEL — stats, heatmap et courbe de vitesse d'un joueur
    # ==================================================================
    if mode == "Individuel":

        st.subheader("Sélection du joueur")
        labels = {node: nom_joueur(node) for node in node_ids}
        label_choisi = st.selectbox("Joueur", list(labels.values()), label_visibility="collapsed")
        node_choisi = [k for k, v in labels.items() if v == label_choisi][0]

        df_joueur = df[df["nodeID"].astype(str) == node_choisi].copy()

        # ---- Statistiques de performance ----
        st.subheader("Performance")

        # Temps dans l'eau
        if "elapsed_s" in df_joueur.columns:
            duree_s = df_joueur["elapsed_s"].max() - df_joueur["elapsed_s"].min()
            temps_eau = formater_temps(duree_s) + " min"
        else:
            temps_eau = "N/A"

        # Distance totale
        if "distance_cumul" in df_joueur.columns:
            dist_km = df_joueur["distance_cumul"].max() / 1000
            distance = f"{dist_km:.1f} km"
        else:
            distance = "N/A"

        # Vitesse moyenne et max (conversion m/s → km/h)
        if "speed" in df_joueur.columns:
            vit_moy = df_joueur["speed"].mean() * 3.6
            vit_max = df_joueur["speed"].max() * 3.6
            vitesse_moy = f"{vit_moy:.1f} km/h"
            vitesse_max = f"{vit_max:.1f} km/h"
        else:
            vitesse_moy = vitesse_max = "N/A"

        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric("Temps dans l'eau", temps_eau)
        col_2.metric("Distance totale",  distance)
        col_3.metric("Vitesse moyenne",  vitesse_moy)
        col_4.metric("Vitesse max",      vitesse_max)

        # ---- Heatmap ----
        st.subheader("Heatmap")

        fig_heat, ax_heat = plt.subplots(figsize=(13, 6))
        dessiner_bassin(ax_heat, f"Heatmap de présence — {label_choisi}")

        x_vals = df_joueur["positionX"].dropna()
        y_vals = df_joueur["positionY"].dropna()

        if len(x_vals) > 20:
            # Grille de densité : compter les positions dans chaque cellule
            heatmap_data, _, _ = np.histogram2d(
                x_vals, y_vals,
                bins=[50, 40],
                range=[[0, 25], [3, 23]],
            )

            # Lissage gaussien pour un rendu plus doux (scipy requis)
            if SCIPY_OK:
                heatmap_data = gaussian_filter(heatmap_data, sigma=1.5)

            # Palette bleu → orange → rouge (peu de présence → beaucoup)
            palette = LinearSegmentedColormap.from_list(
                "vtp_heat", ["#2196F3", "#FFA726", "#EF5350"]
            )
            im = ax_heat.imshow(
                heatmap_data.T,
                extent=[0, 25, 3, 23],
                origin="lower", cmap=palette,
                alpha=0.75, aspect="auto",
            )
            plt.colorbar(im, ax=ax_heat, label="Densité de présence", shrink=0.7)
        else:
            ax_heat.text(12.5, 13, "Pas assez de données",
                         ha="center", color="white", fontsize=12)

        st.pyplot(fig_heat)
        plt.close()

        # ---- Courbe de vitesse ----
        st.subheader("Courbe de vitesse")

        if "speed" in df_joueur.columns and "elapsed_s" in df_joueur.columns:
            df_vit = (
                df_joueur
                .dropna(subset=["speed", "elapsed_s"])
                .sort_values("elapsed_s")
            )

            # Regrouper par tranches de 10 secondes pour lisser la courbe
            df_vit = df_vit.copy()
            df_vit["tranche_10s"] = (df_vit["elapsed_s"] // 10) * 10
            vitesse_lissee = (
                df_vit.groupby("tranche_10s")["speed"]
                .mean()
                .reset_index()
            )
            vitesse_lissee["km_h"]    = vitesse_lissee["speed"] * 3.6
            vitesse_lissee["minutes"] = vitesse_lissee["tranche_10s"] / 60

            fig_vit, ax_vit = plt.subplots(figsize=(13, 4))
            ax_vit.plot(
                vitesse_lissee["minutes"], vitesse_lissee["km_h"],
                color="#1565C0", linewidth=2, marker="o", markersize=3,
            )
            ax_vit.fill_between(
                vitesse_lissee["minutes"], vitesse_lissee["km_h"],
                alpha=0.12, color="#1565C0",
            )
            ax_vit.set_xlabel("Temps (min)")
            ax_vit.set_ylabel("Vitesse (km/h)")
            ax_vit.grid(axis="y", linestyle="--", alpha=0.4)
            ax_vit.set_facecolor("#F8F9FA")
            st.pyplot(fig_vit)
            plt.close()
        else:
            st.info("Colonne 'speed' non disponible (nettoyage requis).")

    # ==================================================================
    # MODE TACTIQUE — zones et trajectoires de tous les joueurs
    # ==================================================================
    elif mode == "Tactique":

        st.subheader("Vue tactique — Répartition par zones")

        if "zone" not in df.columns:
            st.info("Les données de zone ne sont pas disponibles (nettoyage requis).")
        else:
            # Calcul du pourcentage de temps par zone pour chaque joueur
            zones_ordre = ["2m_gauche", "ad_gauche", "transition", "ad_droite", "2m_droite"]
            counts = df.groupby(["nodeID", "zone"]).size().unstack(fill_value=0)
            zones_dispo = [z for z in zones_ordre if z in counts.columns]
            counts = counts[zones_dispo]
            pourcentages = counts.div(counts.sum(axis=1), axis=0) * 100

            # Remplacer les nodeIDs par les noms de joueurs pour l'affichage
            pourcentages.index = [nom_joueur(str(n)) for n in pourcentages.index]

            # Couleurs cohérentes avec les lignes du terrain
            couleurs_zones = {
                "2m_gauche":  "#E74C3C",
                "ad_gauche":  "#F39C12",
                "transition": "#3498DB",
                "ad_droite":  "#F39C12",
                "2m_droite":  "#E74C3C",
            }
            couleurs = [couleurs_zones.get(z, "#95A5A6") for z in zones_dispo]

            fig_zones, ax_zones = plt.subplots(figsize=(13, 5))
            pourcentages.plot(
                kind="bar", ax=ax_zones, stacked=True,
                color=couleurs, edgecolor="white", linewidth=0.5,
            )
            ax_zones.set_xlabel("Joueur", fontsize=10)
            ax_zones.set_ylabel("% du temps", fontsize=10)
            ax_zones.set_title("Temps passé dans chaque zone — tous les joueurs", fontsize=11)
            ax_zones.legend(title="Zone", loc="upper right", fontsize=8)
            ax_zones.tick_params(axis="x", rotation=45)
            ax_zones.set_facecolor("#F8F9FA")
            st.pyplot(fig_zones)
            plt.close()

        # Trajectoires de tous les joueurs superposées sur le bassin
        st.subheader("Trajectoires sur le bassin")

        fig_traj, ax_traj = plt.subplots(figsize=(13, 7))
        dessiner_bassin(ax_traj, "Trajectoires de tous les joueurs")

        for node in node_ids:
            data_node = (
                df[df["nodeID"].astype(str) == node]
                .sort_values("elapsed_s")
            )
            # 1 point sur 5 pour alléger la figure sans perdre la forme des trajets
            data_node = data_node.iloc[::5]
            couleur = couleur_joueur(node_ids, node)
            ax_traj.plot(
                data_node["positionX"], data_node["positionY"],
                color=couleur, alpha=0.45, linewidth=0.8,
                label=nom_joueur(node),
            )

        ax_traj.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
        st.pyplot(fig_traj)
        plt.close()
