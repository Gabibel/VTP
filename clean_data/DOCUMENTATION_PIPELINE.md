# Documentation du pipeline UWB — Water-Polo
## Projet EFREI Research Lab × Fédération Française de Natation

---

## Orientation du terrain

Conformément au cahier des charges (Figure 2) :

- **Axe X (0 – 25 m)** : longueur du terrain — les **buts sont aux extrémités X**
- **Axe Y (0 – 26 m)** : largeur du terrain + marges de 3 m de chaque côté (terrain joué : Y = 3 à 23 m)
- **But gauche** : X ≈ 0 m, centré en Y ≈ 13.25 m
- **But droit** : X ≈ 25 m, centré en Y ≈ 13.25 m

---

## Fichier d'entrée attendu

Colonnes obligatoires dans le CSV brut :

| Colonne | Description |
|---|---|
| `time` | Horodatage de la mesure |
| `nodeID` | Identifiant hexadécimal du tag UWB (ex: `1bb3`) |
| `positionX` | Position en mètres sur la **longueur** du bassin (axe des buts) |
| `positionY` | Position en mètres sur la **largeur** du bassin |
| `positionZ` | Position en mètres en hauteur (≈ 0–1 m pour un joueur en surface) |
| `quality` | Score de qualité du signal UWB (0–100) |

---

## Étapes du pipeline de nettoyage

### Étape 1 — Suppression des NaN *(toujours active)*

Supprime les lignes où X, Y ou Z est vide. Ces trous apparaissent quand le système
n'a pas réussi à calculer une position pour une mesure donnée.

---

### Étape 2 — Filtre qualité

**Paramètre :** `min_quality` (défaut : 30)

Supprime les mesures dont le score de qualité est inférieur au seuil. Un score bas
indique un signal UWB dégradé (obstacle, angle de vue insuffisant). Ces positions
sont peu fiables et introduisent du bruit.

---

### Étape 3 — Filtre géographique (hors bassin)

**Paramètres :** `pool_x_min/max`, `pool_y_min/max`, `pool_z_min/max`

Supprime les positions en dehors des limites physiques du bassin (avec une marge).
Ces outliers sont des artefacts du système de localisation.

Valeurs par défaut :
- X : [-0.5 m, 25.5 m]
- Y : [-0.5 m, 27.0 m]
- Z : [-3.0 m, 3.0 m]

---

### Étape 4 — Suppression des téléportations

**Paramètres :** `max_speed` (défaut : **1.3 m/s**), `teleport_passes` (défaut : 3)

Entre deux mesures consécutives du même joueur, si la vitesse impliquée dépasse
1.3 m/s, la seconde position est supprimée. L'opération est répétée jusqu'à 3 fois
pour éliminer les chaînes de points aberrants.

> **Note :** Ce filtre s'applique sur les données brutes, *avant* le rééchantillonnage.
> Après interpolation (étape 6), la vitesse entre deux points peut légèrement dépasser
> 1.3 m/s sur des zones interpolées — c'est normal et attendu.

---

### Étape 5 — Lissage médian glissant

**Paramètre :** `median_window` (défaut : 5 points)

Applique un filtre médian centré sur une fenêtre de 5 points consécutifs pour X, Y, Z,
par joueur. Atténue les pics de bruit sans supprimer de données. Le médian est préféré
à la moyenne car il est insensible aux valeurs extrêmes.

---

### Étape 6 — Rééchantillonnage à 10 Hz + interpolation

**Paramètres :** `resample_freq_ms` (défaut : 100 ms), `max_interp_gap` (défaut : 2 s)

1. **Moyenne** des positions dans chaque fenêtre de 100 ms (si plusieurs mesures tombent
   dans la même fenêtre, on prend leur moyenne)
2. **Interpolation linéaire** des instants manquants
3. **Les trous > 2 secondes** ne sont pas interpolés et sont supprimés

Résultat : une position par joueur toutes les 100 ms exactement (**10 Hz**).

La colonne `quality` est **supprimée** à cette étape : sa valeur n'est plus
significative sur des points synthétiques interpolés.

---

### Étape 7 — Calcul des colonnes dérivées

Ajout de toutes les colonnes d'analyse à partir des positions nettoyées.

---

## Colonnes du fichier de sortie

### Identification

| Colonne | Type | Description |
|---|---|---|
| `time` | datetime | Horodatage à 10 Hz |
| `nodeID` | string | Identifiant hexadécimal du tag UWB |
| `session` | string | Nom du fichier source (ex: `t1`, `t2`…) |

---

### Position

| Colonne | Unité | Plage typique | Description |
|---|---|---|---|
| `positionX` | m | 0 – 25 | Position sur la **longueur** (axe des buts) |
| `positionY` | m | 3 – 23 | Position sur la **largeur** (zone de jeu) |
| `positionZ` | m | 0 – 1.5 | Hauteur (tag sous le bonnet) |

---

### Cinématiques *(calculées entre deux points à 100 ms d'écart)*

| Colonne | Unité | Plage typique | Description |
|---|---|---|---|
| `distance_step` | m | 0 – 0.13 | Distance parcourue depuis le point précédent. Max théorique : 1.3 m/s × 0.1 s = **0.13 m** |
| `distance_cumul` | m | 0 – ~2000 | Distance totale cumulée depuis le début de la séance |
| `speed` | m/s | 0 – ~1.4 | Vitesse instantanée |
| `acceleration` | m/s² | -15 – 15 | Variation de vitesse. **Positive = accélération, négative = décélération/freinage** |
| `heading` | ° | -180 – 180 | Direction du déplacement. 0° = vers le but droit, ±180° = vers le but gauche |

> La **première ligne de chaque joueur** a `distance_step`, `speed`, `acceleration` et `heading` à `NaN`
> (pas de point précédent pour calculer la différence).

---

### Zones officielles water-polo *(axe X)*

| Colonne | Valeurs possibles | Description |
|---|---|---|
| `zone` | voir tableau ci-dessous | Zone du terrain selon la position X |

| Valeur | X (m) | Correspondance officielle |
|---|---|---|
| `2m_gauche` | 0 – 2 | Zone des 2 m côté but gauche |
| `ad_gauche` | 2 – 6 | Zone attaque/défense côté gauche |
| `transition` | 6 – 19 | Zone de transition (13 m au centre) |
| `ad_droite` | 19 – 23 | Zone attaque/défense côté droit |
| `2m_droite` | 23 – 25 | Zone des 2 m côté but droit |

> `ad_gauche` / `ad_droite` = attaque ou défense selon l'équipe — à croiser avec la colonne `equipe`.

---

### Distances aux buts

| Colonne | Unité | Description |
|---|---|---|
| `dist_but_gauche` | m | Distance euclidienne au but gauche (X=0, Y=13.25) |
| `dist_but_droit` | m | Distance euclidienne au but droit (X=25, Y=13.25) |

---

### Temporelles

| Colonne | Type | Description |
|---|---|---|
| `date` | date | Date de la séance |
| `elapsed_s` | s | Temps écoulé depuis le début du fichier |
| `period` | int | Numéro de la période de jeu (voir tableau ci-dessous) |

| Date du match | Périodes | Durée |
|---|---|---|
| 13/01/2026 | 3 | 20 min chacune |
| 20/01/2026 | 4 | 12 min chacune |
| 03/02/2026 | 4 | 12 min chacunes |

---

### Joueurs *(optionnel — nécessite `--players players.json`)*

| Colonne | Type | Description |
|---|---|---|
| `bonnet` | int | Numéro de bonnet du joueur |
| `equipe` | string | Équipe (`INSEP` ou `invite`) |

> Sans le fichier `--players`, ces deux colonnes sont absentes mais le pipeline tourne normalement.

---

## Usage

```bash
# Basique (sans mapping joueurs)
python traitement_uwb.py --input t1.csv --output t1_clean.xlsx

# Avec mapping joueurs
python traitement_uwb.py --input t1.csv --output t1_clean.xlsx --players players.json

# Plusieurs fichiers d'un coup
python traitement_uwb.py --input t1.csv t2.csv t3.csv --output-dir ./clean/

# Avec paramètres personnalisés
python traitement_uwb.py --input t1.csv --output t1_clean.xlsx --quality 40 --speed 1.3

# Générer un fichier de config réutilisable
python traitement_uwb.py --generate-config config.json
```

---

## Paramètres configurables

| Paramètre CLI | Défaut | Description |
|---|---|---|
| `--quality` | 30 | Score qualité minimum (0–100) |
| `--speed` | 1.3 | Vitesse max pour filtre téléportation (m/s) |
| `--median-window` | 5 | Fenêtre du filtre médian (points) |
| `--freq` | 100 | Pas de rééchantillonnage (ms) |
| `--max-gap` | 2.0 | Trou max interpolé (s) |
| `--pool-x MIN MAX` | -0.5 / 25.5 | Limites X du bassin (m) |
| `--pool-y MIN MAX` | -0.5 / 27.0 | Limites Y du bassin (m) |
| `--pool-z MIN MAX` | -3.0 / 3.0 | Limites Z du bassin (m) |
| `--no-quality` | — | Désactiver le filtre qualité |
| `--no-bounds` | — | Désactiver le filtre géographique |
| `--no-teleport` | — | Désactiver le filtre téléportation |
| `--no-smooth` | — | Désactiver le lissage médian |
| `--no-interp` | — | Désactiver le rééchantillonnage |
| `--players` | — | Fichier JSON de mapping nodeID → joueur/équipe |
| `--config` | — | Fichier JSON de configuration complète |
