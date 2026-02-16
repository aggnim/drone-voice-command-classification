# VoiceStick — Guide utilisateur

> **ENGLISH VERSION BELOW**

Package d'inférence pour prédire des commandes de drone à partir de fichiers audio. Les modèles SVM et MLP sont pré-entraînés et prêts à l'emploi.

## Démarrage rapide

1. Placez vos fichiers `.wav` dans le dossier `data/`
2. Lancez la prédiction :
   ```bash
   python predict.py
   ```
3. Retrouvez les résultats dans `output/predictions/`

## Structure du dossier

```
user_version/
├── predict.py           # Script de prédiction
├── config.yaml          # Configuration (chemins)
├── requirements.txt     # Dépendances Python
├── models/
│   ├── SVM_model/       # Modèle SVM pré-entraîné
│   │   ├── model_svm.pkl
│   │   ├── scaler.pkl
│   │   └── label_encoder.pkl
│   └── MLP_model/       # Modèle MLP pré-entraîné
│       ├── model_mlp.pkl
│       ├── scaler.pkl
│       └── label_encoder.pkl
├── data/                # Déposez vos fichiers .wav ici
└── output/
    └── predictions/     # Résultats générés ici
```

## Configuration (config.yaml)

```yaml
paths:
  models_dir: "models"    # Dossier des modèles pré-entraînés
  data_dir: "data"        # Dossier des fichiers audio .wav
  output_dir: "output"    # Dossier de sortie des prédictions
```

Les chemins sont relatifs au dossier `user_version/`. Ils peuvent être remplacés par des chemins absolus si nécessaire.

## Comment ça marche

`predict.py` utilise la **VAD** (Voice Activity Detection, ou détection d'activité vocale) pour trouver automatiquement les moments où quelqu'un parle dans l'audio. Concrètement, l'algorithme mesure le niveau sonore du signal et le compare à un seuil de silence : tout ce qui dépasse le seuil est considéré comme de la parole, le reste comme du silence ou du bruit de fond.

Chaque segment de parole détecté est ensuite converti en embedding wav2vec2, puis envoyé au classifieur (SVM et/ou MLP) qui identifie la commande prononcée.

## Utilisation de predict.py

### Syntaxe

```bash
python predict.py [config.yaml] [options]
```

### Options

| Option | Par défaut | Description |
|---|---|---|
| `--model {svm,mlp,both}` | `both` | Modèle(s) à utiliser |
| `--audio CHEMIN` | `data/` | Fichier .wav ou dossier de fichiers .wav |
| `--top-db FLOAT` | `30.0` | Seuil de silence VAD en dB sous le pic (voir section VAD) |
| `--min-dur FLOAT` | `0.2` | Durée minimale d'un segment détecté (secondes) |
| `--max-dur FLOAT` | `2.0` | Durée maximale d'un segment détecté (secondes) |

### Exemples

```bash
# Prédiction sur tous les fichiers dans data/, les deux modèles
python predict.py

# Un seul fichier audio
python predict.py --audio data/enregistrement.wav

# MLP uniquement
python predict.py --model mlp

# Fichiers dans un autre dossier
python predict.py --audio /chemin/vers/mes/audios/

# Ajuster la sensibilité du VAD
python predict.py --top-db 25 --min-dur 0.3 --max-dur 1.5
```

## Format de sortie (JSON)

Les résultats sont enregistrés au format JSON dans `output/predictions/commands_{svm,mlp}.json`. Les prédictions `none` (silence/bruit) sont exclues. Les commandes sont groupées par fichier audio et triées par temps de début.

```json
{
  "enregistrement_001.wav": [
    {"start": 0.52, "end": 1.14, "command": "forward"},
    {"start": 2.30, "end": 2.98, "command": "left"},
    {"start": 4.10, "end": 4.75, "command": "up"}
  ],
  "enregistrement_002.wav": [
    {"start": 0.48, "end": 1.22, "command": "right"},
    {"start": 2.55, "end": 3.10, "command": "down"}
  ]
}
```

## Réglage du VAD

La VAD (Voice Activity Detection) détecte automatiquement les segments de parole en mesurant le niveau d'énergie sonore de l'audio. Le paramètre principal est `--top-db` : il définit le seuil de silence en décibels sous le pic le plus fort de l'enregistrement. Par exemple, avec `--top-db 30`, tout ce qui est à plus de 30 dB en dessous du pic est considéré comme du silence.

Les paramètres `--min-dur` et `--max-dur` filtrent ensuite les segments détectés par durée : ceux trop courts (bruits parasites) ou trop longs (plusieurs commandes collées) sont écartés.

Si les résultats ne sont pas satisfaisants :

| Problème | Solution |
|---|---|
| Trop de segments détectés (bruit de fond) | Baisser `--top-db` (ex: 25 ou 20) |
| Des commandes ne sont pas détectées | Augmenter `--top-db` (ex: 35 ou 40) |
| Segments trop courts (mots coupés) | Baisser `--min-dur` (ex: 0.1) |
| Segments trop longs (plusieurs commandes fusionnées) | Baisser `--max-dur` (ex: 1.5) |

## Installation

```bash
pip install -r requirements.txt
```

Nécessite Python 3.10+. Le premier lancement télécharge automatiquement le modèle wav2vec2 (~3 Go).

---

# English Version

# VoiceStick — User Guide

Inference package for predicting drone commands from audio files. The SVM and MLP models are pre-trained and ready to use.

## Quick start

1. Place your `.wav` files in the `data/` folder
2. Run prediction:
   ```bash
   python predict.py
   ```
3. Find results in `output/predictions/`

## Folder structure

```
user_version/
├── predict.py           # Prediction script
├── config.yaml          # Configuration (paths)
├── requirements.txt     # Python dependencies
├── models/
│   ├── SVM_model/       # Pre-trained SVM model
│   │   ├── model_svm.pkl
│   │   ├── scaler.pkl
│   │   └── label_encoder.pkl
│   └── MLP_model/       # Pre-trained MLP model
│       ├── model_mlp.pkl
│       ├── scaler.pkl
│       └── label_encoder.pkl
├── data/                # Place your .wav files here
└── output/
    └── predictions/     # Generated results go here
```

## Configuration (config.yaml)

```yaml
paths:
  models_dir: "models"    # Pre-trained model directory
  data_dir: "data"        # Audio .wav file directory
  output_dir: "output"    # Prediction output directory
```

Paths are relative to the `user_version/` folder. They can be replaced with absolute paths if needed.

## How it works

`predict.py` uses **VAD** (Voice Activity Detection) to automatically find the moments where someone is speaking in the audio. The algorithm measures the signal's loudness and compares it to a silence threshold: anything above the threshold is considered speech, the rest is treated as silence or background noise.

Each detected speech segment is then converted into a wav2vec2 embedding and sent to the classifier (SVM and/or MLP) which identifies the spoken command.

## Using predict.py

### Syntax

```bash
python predict.py [config.yaml] [options]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--model {svm,mlp,both}` | `both` | Model(s) to use |
| `--audio PATH` | `data/` | Single .wav file or directory of .wav files |
| `--top-db FLOAT` | `30.0` | VAD silence threshold in dB below peak (see VAD section) |
| `--min-dur FLOAT` | `0.2` | Minimum detected segment duration (seconds) |
| `--max-dur FLOAT` | `2.0` | Maximum detected segment duration (seconds) |

### Examples

```bash
# Predict on all files in data/, both models
python predict.py

# Single audio file
python predict.py --audio data/recording.wav

# MLP only
python predict.py --model mlp

# Files from another directory
python predict.py --audio /path/to/my/audio/

# Adjust VAD sensitivity
python predict.py --top-db 25 --min-dur 0.3 --max-dur 1.5
```

## Output format (JSON)

Results are saved as JSON in `output/predictions/commands_{svm,mlp}.json`. Predictions of `none` (silence/noise) are excluded. Commands are grouped by audio file and sorted by start time.

```json
{
  "recording_001.wav": [
    {"start": 0.52, "end": 1.14, "command": "forward"},
    {"start": 2.30, "end": 2.98, "command": "left"},
    {"start": 4.10, "end": 4.75, "command": "up"}
  ],
  "recording_002.wav": [
    {"start": 0.48, "end": 1.22, "command": "right"},
    {"start": 2.55, "end": 3.10, "command": "down"}
  ]
}
```

## VAD tuning

VAD (Voice Activity Detection) automatically detects speech segments by measuring the audio's energy level. The main parameter is `--top-db`: it defines the silence threshold in decibels below the loudest peak in the recording. For example, with `--top-db 30`, anything more than 30 dB below the peak is considered silence.

The `--min-dur` and `--max-dur` parameters then filter detected segments by duration: segments that are too short (noise artifacts) or too long (multiple commands merged together) are discarded.

If results are unsatisfactory:

| Problem | Solution |
|---|---|
| Too many segments detected (background noise) | Lower `--top-db` (e.g., 25 or 20) |
| Commands are not being detected | Raise `--top-db` (e.g., 35 or 40) |
| Segments too short (words cut off) | Lower `--min-dur` (e.g., 0.1) |
| Segments too long (multiple commands merged) | Lower `--max-dur` (e.g., 1.5) |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. The first run automatically downloads the wav2vec2 model (~3 GB).
