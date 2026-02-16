# VoiceStick — Guide développeur

> **ENGLISH VERSION BELOW**

Pipeline complet pour entraîner des classifieurs de commandes de drone à partir d'enregistrements audio annotés. Comprend la préparation des données, l'entraînement (SVM et MLP), la prédiction et l'évaluation.

## Démarrage rapide

1. Placez vos fichiers `.TextGrid` dans `data/textgrid/` et les fichiers `.wav` correspondants dans `data/audio/`
2. Préparez les données :
   ```bash
   python prepare_data.py
   ```
3. Entraînez les modèles :
   ```bash
   python train_svm.py
   python train_mlp.py
   ```
4. Prédisez sur l'audio de test :
   ```bash
   python predict.py --use-ground-truth
   ```
5. Évaluez les prédictions :
   ```bash
   python evaluate.py
   ```

## Structure du dossier

### Avant exécution

```
dev_version/
├── prepare_data.py      # Préparation des données
├── train_svm.py         # Entraînement SVM
├── train_mlp.py         # Entraînement MLP
├── predict.py           # Prédiction sur audio brut
├── evaluate.py          # Évaluation des prédictions
├── config.yaml          # Configuration
├── requirements.txt     # Dépendances Python
├── data/
│   ├── textgrid/        # Fichiers .TextGrid (annotations Praat)
│   └── audio/           # Fichiers .wav (enregistrements)
└── output/              # Vide au départ
```

### Après exécution complète

```
output/
├── dataset.csv                  # Dataset complet (tous les participants)
├── train/
│   ├── dataset_train.csv        # Données d'entraînement
│   ├── audio_segments/          # Segments audio 16 kHz
│   └── all_embeddings.npz       # Embeddings wav2vec2
├── test/
│   ├── dataset_test.csv         # Labels ground truth (test)
│   └── audio/                   # Fichiers audio bruts (test)
├── SVM_model/
│   ├── model_svm.pkl            # Modèle entraîné
│   ├── scaler.pkl               # StandardScaler
│   ├── label_encoder.pkl        # Encodeur de labels
│   ├── cv_results_svm.json      # Résultats cross-validation
│   ├── confusion_matrix_svm.png # Matrice de confusion
│   └── svm_output.txt           # Log d'entraînement
├── MLP_model/
│   ├── model_mlp.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── cv_results_mlp.json
│   ├── confusion_matrix_mlp.png
│   └── mlp_output.txt
├── predictions/
│   ├── predictions_svm.csv      # Prédictions SVM
│   └── predictions_mlp.csv      # Prédictions MLP
└── evaluation/
    ├── eval_svm.json            # Métriques SVM
    ├── eval_mlp.json            # Métriques MLP
    ├── comparison.json          # Comparaison côte à côte
    ├── confusion_matrix_eval_svm.png
    ├── confusion_matrix_eval_mlp.png
    └── evaluation_output.txt
```

## Format des données

### Fichiers audio

- Format : `.wav`
- Fréquence d'échantillonnage : quelconque (réenregistré à 16 kHz en interne)
- Convention de nommage : `DD_MM_YY_HH_MM_SS_NNN.wav`
  - `DD_MM_YY_HH_MM_SS` : identifiant du participant (date/heure de la session)
  - `NNN` : numéro de tentative (`000` à `005`)

### Fichiers TextGrid (Praat)

- Encodage : UTF-8 ou UTF-16 avec BOM (détection automatique)
- Un tier nommé `commands` (insensible à la casse)
- Labels : les 8 commandes directionnelles + `none`

```
forward | backward | left | right | up | down | yawleft | yawright | none
```

#### Comment annoter

1. Ouvrez le fichier `.wav` dans Praat et créez un objet TextGrid associé
2. Ajoutez un tier de type **interval** nommé `commands`
3. Pour chaque passage où le locuteur prononce une commande directionnelle (`forward`, `backward`, `left`, `right`, `up`, `down`, `yawleft`, `yawright`), créez un segment couvrant la durée de la commande et inscrivez le label correspondant
4. Pour les passages où le locuteur parle sans prononcer de commande directionnelle (parole non-pertinente, hésitations, commentaires, etc.), créez un segment et annotez-le `none`
5. Les silences (aucune activité vocale) sont laissés sans annotation — ne créez pas de segment pour les parties silencieuses

En résumé : toute parole doit être annotée (commande ou `none`), seuls les silences restent non annotés. Les intervalles vides sont automatiquement traités comme `none` par le pipeline.

## Configuration (config.yaml)

```yaml
paths:
  textgrid_dir: "data/textgrid"    # Annotations Praat (.TextGrid)
  audio_dir: "data/audio"          # Enregistrements audio (.wav)
  output_dir: "output"             # Sorties du pipeline

data_preparation:
  tier_name: "commands"    # Nom du tier TextGrid
  skip_if_cached: true     # Réutiliser les fichiers intermédiaires existants
  test_size: 0.15          # Proportion de participants réservés pour le test
  random_seed: 42          # Graine pour la reproductibilité du split

training:
  balance_classes: true    # Sous-échantillonner la classe "none"
  none_ratio: 1.0          # Ratio max de "none" vs la plus grande classe de commande
  n_folds: 5               # Nombre de folds pour la cross-validation (GroupKFold)
```

## Étapes du pipeline

### 1. Préparation des données — `python prepare_data.py`

Cinq étapes exécutées séquentiellement :

1. **Parsing** des TextGrids : extrait les annotations du tier `commands` vers `dataset.csv`
2. **Split** des participants 85/15 en ensembles train/test (par participant, pas par segment)
3. **Segmentation** de l'audio d'entraînement en clips WAV 16 kHz dans `train/audio_segments/`
4. **Extraction d'embeddings** wav2vec2-FR-7K-large (mean-pooling) vers `train/all_embeddings.npz`
5. **Copie** des fichiers audio bruts des participants de test vers `test/audio/`

L'option `skip_if_cached: true` permet de reprendre une exécution interrompue sans recalculer les étapes déjà complétées.

### 2. Entraînement SVM — `python train_svm.py`

- Lit les données depuis `output/train/`
- Cross-validation GroupKFold à 5 folds (groupement par participant)
- Par fold : équilibrage des classes, StandardScaler, SVC(kernel='rbf', C=10, class_weight='balanced')
- Affiche les métriques par fold et la moyenne
- Entraîne le modèle final sur toutes les données d'entraînement
- Sauvegarde dans `output/SVM_model/`

### 3. Entraînement MLP — `python train_mlp.py`

Même structure que le SVM :

- Cross-validation GroupKFold à 5 folds
- Par fold : équilibrage des classes, StandardScaler, MLPClassifier
- Sauvegarde dans `output/MLP_model/`

### 4. Prédiction — `python predict.py --use-ground-truth`

En mode `--use-ground-truth` (recommandé pour l'évaluation) :

- Utilise les bornes temporelles de `dataset_test.csv`
- Extrait les embeddings wav2vec2 des segments de test
- Classifie avec SVM et/ou MLP
- Sauvegarde les prédictions CSV dans `output/predictions/`

Autres modes disponibles :

```bash
python predict.py                          # VAD sur test/audio/, les deux modèles
python predict.py --model svm              # SVM uniquement
python predict.py --audio path/to/file.wav # N'importe quel fichier audio
```

### 5. Évaluation — `python evaluate.py`

- Compare les prédictions (`predictions_svm.csv`, `predictions_mlp.csv`) aux labels ground truth
- Par modèle : accuracy, F1-macro, F1-weighted, rapport par classe, matrice de confusion
- Tableau de comparaison côte à côte si les deux modèles sont disponibles
- Sauvegarde dans `output/evaluation/`

## Installation

```bash
pip install -r requirements.txt
```

Nécessite Python 3.10+. Le premier lancement de `prepare_data.py` télécharge automatiquement le modèle wav2vec2 (~3 Go).

---

# English Version

# VoiceStick — Developer Guide

Full pipeline for training drone command classifiers from annotated audio recordings. Includes data preparation, training (SVM and MLP), prediction, and evaluation.

## Quick start

1. Place your `.TextGrid` files in `data/textgrid/` and matching `.wav` files in `data/audio/`
2. Prepare the data:
   ```bash
   python prepare_data.py
   ```
3. Train the models:
   ```bash
   python train_svm.py
   python train_mlp.py
   ```
4. Predict on test audio:
   ```bash
   python predict.py --use-ground-truth
   ```
5. Evaluate predictions:
   ```bash
   python evaluate.py
   ```

## Folder structure

### Before running

```
dev_version/
├── prepare_data.py      # Data preparation
├── train_svm.py         # SVM training
├── train_mlp.py         # MLP training
├── predict.py           # Prediction on raw audio
├── evaluate.py          # Prediction evaluation
├── config.yaml          # Configuration
├── requirements.txt     # Python dependencies
├── data/
│   ├── textgrid/        # .TextGrid files (Praat annotations)
│   └── audio/           # .wav files (recordings)
└── output/              # Empty initially
```

### After full pipeline run

```
output/
├── dataset.csv                  # Full dataset (all participants)
├── train/
│   ├── dataset_train.csv        # Training data
│   ├── audio_segments/          # 16 kHz audio segments
│   └── all_embeddings.npz       # wav2vec2 embeddings
├── test/
│   ├── dataset_test.csv         # Ground truth labels (test)
│   └── audio/                   # Raw audio files (test)
├── SVM_model/
│   ├── model_svm.pkl            # Trained model
│   ├── scaler.pkl               # StandardScaler
│   ├── label_encoder.pkl        # Label encoder
│   ├── cv_results_svm.json      # Cross-validation results
│   ├── confusion_matrix_svm.png # Confusion matrix
│   └── svm_output.txt           # Training log
├── MLP_model/
│   ├── model_mlp.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── cv_results_mlp.json
│   ├── confusion_matrix_mlp.png
│   └── mlp_output.txt
├── predictions/
│   ├── predictions_svm.csv      # SVM predictions
│   └── predictions_mlp.csv      # MLP predictions
└── evaluation/
    ├── eval_svm.json            # SVM metrics
    ├── eval_mlp.json            # MLP metrics
    ├── comparison.json          # Side-by-side comparison
    ├── confusion_matrix_eval_svm.png
    ├── confusion_matrix_eval_mlp.png
    └── evaluation_output.txt
```

## Data format

### Audio files

- Format: `.wav`
- Sample rate: any (resampled to 16 kHz internally)
- Naming convention: `DD_MM_YY_HH_MM_SS_NNN.wav`
  - `DD_MM_YY_HH_MM_SS`: participant identifier (session date/time)
  - `NNN`: attempt number (`000` to `005`)

### TextGrid files (Praat)

- Encoding: UTF-8 or UTF-16 with BOM (auto-detected)
- A tier named `commands` (case-insensitive)
- Labels: the 8 directional commands + `none`

```
forward | backward | left | right | up | down | yawleft | yawright | none
```

#### How to annotate

1. Open the `.wav` file in Praat and create an associated TextGrid object
2. Add an **interval** tier named `commands`
3. For each passage where the speaker utters a directional command (`forward`, `backward`, `left`, `right`, `up`, `down`, `yawleft`, `yawright`), create a segment spanning the duration of the command and enter the corresponding label
4. For passages where the speaker is talking but not uttering a directional command (irrelevant speech, hesitations, comments, etc.), create a segment and label it `none`
5. Silences (no vocal activity) are left unannotated — do not create segments for silent parts

In short: all speech must be annotated (command or `none`), only silences are left unannotated. Empty intervals are automatically treated as `none` by the pipeline.

## Configuration (config.yaml)

```yaml
paths:
  textgrid_dir: "data/textgrid"    # Praat annotations (.TextGrid)
  audio_dir: "data/audio"          # Audio recordings (.wav)
  output_dir: "output"             # Pipeline outputs

data_preparation:
  tier_name: "commands"    # TextGrid tier name
  skip_if_cached: true     # Reuse existing intermediate files
  test_size: 0.15          # Fraction of participants held out for testing
  random_seed: 42          # Seed for reproducible train/test split

training:
  balance_classes: true    # Subsample the "none" class
  none_ratio: 1.0          # Max ratio of "none" vs largest command class
  n_folds: 5               # Number of cross-validation folds (GroupKFold)
```

## Pipeline steps

### 1. Data preparation — `python prepare_data.py`

Five steps executed sequentially:

1. **Parsing** TextGrids: extracts annotations from the `commands` tier into `dataset.csv`
2. **Splitting** participants 85/15 into train/test sets (by participant, not by segment)
3. **Segmenting** training audio into 16 kHz WAV clips in `train/audio_segments/`
4. **Extracting** wav2vec2-FR-7K-large embeddings (mean-pooling) into `train/all_embeddings.npz`
5. **Copying** raw audio files for test participants into `test/audio/`

The `skip_if_cached: true` option allows resuming an interrupted run without recomputing completed steps.

### 2. SVM training — `python train_svm.py`

- Reads data from `output/train/`
- 5-fold GroupKFold cross-validation (grouped by participant)
- Per fold: class balancing, StandardScaler, SVC(kernel='rbf', C=10, class_weight='balanced')
- Reports per-fold and average F1 metrics
- Trains final model on all training data
- Saves to `output/SVM_model/`

### 3. MLP training — `python train_mlp.py`

Same structure as SVM:

- 5-fold GroupKFold cross-validation
- Per fold: class balancing, StandardScaler, MLPClassifier
- Saves to `output/MLP_model/`

### 4. Prediction — `python predict.py --use-ground-truth`

In `--use-ground-truth` mode (recommended for evaluation):

- Uses segment boundaries from `dataset_test.csv`
- Extracts wav2vec2 embeddings from test segments
- Classifies with SVM and/or MLP
- Saves prediction CSVs to `output/predictions/`

Other available modes:

```bash
python predict.py                          # VAD on test/audio/, both models
python predict.py --model svm              # SVM only
python predict.py --audio path/to/file.wav # Any audio file
```

### 5. Evaluation — `python evaluate.py`

- Compares predictions (`predictions_svm.csv`, `predictions_mlp.csv`) against ground truth labels
- Per model: accuracy, F1-macro, F1-weighted, per-class report, confusion matrix
- Side-by-side comparison table when both models are available
- Saves to `output/evaluation/`

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. The first run of `prepare_data.py` automatically downloads the wav2vec2 model (~3 GB).
