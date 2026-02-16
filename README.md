# VoiceStick — Classification de commandes vocales pour drone

> **ENGLISH VERSION BELOW**

Pipeline de classification parole-vers-commande pour le pilotage de drones. Des enregistrements audio de locuteurs francophones prononçant des commandes de pilotage sont convertis en embeddings wav2vec2, puis classifiés par des modèles SVM ou MLP.

## Classes de commandes

Les 9 classes reconnues :

| Commande | Description |
|---|---|
| `forward` | Avancer |
| `backward` | Reculer |
| `left` | Translation à gauche |
| `right` | Translation à droite |
| `up` | Monter |
| `down` | Descendre |
| `yawleft` | Rotation à gauche (lacet) |
| `yawright` | Rotation à droite (lacet) |
| `none` | Pas de commande / commande non-explicite |

## Contenu du dépôt

| Dossier | Description | Usage |
|---|---|---|
| `user_version/` | Package d'inférence avec modèles pré-entraînés | Utilisateurs finaux qui veulent simplement obtenir des prédictions sur leurs fichiers audio |
| `dev_version/` | Pipeline complet d'entraînement | Développeurs qui veulent entraîner leurs propres modèles sur de nouvelles données |

Chaque dossier contient son propre `README.md` avec les instructions détaillées.

## Performance des modèles (jeu de test, 9 participants)

| Métrique | SVM | MLP |
|---|---|---|
| Accuracy | 0.836 | **0.866** |
| F1-macro | 0.735 | **0.774** |
| F1-weighted | 0.843 | **0.868** |

## Stack technique

- **Embeddings** : [wav2vec2-FR-7K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large) (LeBenchmark) — embeddings de 1024 dimensions
- **Classifieurs** : SVM (noyau RBF) et MLP (scikit-learn)
- **Segmentation** : VAD par énergie via librosa (`librosa.effects.split`)
- **Validation croisée** : GroupKFold à 5 folds par participant (indépendance locuteur)

## Installation

```bash
git clone https://github.com/aggnim/drone-voice-command-classification.git
cd drone-voice-command-classification
pip install -r requirements.txt
```

## Crédits

- Modèle wav2vec2 : [LeBenchmark](https://huggingface.co/LeBenchmark) — wav2vec2-FR-7K-large
- Classification : [scikit-learn](https://scikit-learn.org/)
- Traitement audio : [librosa](https://librosa.org/)

---

# English Version

# VoiceStick — Drone Voice Command Classification

Speech-to-command classification pipeline for drone piloting. Audio recordings of French speakers issuing piloting commands are converted into wav2vec2 embeddings, then classified using SVM or MLP models.

## Command classes

The 9 recognized classes:

| Command | Description |
|---|---|
| `forward` | Move forward |
| `backward` | Move backward |
| `left` | Strafe left |
| `right` | Strafe right |
| `up` | Ascend |
| `down` | Descend |
| `yawleft` | Rotate left (yaw) |
| `yawright` | Rotate right (yaw) |
| `none` | No command / non-explicit command |

## Repository contents

| Folder | Description | Use case |
|---|---|---|
| `user_version/` | Inference package with pre-trained models | End users who simply want predictions on their audio files |
| `dev_version/` | Full training pipeline | Developers who want to train their own models on new data |

Each folder contains its own `README.md` with detailed instructions.

## Model performance (test set, 9 participants)

| Metric | SVM | MLP |
|---|---|---|
| Accuracy | 0.836 | **0.866** |
| F1-macro | 0.735 | **0.774** |
| F1-weighted | 0.843 | **0.868** |

## Tech stack

- **Embeddings**: [wav2vec2-FR-7K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large) (LeBenchmark) — 1024-dimensional embeddings
- **Classifiers**: SVM (RBF kernel) and MLP (scikit-learn)
- **Segmentation**: Energy-based VAD via librosa (`librosa.effects.split`)
- **Cross-validation**: 5-fold GroupKFold by participant (speaker-independent)

## Installation

```bash
git clone https://github.com/aggnim/drone-voice-command-classification.git
cd drone-voice-command-classification
pip install -r requirements.txt
```

## Credits

- wav2vec2 model: [LeBenchmark](https://huggingface.co/LeBenchmark) — wav2vec2-FR-7K-large
- Classification: [scikit-learn](https://scikit-learn.org/)
- Audio processing: [librosa](https://librosa.org/)
