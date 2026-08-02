# NEU-Surface-Detect

End-to-end MLOps pipeline for detecting surface defects using the [NEU Surface Defect dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).

## Project Goal

Build a production-ready computer vision system that:

- Ingests and preprocesses defect images
- Trains a defect classification model with PyTorch
- Tracks experiments and model versions with MLflow
- Serves predictions through a FastAPI API
- Supports containerized deployment, CI/CD, monitoring, and retraining

## Defect Classes

The dataset contains six grayscale defect categories:

| Class | Description |
|-------|-------------|
| `crazing` | Fine cracks on the surface |
| `inclusion` | Non-metallic inclusions |
| `patches` | Patches of uneven texture |
| `pitted_surface` | Small pits or holes |
| `rolled-in_scale` | Rolled-in oxide scale |
| `scratches` | Linear scratch marks |

## Project Structure

```text
NEU-Surface-Detect/
├── data_ingestion/     # Download, organize, and inspect dataset
├── training/           # Model training pipeline
├── models/             # Saved checkpoints and MLflow registry
├── inference/          # FastAPI prediction service
├── cicd/               # GitHub Actions workflows
├── monitoring/         # Drift detection and logging
├── scripts/            # Environment setup and verification
├── dataset/            # Local dataset (gitignored)
└── docs/               # Project plans and notes (gitignored)
```

## Requirements

- Python 3.10 or 3.11
- macOS, Linux, or WSL

## License

See [LICENSE](LICENSE).
