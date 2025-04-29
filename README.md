# Machine Vision Pipeline for Pose Analysis

## Project Structure
```
├── config/             # Configuration file
│   └── config.json     # Model parameters and paths
├── data/               # Raw and cached video datasets
├── outputs/            # Training logs and model checkpoints
└── src/                # Source file folder
```

## Installation
```bash
pip install -r requirements.txt
```

## Webcam Inference Workflow
1. Start real-time inference with webcam:
```bash
python -m src.inference.inference
```


Inference:
- Press 'Q' to exit
- Visual feedback shows landmark confidence scores
- Terminal also prints confidence scores

## Configuration (config.json)
All relevant configurations can be done in /config/config.json
If setting is not found there it is not adivsable to change.