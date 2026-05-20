# Forex AI/ML Signals Bot

Zero-cost, GitOps-driven forex signal generation system using Kaggle GPU, GitHub Actions, and Termux.

## Architecture
- **Kaggle**: Heavy compute (ETL, model training, GPU/TPU)
- **GitHub**: Orchestration, model registry, CI/CD, secret management
- **Cloud Backup**: Immutable log/model storage via rclone
- **Termux**: Edge inference, signal dispatch, health monitoring

## Pairs Tracked
EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, XAUUSD, BTCUSD

## Timeframes
15m (entry), 1h, 4h, 1d

## Quick Start
**See [DEPLOYMENT.md](DEPLOYMENT.md) for exact step-by-step instructions.**

## Project Structure
```
signals-bot/
├── .github/workflows/      # CI/CD orchestration
├── config/                 # YAML configs (forex pairs, thresholds)
├── kaggle/                 # Notebook sources
│   ├── etl/                # Data ingestion
│   ├── agents/             # Model training notebooks
│   └── meta/               # Ensemble/meta-learner
├── termux/                 # Edge runtime
├── models/                 # ONNX artifacts (gitignored)
└── logs/                   # Local SQLite + JSONL
```

## Configuration
Edit `config/signals.yaml` to adjust:
- Traded pairs and timeframes
- Signal confidence thresholds
- Agent weights
- Risk limits

## License
MIT - Zero-cost stack only.
