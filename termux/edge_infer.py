#!/usr/bin/env python3
"""
Forex Bot - Edge Inference Engine (Termux)
Downloads latest ONNX models from GitHub Releases and runs real-time inference.
If meta_ensemble.onnx exists, uses it as the final arbitrator.
Otherwise falls back to weighted voting.
"""

import os
import sys
import json
import time
import yaml
import logging
import numpy as np
import pandas as pd
import onnxruntime as ort
from datetime import datetime, timedelta
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/edge_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("edge_infer")

class ForexEdgeBot:
    def __init__(self, config_path="config/signals.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.agents = {}
        self.meta_model = None
        self.load_models()

    def load_models(self):
        """Load all ONNX models from models/ directory."""
        model_dir = "models"
        if not os.path.exists(model_dir):
            logger.error("Models directory not found. Run model download first.")
            sys.exit(1)

        for fname in os.listdir(model_dir):
            if not fname.endswith(".onnx"):
                continue
            path = os.path.join(model_dir, fname)
            name = fname.replace(".onnx", "")
            try:
                sess = ort.InferenceSession(path)
                if name == "meta_ensemble":
                    self.meta_model = sess
                    logger.info(f"Loaded META model: {fname}")
                else:
                    self.agents[name] = sess
                    logger.info(f"Loaded agent: {fname}")
            except Exception as e:
                logger.error(f"Failed to load {fname}: {e}")

        if not self.agents:
            logger.error("No valid agent models loaded.")
            sys.exit(1)

        if self.meta_model:
            logger.info("Meta-ensemble detected. Will use learned fusion.")
        else:
            logger.info("No meta-ensemble. Using fixed-weight voting.")

    def fetch_live_features(self, pair_alias, agent_type):
        """Fetch latest bars and compute features for a specific agent."""
        symbol_map = {
            "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
            "XAUUSD": "GC=F", "BTCUSD": "BTC-USD",
        }
        symbol = symbol_map.get(pair_alias)
        if not symbol:
            return None

        # Different agents need different timeframes
        tf_map = {
            "price_action_lstm": "1h",
            "macro_sentiment_xgb": "1d",
            "volatility_regime_rf": "1h",
        }
        interval = tf_map.get(agent_type, "15m")

        try:
            df = yf.download(symbol, period="5d", interval=interval, progress=False)
            if df.empty or len(df) < 30:
                return None
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.reset_index()

            # Compute features
            df["rsi_14"] = self._rsi(df["Close"], 14)
            df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["ema_20_50_cross"] = (df["ema_20"] > df["ema_50"]).astype(int)
            atr = self._atr(df, 14)
            df["atr_14"] = atr
            macd, signal, hist = self._macd(df["Close"])
            df["macd_hist"] = hist
            df["returns"] = df["Close"].pct_change().fillna(0)

            # LSTM needs sequences; others need single vectors
            if agent_type == "price_action_lstm":
                feature_cols = ["rsi_14", "ema_20_50_cross", "bollinger_position", "macd_hist", "atr_14", "returns"]
                # Compute bollinger position
                sma_20 = df["Close"].rolling(20).mean()
                std_20 = df["Close"].rolling(20).std()
                upper = sma_20 + 2 * std_20
                lower = sma_20 - 2 * std_20
                df["bollinger_position"] = (df["Close"] - lower) / (upper - lower + 1e-9)

                if len(df) < 48:
                    return None
                seq = df.iloc[-48:][feature_cols].values.astype(np.float32)
                return seq.reshape(1, 48, len(feature_cols))
            else:
                # Single-row feature vector for tree-based models
                # Simplified: use last row technicals + macro placeholders
                last = df.iloc[-1]
                vec = np.array([
                    last["rsi_14"] / 100.0,
                    float(last["ema_20_50_cross"]),
                    last.get("bollinger_position", 0.5),
                    last["macd_hist"] / last["Close"] if last["Close"] != 0 else 0,
                    last["atr_14"] / last["Close"] if last["Close"] != 0 else 0,
                    last["returns"],
                    0.0, 0.0, 0.0  # macro placeholders (fetched from cache in production)
                ], dtype=np.float32)
                return vec.reshape(1, -1)

        except Exception as e:
            logger.error(f"Live fetch failed for {pair_alias} ({agent_type}): {e}")
            return None

    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _atr(self, df, period=14):
        hl = df["High"] - df["Low"]
        hc = np.abs(df["High"] - df["Close"].shift())
        lc = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _macd(self, series):
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def predict(self, pair_alias):
        """Run inference for a single pair."""
        votes = {}

        # Collect agent predictions
        for name, sess in self.agents.items():
            features = self.fetch_live_features(pair_alias, name)
            if features is None:
                continue
            try:
                inp_name = sess.get_inputs()[0].name
                pred = sess.run(None, {inp_name: features})[0]
                votes[name] = float(pred.flatten()[0])
            except Exception as e:
                logger.warning(f"Inference failed for {name}: {e}")

        if not votes:
            return None

        # META-ENSEMBLE PATH (preferred)
        if self.meta_model and len(votes) >= 3:
            try:
                # Order must match meta-learner training: [lstm, xgb, rf]
                ordered = [
                    votes.get("price_action_lstm", 0.5),
                    votes.get("macro_sentiment_xgb", 0.5),
                    votes.get("volatility_regime_rf", 0.5)
                ]
                meta_input = np.array([ordered], dtype=np.float32)
                inp_name = self.meta_model.get_inputs()[0].name
                meta_pred = self.meta_model.run(None, {inp_name: meta_input})[0]
                score = float(meta_pred.flatten()[0])
                source = "meta_ensemble"
            except Exception as e:
                logger.warning(f"Meta-ensemble failed: {e}. Falling back to voting.")
                score = None
                source = "weighted_fallback"
        else:
            score = None
            source = "weighted_fallback"

        # WEIGHTED VOTING FALLBACK
        if score is None:
            weights = self.config.get("agents", {}).get("weights", {})
            total_weight = sum(weights.get(k, 1.0) for k in votes.keys())
            score = sum(votes[k] * weights.get(k, 1.0) for k in votes) / total_weight

        # Direction mapping
        direction = "NEUTRAL"
        if score > 0.6:
            direction = "LONG"
        elif score < 0.4:
            direction = "SHORT"

        return {
            "pair": pair_alias,
            "direction": direction,
            "confidence": round(score, 4),
            "source": source,
            "votes": votes,
            "timestamp": datetime.utcnow().isoformat()
        }

    def run_cycle(self):
        """Run one full inference cycle across all pairs."""
        pairs = self.config.get("pairs", {}).get("major", [])
        results = []
        for p in pairs:
            alias = p.get("alias")
            if not alias:
                continue
            res = self.predict(alias)
            if res:
                results.append(res)
                logger.info(f"Signal: {alias} -> {res['direction']} ({res['confidence']}) [{res['source']}]")
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run once and exit")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule loop")
    args = parser.parse_args()

    bot = ForexEdgeBot()

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        bot.run_cycle()
    elif args.schedule:
        import schedule
        schedule.every(4).hours.do(bot.run_cycle)
        logger.info("Scheduler started. Running every 4 hours.")
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        bot.run_cycle()
