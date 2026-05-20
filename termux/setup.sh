#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# Forex Bot - Termux Bootstrap
# Run this once after installing Termux from F-Droid
# ============================================================

set -e

echo "========================================"
echo "  FOREX BOT - TERMUX SETUP"
echo "========================================"

# 1. Update base packages
echo "[1/7] Updating Termux base packages..."
apt update && apt upgrade -y

# 2. Install core dependencies
echo "[2/7] Installing core packages..."
apt install -y git python python-pip gh rclone openssh curl jq

# 3. Configure storage access
echo "[3/7] Requesting storage permission..."
termux-setup-storage

# 4. Create project workspace
echo "[4/7] Creating workspace..."
mkdir -p ~/forex-bot
cd ~/forex-bot

# 5. Clone repository (user will replace with their repo URL)
echo "[5/7] Repository setup..."
if [ ! -d ".git" ]; then
    echo "      Please run: gh auth login"
    echo "      Then: git clone https://github.com/YOUR_USER/signals-bot.git ."
    echo "      Skipping auto-clone."
fi

# 6. Install Python ML stack
echo "[6/7] Installing Python ML stack..."
pip install --upgrade pip
pip install onnxruntime yfinance pandas numpy pyyaml requests schedule     python-telegram-bot scikit-learn matplotlib seaborn

# 7. Setup rclone remotes (user will configure interactively)
echo "[7/7] Rclone setup..."
if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo "      Run 'rclone config' to set up:"
    echo "        - gdrive (Google Drive)"
    echo "        - b2 (Backblaze B2)"
    echo "        - crypt (encrypted layer on top)"
fi

echo ""
echo "========================================"
echo "  SETUP COMPLETE"
echo "========================================"
echo "Next steps:"
echo "  1. gh auth login"
echo "  2. git clone <your-repo> ~/forex-bot"
echo "  3. cd ~/forex-bot && cp config/signals.yaml config/signals.local.yaml"
echo "  4. rclone config"
echo "  5. python termux/edge_infer.py --dry-run"
