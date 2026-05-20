# DEPLOYMENT GUIDE
## What To Do With The Downloaded Files

You downloaded the `signals-bot` folder to Google Drive. Here is the exact step-by-step to activate the system.

---

## PHASE A: Create Your GitHub Repository (5 minutes)

1. Go to https://github.com/new
2. Repository name: `signals-bot`
3. Visibility: **Public** (required for free GitHub Actions)
4. Do NOT initialize with README (we already have one)
5. Click **Create repository**

### Upload the files:
```bash
# On your computer, extract the signals-bot folder from Google Drive
# Open terminal / command prompt inside the extracted signals-bot folder

git init
git add .
git commit -m "Initial commit: Forex bot foundation"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/signals-bot.git
git push -u origin main
```

---

## PHASE B: Configure GitHub Secrets (3 minutes)

In your GitHub repo, go to:
**Settings → Secrets and variables → Actions → New repository secret**

Add these 4 secrets EXACTLY:

| Secret Name | Value |
|-------------|-------|
| `KAGGLE_USERNAME` | `chamberbot` |
| `KAGGLE_KEY` | `KGAT_a732ff4152d45e5b742d6ed2e464a67d` |
| `TELEGRAM_BOT_TOKEN` | `8214823027:AAHVfjk9KxRGGlS9svqKaiw4Qg0DFhx0o-8` |
| `TELEGRAM_CHAT_ID` | `7779937295` |

**Why secrets?** These are encrypted by GitHub. They are never visible in code or logs.

---

## PHASE C: Kaggle Setup (10 minutes)

1. Go to https://www.kaggle.com/chamberbot/account
2. Scroll to **API** section → Click **Create New Token**
3. A `kaggle.json` file downloads. Keep it safe.
4. Go to https://www.kaggle.com/code → **New Notebook**
5. In the notebook, click **File → Import Notebook** (or copy-paste the code from `kaggle/etl/daily_etl.py`)
6. In the right panel, click **Add Data → Upload** → upload `kaggle/etl/daily_etl.py` if needed
7. **IMPORTANT**: In notebook settings, upload your `kaggle.json` credentials file
8. Run the notebook once manually to verify it works
9. If successful, it will create dataset `chamberbot/forex-raw-data`
10. Click the **Schedule** button (top right) → Set to **Daily at 00:00 UTC**

---

## PHASE D: Termux Setup on Android (15 minutes)

1. Install **Termux** from F-Droid (NOT Google Play version)
2. Open Termux and run:
```bash
bash ~/storage/shared/signals-bot/termux/setup.sh
```
(Assuming you copied the folder from Google Drive to phone storage)

3. Authenticate GitHub:
```bash
gh auth login
# Select HTTPS → Paste your GitHub token
```

4. Clone your repo:
```bash
cd ~
git clone https://github.com/YOUR_GITHUB_USERNAME/signals-bot.git forex-bot
cd forex-bot
```

5. Create local config copy:
```bash
cp config/signals.yaml config/signals.local.yaml
```

6. Test dry run:
```bash
python termux/edge_infer.py --dry-run
```

---

## PHASE E: Cloud Backup with rclone (10 minutes)

```bash
rclone config
```

Set up these remotes:
1. **gdrive** → Google Drive (auto-auth via browser link)
2. **b2** → Backblaze B2 (create free account at backblaze.com, get key ID + application key)
3. **crypt** → Encrypt layer on top of gdrive

Then sync:
```bash
rclone sync logs/ gdrive:forex-bot/logs
rclone sync models/ b2:forex-bot-models
```

---

## FILE MAP (What each file does)

| File | Where it lives | What it does |
|------|---------------|--------------|
| `config/signals.yaml` | GitHub + Termux | Master trading configuration |
| `config/kaggle_notebooks.json` | GitHub | Registry of all Kaggle notebook IDs |
| `.github/workflows/nightly_etl.yml` | GitHub | Automated health checks + model releases |
| `kaggle/etl/daily_etl.py` | Kaggle Notebook | Daily data ingestion + feature engineering |
| `termux/setup.sh` | Termux (run once) | Installs all dependencies |
| `termux/edge_infer.py` | Termux | Live signal generation on your phone |

---

## NEXT STEP

Once you confirm Phases A-E are complete, we will proceed to **Step 2: Kaggle Agent Swarm** (the actual AI model training notebooks).
