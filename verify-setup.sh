#!/bin/bash
# verify-signalstack.sh - Quick verification script for SignalStack setup

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  SignalStack Environment Verification"
echo "═══════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

echo ""
echo "📦 Dependencies"
echo "───────────────────────────────────────────────────────────────"

python3 --version 2>/dev/null
check "Python 3 installed"

python3 -c "import pandas" 2>/dev/null
check "pandas installed"

python3 -c "import yfinance" 2>/dev/null
check "yfinance installed (required for training)"

python3 -c "import torch" 2>/dev/null
check "PyTorch installed"

python3 -c "import xgboost" 2>/dev/null
check "XGBoost installed"

python3 -c "import mlflow" 2>/dev/null
check "MLflow installed"

echo ""
echo "🚀 Docker & Services"
echo "───────────────────────────────────────────────────────────────"

command -v docker >/dev/null 2>&1
check "Docker installed"

docker ps >/dev/null 2>&1
check "Docker daemon running"

if command -v docker-compose >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker-compose installed"
elif docker compose version >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} docker compose (V2) available"
else
    echo -e "${YELLOW}⚠${NC} docker-compose not found (may need: docker compose)"
fi

echo ""
echo "📁 Project Structure"
echo "───────────────────────────────────────────────────────────────"

[ -f "requirements.txt" ] && check "requirements.txt exists"
[ -f "docker-compose.yml" ] && check "docker-compose.yml exists"
[ -d "training" ] && check "training/ directory exists"
[ -f "training/train.py" ] && check "training/train.py exists"
[ -f "training/data_loader.py" ] && check "training/data_loader.py exists"
[ -f "QUICKSTART.md" ] && check "QUICKSTART.md documentation exists"

echo ""
echo "🧪 Quick Test"
echo "───────────────────────────────────────────────────────────────"

# Test yfinance data loading
python3 << 'EOF'
try:
    import yfinance as yf
    
    # Quick fetch test (small date range)
    ticker = yf.download("AAPL", start="2024-12-01", end="2024-12-05", 
                         interval="1h", progress=False, verbose=False)
    
    if len(ticker) > 0:
        print(f"✓ yfinance data fetch successful ({len(ticker)} rows)")
    else:
        print("✗ yfinance returned no data")
        exit(1)
except Exception as e:
    print(f"✗ yfinance test failed: {e}")
    exit(1)
EOF
check "yfinance data fetch test"

# Test training data loader import
python3 << 'EOF'
try:
    from training.data_loader import YFinanceLoader, TimescaleLoader, FEATURE_COLS
    
    loader = YFinanceLoader(symbols=["AAPL"])
    assert len(FEATURE_COLS) == 10, f"Expected 10 features, got {len(FEATURE_COLS)}"
    print(f"✓ YFinanceLoader instantiated successfully")
except Exception as e:
    print(f"✗ data_loader import failed: {e}")
    exit(1)
EOF
check "training.data_loader module"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ All checks passed! Ready to train."
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Train with yfinance (FREE):"
echo "     python -m training.train --symbols AAPL --start 2024-01-01 --end 2024-03-31 --data-source yfinance"
echo ""
echo "  2. View guide: cat QUICKSTART.md"
echo ""
