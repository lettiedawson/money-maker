# Money Maker — Value Investing Tool

A lightweight Python command-line worksheet to guide **value investing stock purchase decisions**.  
It outputs a **Markdown report** and a **CSV** of key metrics, checks, and a quick decision heuristic.

---

## 📦 Features
- Input your own fundamentals (offline JSON file).
- Optional auto-fetch of some metrics via [yfinance].
- Computes:
  - ROE, D/E, dividend payout ratio
  - Free Cash Flow trend and CAGR
  - Graham Intrinsic Value (heuristic)
  - Margin of Safety
- Checklist of value-investing rules.
- Final suggestion: **Buy / Consider / Pass**.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/money_maker.git
cd money_maker

### 2. Create and activate a virtual environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

### 3. Install requirements
```bash
pip install -r requirements.txt

## 🛠 Usage
### Offline mode (recommended to start)
Edit `sample_inputs.json` with your company’s data, then run:
```bash
python value_sheet.py --input sample_inputs.json --out report

OUTPUTS:
- `report.md` ==> nicely formatted worksheet
- `report.csv` ==> raw mterics in table formatted

### Online fetch (Optional)
If you have internet and `yfinance` installed: 
```bash
python value_sheet.py --ticker AAPL --fetch --roe 0.22 --de 0.5 --out apple

## 🧩 Files
- `value_sheet.py` → main script
- `sample_inputs.json` → example input data
- `requirements.txt` → required Python packages
- `.gitignore` → ignores venv, caches, logs, etc.
- `README.md` → this file

## ⚠️ Disclaimer
This tool is for _educational purposes only_. It does not constitute _financial advice_.  Always do your own research before making investment decisions. 