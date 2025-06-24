# ATR-BTC

This repository contains a complete pipeline for analyzing and backtesting Bitcoin (BTC/USDT) perpetual futures strategies using ATR (Average True Range) and EMA (Exponential Moving Average) indicators. The project is based on 15-minute candlestick data from Bybit for the year 2024.

## 📈 Project Highlights

* **Data Exploration & Cleaning**

  * Load BTC/USDT data
  * Remove duplicates and irrelevant columns
  * Convert and index by timestamp

* **Technical Indicators**

  * Compute EMA (10, 20, 50, 100, 200)
  * Compute ATR (10, 20, 50)

* **Trade Setup Identification**

  * Long trades: bullish candles, lower highs/lows, close above EMA
  * Short trades: bearish candles, higher highs/lows, close below EMA

* **Backtesting Engine**

  * Position sizing based on risk %
  * Slippage and commission modeling
  * Equity curve generation
  * Trade performance metrics (win rate, expectancy, profit factor, max drawdown)

## 🗂 Repository Structure

```
.
├── data/                    # BTC/USDT CSV files (Bybit export)
├── notebooks/                # Jupyter notebooks for analysis
├── README.md                 # Project documentation
```

## 🚀 Getting Started

### 1️⃣ Clone this repo

```bash
git clone https://github.com/ilyasustun/ATR-BTC.git
cd ATR-BTC
```

### 2️⃣ Install dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

Install required packages:

```bash
pip install pandas numpy matplotlib
```

### 3️⃣ Run the notebook or scripts

Open and run the Jupyter notebook:

```bash
jupyter notebook
```

Or run scripts directly:

```bash
python scripts/your_script.py
```

### 4️⃣ Data

Place your Bybit BTC/USDT CSV files in the `data/` folder. Update file paths in your code accordingly.

## 📊 Example Metrics (from backtest)

* Initial balance: \$10,000
* Win rate: \~55% (example)
* Profit factor: 1.4
* Max drawdown: 12%

*(Actual results will vary depending on parameters)*

## 🔧 Parameters

* `ema_periods`: List of EMA periods (default: `[10, 20]`)
* `atr_periods`: List of ATR periods (default: `[10, 20]`)
* `slippage_pct`: Default `0.05%`
* `commission_pct`: Default `0.075%`
* `risk_per_trade_pct`: Default `1%`

## 📌 Notes

⚠ This code is for educational and research purposes. Backtested performance does not guarantee future results.

## 📄 License

MIT License. See the [LICENSE](LICENSE) file.

## 🤝 Author

Made by [Ilyas Ustun](https://github.com/ilyasustun).

