Here’s a **README.md** draft for your GitHub repository based on the analysis you performed (as extracted from your PDF):

---

# BTC Trade Analysis

This repository contains code and analysis for backtesting Bitcoin (BTC/USDT) perpetual futures trading strategies using 15-minute candlestick data from Bybit for the year 2024.

## 📈 Overview

The project covers:

* Data loading, cleaning, and preprocessing
* Calculation of technical indicators (EMA, ATR)
* Identification of long and short trade setups
* Backtesting of trading strategies with position sizing, slippage, and commission
* Performance metrics and equity curve visualization

## ⚙️ Features

* **Data Preprocessing**

  * Cleans redundant columns
  * Removes duplicates
  * Converts timestamps and sets proper index

* **Technical Indicators**

  * Exponential Moving Averages (EMA) for multiple periods
  * Average True Range (ATR) for volatility

* **Trade Setup Logic**

  * Long trades: bullish candles, lower high/low, close above EMA
  * Short trades: bearish candles, higher high/low, close below EMA
  * Customizable ATR-based targets and stop losses

* **Backtesting**

  * Includes slippage, commission, risk management
  * Calculates equity curve and performance metrics
  * Computes max drawdown, win rate, profit factor, expectancy

## 🚀 Quick Start

1️⃣ Clone the repository:

```bash
git clone https://github.com/yourusername/btc-trade-analysis.git
cd btc-trade-analysis
```

2️⃣ Place your BTC/USDT CSV data in the `data/` directory.

3️⃣ Run the Jupyter notebook or scripts:

```bash
jupyter notebook
```

or

```bash
python scripts/your_script.py
```

## 📊 Example Metrics

* Initial balance: \$10,000
* Metrics reported: total return, win rate, profit factor, average trade duration, max drawdown

## 📌 Requirements

* Python 3.x
* `pandas`
* `numpy`
* `matplotlib`

Install with:

```bash
pip install -r requirements.txt
```

*(create a `requirements.txt` if you haven't yet)*

## 📝 Notes

* The trading logic is based on historical data and may not reflect future performance.
* Adjust risk parameters, slippage, and commission to suit your broker/exchange.

## 📄 License

MIT License. See `LICENSE` file for details.

---

If you'd like, I can help generate the `requirements.txt`, a sample notebook, or a starter `scripts/` layout! Let me know your repo name, author info, and license if you want them filled in properly.
