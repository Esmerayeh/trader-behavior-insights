# PrimeTrade Assignment — Sentiment-Conditioned Trader DNA

This submission is designed to stand out from a standard EDA.

Instead of only asking **“Are traders better in Fear or Greed?”**, it builds a **behavioral intelligence layer** on top of the dataset:

- **Regime Lens:** how realized trader performance changes across Fear, Neutral, Greed, Extreme Fear, and Extreme Greed.
- **Contrarian Edge Matrix:** whether long or short positioning performs better inside each sentiment regime.
- **Transition Lens:** whether sentiment *changes* matter more than static sentiment.
- **Trader DNA Clustering:** how wallets separate into distinct behavioral archetypes.

## Data Used
- `historical_data.csv` — Hyperliquid trade history
- `Fear_Greed_Index.csv` — Fear & Greed dataset

## What makes this submission stronger
1. It goes beyond descriptive charts and introduces a reusable **trader segmentation framework**.
2. It extracts **strategy-ready insights**, not just observations.
3. It includes a **CLI script** with logging and error handling:
   ```bash
   python run_analysis.py --trades /path/to/historical_data.csv --sentiment /path/to/sentiment.tsv --out outputs
   ```
4. It writes a machine-readable `insights.json`, which makes the work extensible for dashboards, automation, or an AI layer later.

## Headline Findings
- **Extreme Greed** was the richest regime *per closing trade*: average realized PnL = **$130.21** and average ROI = **7.67%**.
- **Fear** produced the largest **total** realized PnL pool: **$3.36M**, because more trades were closed there.
- The strongest desks were often **contrarian** rather than pure trend-followers:
  - In **Extreme Greed**, **short-side closes** generated **$2.53M** at **9.75% average ROI**
  - Long-side closes generated only **$0.19M** at **3.08% average ROI**
- Sentiment **transitions** mattered:
  - **Fear → Greed** days had the highest average daily realized PnL among transitions with at least 2 observations: **$208.6K/day**

## Trader DNA Archetypes
The wallet clustering step found four distinct archetypes:

### 1) Diversified Alpha Scalers
- 7 wallets
- Highest average wallet PnL: **$543.8K**
- Broadest market coverage: **66.6 coins on average**
- Strong ROI with balanced directional behavior

### 2) Volume-First Short Specialists
- 7 wallets
- Very high short share: **84.4%**
- Highest notional deployment
- Best fit for euphoric fade setups

### 3) Opportunistic Swing Traders
- 12 wallets
- Lower edge and weaker ROI quality
- More mixed directional behavior

### 4) Precision Snipers
- 6 wallets
- Small activity footprint, but extremely high efficiency
- Average win rate near **98.8%**
- Highest average ROI, but on far fewer trades

## Strategy Interpretation
The main lesson is not a simplistic “fear good / greed bad” rule.

The stronger interpretation is:

- **Euphoria is fadeable**
- **Fear is a deep liquidity and exit-opportunity zone**
- **Wallets must be segmented before they are copied**
- **Regime shifts are often more informative than regime levels**

## Files
- `run_analysis.py` — reproducible CLI analysis script
- `outputs/insights.json` — strict machine-readable summary
- `outputs/*.csv` — summary tables
- `outputs/*.png` — charts
- `notebooks/Trader_DNA_Analysis.ipynb` — notebook version of the analysis

## Notes
- The uploaded trade CSV did **not** include an explicit leverage column, so the analysis uses realized PnL, notional size, and trade direction instead.
- A handful of trade rows had no matching sentiment day and were dropped automatically by the pipeline.
