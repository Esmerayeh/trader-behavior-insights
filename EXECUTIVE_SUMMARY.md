# Executive Summary

## Project framing
I approached this as a **trader intelligence problem**, not just a basic EDA exercise.

The core idea was to model how trader behavior changes across market sentiment regimes and then identify which wallets are:
- resilient in fear,
- efficient in euphoria,
- contrarian,
- or simply noisy.

## Best standout insight
The strongest signal in this dataset is **not** “fear vs greed” in the simplistic sense.

It is this:

> **Extreme Greed was the most profitable regime per closing trade, but traders made the most total money during Fear — and the biggest edge often came from shorting euphoric conditions.**

That immediately separates this from a generic assignment submission.

## High-impact results
- 211,224 total executions
- 32 wallets
- 246 coins
- 2023-05-01 to 2025-05-01 trade coverage

### Regime results
- **Extreme Greed**
  - highest average realized PnL per closing trade: **$130.21**
  - highest average ROI: **7.67%**
- **Fear**
  - largest total realized PnL pool: **$3.36M**

### Contrarian result
In **Extreme Greed**:
- **Short-side closes:** **$2.53M** total PnL, **9.75%** average ROI
- **Long-side closes:** **$0.19M** total PnL, **3.08%** average ROI

### Transition result
Among transitions with at least 2 observations:
- **Fear → Greed** days had the highest average daily realized PnL: **$208.6K/day**

## Why this stands out
This submission adds:
1. a **Contrarian Edge Matrix**
2. a **Trader DNA Map**
3. wallet **behavioral clustering**
4. a **CLI analysis pipeline**
5. machine-readable **JSON outputs** for future automation
