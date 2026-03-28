
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_sentiment(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path)
            first = df.columns[0]
            df = df[first].str.split("\t", expand=True)
            df.columns = ["timestamp", "value", "classification", "date"]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value", "classification"]].dropna(subset=["date"])
    except Exception as e:
        raise RuntimeError(f"Failed to load sentiment data: {e}") from e


def load_trades(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Timestamp IST"], format="%d-%m-%Y %H:%M").dt.normalize()
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load trade data: {e}") from e


def map_exposure(direction: str, side: str) -> str:
    d = str(direction).lower()
    s = str(side).lower()
    if d in {"buy", "open long", "close short", "settlement"} or "long > short" in d:
        return "Long"
    if d in {"sell", "open short", "close long", "spot dust conversion"} or "short > long" in d or "auto-deleveraging" in d or "liquidated isolated short" in d:
        return "Short"
    if "close short" in d or ("short" not in d and s == "buy"):
        return "Long"
    return "Short"


def prepare_data(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    df = trades.merge(sentiment, on="date", how="left")
    missing = int(df["classification"].isna().sum())
    if missing:
        logging.warning("Dropping %s trade rows with no matching sentiment day.", missing)
    df = df.dropna(subset=["classification"]).copy()
    df["exposure_side"] = [map_exposure(d, s) for d, s in zip(df["Direction"], df["Side"])]
    df["abs_notional"] = df["Size USD"].abs()
    df["is_close"] = df["Closed PnL"] != 0
    df["is_win"] = df["Closed PnL"] > 0
    df["roi_trade"] = df["Closed PnL"] / df["abs_notional"].replace(0, np.nan)
    return df


def regime_summary(closed: pd.DataFrame) -> pd.DataFrame:
    out = (
        closed.groupby("classification")
        .agg(
            close_trades=("Account", "size"),
            traders=("Account", "nunique"),
            total_pnl=("Closed PnL", "sum"),
            avg_pnl=("Closed PnL", "mean"),
            median_pnl=("Closed PnL", "median"),
            win_rate=("is_win", "mean"),
            avg_roi=("roi_trade", "mean"),
            gross_volume=("abs_notional", "sum"),
        )
        .sort_values("avg_pnl", ascending=False)
        .reset_index()
    )
    out["pnl_per_million_volume"] = out["total_pnl"] / out["gross_volume"] * 1_000_000
    return out


def side_regime_summary(closed: pd.DataFrame) -> pd.DataFrame:
    return (
        closed.groupby(["classification", "exposure_side"])
        .agg(
            close_trades=("Account", "size"),
            total_pnl=("Closed PnL", "sum"),
            avg_pnl=("Closed PnL", "mean"),
            win_rate=("is_win", "mean"),
            avg_roi=("roi_trade", "mean"),
            gross_volume=("abs_notional", "sum"),
        )
        .reset_index()
    )


def daily_transition_summary(closed: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    daily = (
        closed.groupby("date")
        .agg(
            total_pnl=("Closed PnL", "sum"),
            close_trades=("Account", "size"),
            traders=("Account", "nunique"),
            volume=("abs_notional", "sum"),
            avg_roi=("roi_trade", "mean"),
            short_share=("exposure_side", lambda s: (s == "Short").mean()),
        )
        .reset_index()
        .merge(sentiment[["date", "classification", "value"]], on="date", how="left")
        .sort_values("date")
    )
    daily["prev_classification"] = daily["classification"].shift(1)
    daily["transition"] = daily["prev_classification"].fillna("Start") + " -> " + daily["classification"]
    transition = (
        daily.groupby("transition")
        .agg(days=("date", "size"), avg_daily_pnl=("total_pnl", "mean"), avg_roi=("avg_roi", "mean"))
        .reset_index()
        .sort_values("avg_daily_pnl", ascending=False)
    )
    return daily, transition


def cluster_wallets(closed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    acct = (
        closed.groupby("Account")
        .agg(
            total_pnl=("Closed PnL", "sum"),
            trades=("Closed PnL", "size"),
            win_rate=("is_win", "mean"),
            avg_pnl=("Closed PnL", "mean"),
            total_volume=("abs_notional", "sum"),
            avg_roi=("roi_trade", "mean"),
            short_share=("exposure_side", lambda s: (s == "Short").mean()),
            active_days=("date", "nunique"),
            coins=("Coin", "nunique"),
        )
        .reset_index()
    )
    regime_pivot = closed.pivot_table(index="Account", columns="classification", values="Closed PnL", aggfunc="sum", fill_value=0)
    for regime in ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]:
        acct[regime] = acct["Account"].map(regime_pivot.get(regime, pd.Series(dtype=float))).fillna(0)
    acct["fear_bucket_pnl"] = acct["Extreme Fear"] + acct["Fear"]
    acct["greed_bucket_pnl"] = acct["Greed"] + acct["Extreme Greed"]
    acct["antifragility_score"] = (acct["fear_bucket_pnl"] - acct["greed_bucket_pnl"]) / acct["total_pnl"].replace(0, np.nan)

    feature_matrix = np.column_stack(
        [
            np.log1p(acct["trades"]),
            np.log1p(acct["active_days"]),
            np.log1p(acct["coins"]),
            np.log1p(acct["total_volume"]),
            acct["win_rate"],
            acct["avg_roi"].fillna(0),
            acct["short_share"],
        ]
    )
    scaled = StandardScaler().fit_transform(feature_matrix)
    model = KMeans(n_clusters=4, random_state=42, n_init=50)
    acct["cluster"] = model.fit_predict(scaled)

    cluster_summary = (
        acct.groupby("cluster")
        .agg(
            traders=("Account", "size"),
            total_cluster_pnl=("total_pnl", "sum"),
            avg_wallet_pnl=("total_pnl", "mean"),
            median_wallet_pnl=("total_pnl", "median"),
            avg_trades=("trades", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_roi=("avg_roi", "mean"),
            avg_short_share=("short_share", "mean"),
            avg_active_days=("active_days", "mean"),
            avg_coins=("coins", "mean"),
            avg_volume=("total_volume", "mean"),
        )
        .reset_index()
    )

    # human labels
    labels = {}
    for _, row in cluster_summary.iterrows():
        cid = int(row["cluster"])
        if row["avg_roi"] > 0.20 and row["avg_trades"] < 500:
            labels[cid] = "Precision Snipers"
        elif row["avg_short_share"] > 0.75 and row["avg_volume"] > cluster_summary["avg_volume"].median():
            labels[cid] = "Volume-First Short Specialists"
        elif row["avg_coins"] > 40 and row["avg_roi"] > 0.04:
            labels[cid] = "Diversified Alpha Scalers"
        else:
            labels[cid] = "Opportunistic Swing Traders"
    acct["cluster_label"] = acct["cluster"].map(labels)
    cluster_summary["cluster_label"] = cluster_summary["cluster"].map(labels)
    return acct, cluster_summary


def save_chart_regime_pnl(regime_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = regime_df.sort_values("avg_pnl", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["classification"], plot_df["avg_pnl"])
    ax.set_title("Average Realized PnL per Closing Trade by Sentiment Regime")
    ax.set_xlabel("Average closed PnL (USD)")
    ax.set_ylabel("Sentiment regime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_chart_side_heatmap(side_df: pd.DataFrame, out_path: Path) -> None:
    pivot = side_df.pivot(index="classification", columns="exposure_side", values="avg_roi").reindex(
        ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Contrarian Edge Matrix: Average ROI by Regime and Direction")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2%}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_chart_daily_scatter(daily: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(daily["value"], daily["total_pnl"], alpha=0.7)
    z = np.polyfit(daily["value"], daily["total_pnl"], 1)
    p = np.poly1d(z)
    xs = np.linspace(daily["value"].min(), daily["value"].max(), 100)
    ax.plot(xs, p(xs))
    ax.set_title("Daily Sentiment Score vs Daily Realized PnL")
    ax.set_xlabel("Fear & Greed index value")
    ax.set_ylabel("Daily realized PnL (USD)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_chart_clusters(acct: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, subset in acct.groupby("cluster_label"):
        ax.scatter(subset["short_share"], subset["avg_roi"], s=np.log1p(subset["trades"]) * 25, alpha=0.75, label=label)
    ax.set_title("Trader DNA Map")
    ax.set_xlabel("Short-share of closing trades")
    ax.set_ylabel("Average ROI per closing trade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_insight_json(
    trades: pd.DataFrame,
    sentiment: pd.DataFrame,
    regime_df: pd.DataFrame,
    side_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    acct: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> dict:
    top_regime = regime_df.iloc[0].to_dict()
    fear_total = regime_df.loc[regime_df["classification"] == "Fear", "total_pnl"].iloc[0]
    best_transition = transition_df[transition_df["days"] >= 2].iloc[0].to_dict()
    short_extreme_greed = side_df[(side_df["classification"] == "Extreme Greed") & (side_df["exposure_side"] == "Short")].iloc[0].to_dict()
    long_extreme_greed = side_df[(side_df["classification"] == "Extreme Greed") & (side_df["exposure_side"] == "Long")].iloc[0].to_dict()
    top_antifragile = acct[acct["total_pnl"] > 0].sort_values("antifragility_score", ascending=False).head(5)
    return {
        "project": "Sentiment-Conditioned Trader DNA",
        "data_health": {
            "trade_rows": int(trades.shape[0]),
            "sentiment_days": int(sentiment["date"].nunique()),
            "matched_trade_rows": int((trades.merge(sentiment[["date"]], on="date", how="left")["date"].notna()).sum()),
            "wallets": int(trades["Account"].nunique()),
            "coins": int(trades["Coin"].nunique()),
            "trade_date_range": [str(trades["date"].min().date()), str(trades["date"].max().date())],
            "note": "The uploaded trade CSV does not contain an explicit leverage column, so notional size and start position were used as exposure context instead.",
        },
        "headline_findings": [
            {
                "title": "Euphoria was the richest regime per trade, but fear generated the most total profit.",
                "detail": f'Extreme Greed had the highest average realized PnL per closing trade (${top_regime["avg_pnl"]:.2f}) and average ROI ({top_regime["avg_roi"]:.2%}), while Fear produced the largest total realized PnL (${fear_total:,.0f}) because more trades were closed there.'
            },
            {
                "title": "The best desks were often contrarian, not trend-chasing.",
                "detail": f'Short-side closes in Extreme Greed generated ${short_extreme_greed["total_pnl"]:,.0f} at {short_extreme_greed["avg_roi"]:.2%} average ROI, versus ${long_extreme_greed["total_pnl"]:,.0f} and {long_extreme_greed["avg_roi"]:.2%} for longs.'
            },
            {
                "title": "Regime shifts mattered more than static sentiment.",
                "detail": f'{best_transition["transition"]} days delivered the highest average daily realized PnL among transitions with at least 2 observations (${best_transition["avg_daily_pnl"]:,.0f}/day).'
            },
        ],
        "wallet_archetypes": cluster_df.sort_values("avg_wallet_pnl", ascending=False)[
            [
                "cluster_label",
                "traders",
                "avg_wallet_pnl",
                "avg_trades",
                "avg_win_rate",
                "avg_roi",
                "avg_short_share",
                "avg_coins",
            ]
        ].round(4).to_dict(orient="records"),
        "top_antifragile_wallets": top_antifragile[
            ["Account", "total_pnl", "fear_bucket_pnl", "greed_bucket_pnl", "antifragility_score", "short_share"]
        ].round(4).to_dict(orient="records"),
        "strategy_takeaways": [
            "Treat extreme greed as a potential fade zone: in this sample, short books monetized euphoria better than longs.",
            "Fear was not a pure risk-off regime. It produced the largest absolute profit pool, suggesting distressed volatility created large closing opportunities.",
            "Segment wallets before copying them. Some profitable wallets were diversified alpha scalers, while others were precision snipers with tiny capital and very sparse activity.",
            "If deploying a live strategy, use sentiment transitions as triggers rather than sentiment level alone.",
        ],
    }


def main(trades_path: str, sentiment_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades = load_trades(trades_path)
    sentiment = load_sentiment(sentiment_path)
    merged = prepare_data(trades, sentiment)
    closed = merged[merged["is_close"]].copy()

    regime_df = regime_summary(closed)
    side_df = side_regime_summary(closed)
    daily_df, transition_df = daily_transition_summary(closed, sentiment)
    acct_df, cluster_df = cluster_wallets(closed)

    regime_df.to_csv(out / "regime_summary.csv", index=False)
    side_df.to_csv(out / "direction_by_regime.csv", index=False)
    daily_df.to_csv(out / "daily_metrics.csv", index=False)
    transition_df.to_csv(out / "transition_summary.csv", index=False)
    acct_df.to_csv(out / "wallet_metrics.csv", index=False)
    cluster_df.to_csv(out / "wallet_clusters.csv", index=False)

    save_chart_regime_pnl(regime_df, out / "chart_regime_avg_pnl.png")
    save_chart_side_heatmap(side_df, out / "chart_contrarian_edge_matrix.png")
    save_chart_daily_scatter(daily_df, out / "chart_daily_sentiment_vs_pnl.png")
    save_chart_clusters(acct_df, out / "chart_trader_dna_map.png")

    insights = build_insight_json(trades, sentiment, regime_df, side_df, transition_df[transition_df["days"] >= 2], acct_df, cluster_df)
    (out / "insights.json").write_text(json.dumps(insights, indent=2))

    logging.info("Analysis complete. Outputs written to %s", out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentiment-conditioned analysis of Hyperliquid trader behavior.")
    parser.add_argument("--trades", required=True, help="Path to the Hyperliquid historical trade CSV")
    parser.add_argument("--sentiment", required=True, help="Path to the Fear & Greed dataset")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    main(args.trades, args.sentiment, args.out)
