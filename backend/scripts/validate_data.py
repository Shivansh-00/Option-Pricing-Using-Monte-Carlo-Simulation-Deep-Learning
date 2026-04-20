"""Quick data quality validation for generated CSVs."""
import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

files = {
    "spot_prices.csv": ["timestamp","spot","open","high","low","close","volume"],
    "implied_volatility.csv": ["timestamp","iv_1W","iv_1M","iv_2M","iv_3M","iv_6M","iv_1Y","atm_iv","skew_25d","skew_10d"],
    "option_chain.csv": ["timestamp","strike","expiry","option_type","bid","ask","mid","last","volume","open_interest","implied_vol","delta","gamma","vega","theta"],
    "market_indicators.csv": ["timestamp","sma_20","sma_50","ema_12","ema_26","rsi_14","bb_upper","bb_lower","macd","macd_signal","atr_14"],
    "market_data.csv": ["date","spot","rate","vix","volume"],
}

for fname, expected_cols in files.items():
    path = os.path.join(DATA_DIR, fname)
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)

    col_ok = cols == expected_cols
    empty_count = 0
    negative_prices = 0
    for row in rows:
        for k, v in row.items():
            if v == "" or v == "nan" or v == "NaN":
                empty_count += 1
            if k in ("spot", "close", "bid", "ask", "mid") and v and v != "":
                try:
                    if float(v) < 0:
                        negative_prices += 1
                except ValueError:
                    pass

    status = "PASS" if col_ok and negative_prices == 0 else "WARN"
    print(f"[{status}] {fname}: {len(rows)} rows, cols_match={col_ok}, empty_cells={empty_count}, negative_prices={negative_prices}")
