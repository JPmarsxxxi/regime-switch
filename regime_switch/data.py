"""Cached loaders for the free data sources: Yahoo Finance, FRED, Ken French's data library."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _cached(name, fetch):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    obj = fetch()
    obj.to_pickle(path)
    return obj


def yahoo_close(ticker):
    def fetch():
        import yfinance as yf

        df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.dropna().rename(ticker)

    return _cached("yahoo_" + ticker.replace("^", ""), fetch)


def fred(series_id):
    def fetch():
        text = requests.get(FRED_URL.format(series_id), timeout=60).text
        df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True)
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().rename(series_id)

    return _cached("fred_" + series_id, fetch)


def french(name, section=0):
    """One table of a Ken French CSV file (section 0 = the first table, usually value-weighted returns), in decimals."""

    def fetch():
        raw = requests.get(FRENCH_URL.format(name), timeout=120).content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        return pd.Series([text])

    text = _cached("french_" + name, fetch).iloc[0]
    tables, current = [], None
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) > 1 and cells[0] == "" and current is None:
            current = {"cols": cells[1:], "rows": []}
        elif current is not None and cells[0].isdigit() and len(cells) > 1:
            current["rows"].append(cells)
        elif current is not None and current["rows"]:
            tables.append(current)
            current = None
    if current is not None and current["rows"]:
        tables.append(current)
    t = tables[section]
    idx = [r[0] for r in t["rows"]]
    df = pd.DataFrame([r[1 : len(t["cols"]) + 1] for r in t["rows"]], index=idx, columns=t["cols"]).astype(float)
    fmt = "%Y%m%d" if len(idx[0]) == 8 else "%Y%m"
    df.index = pd.to_datetime(df.index, format=fmt)
    return df / 100.0
