# what I'm doing: fastapi app that loads compact features, exposes options for the UI, and returns sanitized recommendations as JSON

from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# what I'm doing: locate data folders relative to this file
ROOT = Path(__file__).parent.resolve()
DATA = ROOT / "data"
PROC = DATA / "processed"
ART  = DATA / "artifacts"

app = FastAPI(title="UK Grocery Recommender")

# what I'm doing: mount static and templates
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

# ---------- load data & helpers ----------

def _read_features() -> pd.DataFrame:
    # preference: pickle -> csv.gz
    pkl = PROC / "features_v1.pkl"
    csv = PROC / "features_v1.csv.gz"
    if pkl.exists():
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_csv(csv, low_memory=False)
    # minimal schema guarantee
    need = ["category","brand","price_gbp_sainsburys","price_gbp_tesco","size_grams"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    # coerce
    df["size_grams"] = pd.to_numeric(df["size_grams"], errors="coerce")
    for c in ["price_gbp_sainsburys","price_gbp_tesco"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _read_names_min() -> Optional[pd.DataFrame]:
    p = ART / "names_lookup_min.csv"
    if p.exists():
        d = pd.read_csv(p)
        keep = [c for c in d.columns if c in {"category","brand","name_sainsburys","name_tesco"}]
        return d[keep]
    return None

FE: pd.DataFrame  = _read_features()
NAMES_MIN          = _read_names_min()

# what I'm doing: compute dropdown options (categories, brands per category, common sizes per category)
def _build_options(fe: pd.DataFrame):
    categories: List[str] = sorted(fe["category"].dropna().astype(str).unique().tolist())
    brands_by_cat: Dict[str, List[str]] = {}
    sizes_by_cat: Dict[str, List[int]]  = {}

    for cat in categories:
        sub = fe.loc[fe["category"] == cat]
        brands = (
            sub["brand"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
        )
        brands = sorted(brands)[:50]  # trim long lists
        brands_by_cat[cat] = brands

        # suggest frequent sizes (rounded to nearest 50g), top 10
        sizes = (
            sub["size_grams"].dropna()
            .round(-1)  # smooth tiny noise
            .round(-2) # nearest 100
            .astype(int)
        )
        if not sizes.empty:
            top_sizes = (
                sizes.value_counts()
                .sort_index()
                .sort_values(ascending=False)
                .head(12)
                .index.astype(int)
                .tolist()
            )
        else:
            top_sizes = []
        sizes_by_cat[cat] = top_sizes

    return categories, brands_by_cat, sizes_by_cat

CATEGORIES, BRANDS_BY_CAT, SIZES_BY_CAT = _build_options(FE)

# what I'm doing: display-name enrichment from names_min if available
def _attach_display_names(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if NAMES_MIN is not None:
        d = d.merge(NAMES_MIN, on=["category","brand"], how="left")
    # fallbacks if names missing
    for side in ["sainsburys","tesco"]:
        col = f"name_{side}"
        if col not in d.columns:
            d[col] = np.nan
        mask = d[col].isna()
        d.loc[mask, col] = (
            d.loc[mask, ["brand","category","size_grams"]]
              .assign(size_txt=lambda x: x["size_grams"].fillna(0).round().astype(int).astype(str).replace("0",""))
              .apply(lambda r: f"{r['brand']} {r['category']} {r['size_txt']}g".strip(), axis=1)
        )
        # beautify case
        d[col] = d[col].astype(str).str.replace("_"," ").str.title()
    return d

# what I'm doing: safe JSON (no NaN/inf) + rounded prices
def _sanitize_for_json(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].replace([np.inf, -np.inf], np.nan).round(2)
    out = out.replace({np.nan: None})
    return out.to_dict(orient="records")

# ---------- core recommend ----------

class RecRequest(BaseModel):
    category: str
    brand: Optional[str] = None   # "Any" or "" treated as None
    size_grams: Optional[float] = None  # optional
    top_n: int = 10

def _recommend(req: RecRequest) -> pd.DataFrame:
    d = FE.copy()

    # filters
    d = d[d["category"].astype(str).str.lower() == str(req.category).lower()]

    brand = (req.brand or "").strip()
    if brand and brand.lower() != "any":
        d = d[d["brand"].astype(str).str.lower() == brand.lower()]

    # require rows priced at both retailers
    d = d.dropna(subset=["price_gbp_sainsburys","price_gbp_tesco"])

    # optional size window: ±20%
    if req.size_grams:
        sz = float(req.size_grams)
        lo, hi = 0.8*sz, 1.2*sz
        d = d[d["size_grams"].between(lo, hi, inclusive="both")]

    if d.empty:
        return d

    # cheapest retailer logic
    d["cheapest_retailer"] = np.where(
        d["price_gbp_sainsburys"] <= d["price_gbp_tesco"], "sainsburys", "tesco"
    )
    d["cheapest_price"] = d[["price_gbp_sainsburys","price_gbp_tesco"]].min(axis=1)
    d["other_price"]    = d[["price_gbp_sainsburys","price_gbp_tesco"]].max(axis=1)
    d["gap_pct"]        = ((d["other_price"] - d["cheapest_price"]) / d["cheapest_price"] * 100).round(2)

    # names for display
    d = _attach_display_names(d)

    # ordering: biggest % savings first
    d = d.sort_values(["gap_pct","cheapest_price"], ascending=[False, True]).head(req.top_n)

    keep = [
        "category","brand","cheapest_retailer","cheapest_price","other_price","gap_pct",
        "name_sainsburys","name_tesco","size_grams"
    ]
    return d[keep]

# ---------- routes ----------

@app.get("/")
async def home(request: Request):
    # what I'm doing: pass dropdown options into the template
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_title": "UK Grocery Recommender",
            "categories": CATEGORIES,
            "brands_map": BRANDS_BY_CAT,
            "sizes_map": SIZES_BY_CAT,
        },
    )

@app.post("/api/recommend")
async def recommend(payload: RecRequest):
    df = _recommend(payload)
    records = _sanitize_for_json(df)
    return JSONResponse(content=records)
