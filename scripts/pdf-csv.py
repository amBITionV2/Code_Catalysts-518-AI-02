# scripts/pdf-csv-nojava.py
# Recursively converts PDFs under ./datasets/** to tidy CSVs with City, without Java (uses camelot stream + pdfplumber fallback)

import re
import sys
from pathlib import Path
from slugify import slugify

import pandas as pd

# Optional imports guarded, so script still runs if one is missing
try:
    import camelot
except Exception as e:
    camelot = None

try:
    import pdfplumber
except Exception as e:
    pdfplumber = None

DATA_DIR = Path("datasets")
OUT_DIR  = Path("out")
GROUP_OUT_BY_YEAR = True

OUT_DIR.mkdir(exist_ok=True, parents=True)

CITY_ALIASES = {
    "bengaluru": ["bengaluru","bangalore"],
    "mangaluru": ["mangaluru","mangalore"],
    "mysuru": ["mysuru","mysore"],
    "belagavi": ["belagavi","belgaum"],
    "hubballi": ["hubballi","hubli"],
    "dharwad": ["dharwad"],
    "shivamogga": ["shivamogga","shimoga"],
    "tumakuru": ["tumakuru","tumkur"],
    "kalaburagi": ["kalaburagi","gulbarga"],
    "ballari": ["ballari","bellary"],
    "davangere": ["davangere"],
    "hassan": ["hassan"],
    "chikkamagaluru": ["chikkamagaluru","chikmagalur"],
    "chitradurga": ["chitradurga"],
    "bagalkot": ["bagalkot"],
    "vijayapura": ["vijayapura","bijapur"],
    "bidar": ["bidar"],
    "yadgir": ["yadgir","yadagiri"],
    "koppal": ["koppal"],
    "gadag": ["gadag"],
    "uttara kannada": ["uttara kannada","karwar","sirsi","dandeli"],
    "dakshina kannada": ["dakshina kannada","puttur","bantwal","sullia"],
    "udupi": ["udupi","manipal"],
    "kodagu": ["kodagu","madikeri"],
    "mandya": ["mandya"],
    "ramanagara": ["ramanagara","ramanagaram"],
    "chamarajanagar": ["chamarajanagar"],
    "kolar": ["kolar"],
    "chikkaballapura": ["chikkaballapura","chickballapur"],
    "raichur": ["raichur"],
}
ALIAS_TO_CANON = {a.lower(): canon for canon, aliases in CITY_ALIASES.items() for a in aliases}

CODE_RE = re.compile(r"^(E\d{3})\s+(.*)$")
YEAR_RE = re.compile(r"(19|20)\d{2}")
ROUND_MAP = {
    "mock": "Mock", "trial": "Mock",
    "1st": "R1", "r1": "R1", "round1": "R1", "round-1": "R1",
    "2nd": "R2", "r2": "R2", "round2": "R2", "round-2": "R2",
    "3rd": "R3", "r3": "R3", "ext": "R3", "ext rnd": "R3", "extended": "R3", "extra": "R3", "round3": "R3", "round-3": "R3",
}

def normalize_header(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()

def canon_city(text: str | None) -> str | None:
    if not text:
        return None
    low = text.lower()
    for alias, canon in ALIAS_TO_CANON.items():
        if alias in low:
            return canon
    return None

def split_name_city(name_city: str | None) -> tuple[str | None, str | None]:
    if not isinstance(name_city, str):
        return None, None
    parts = re.split(r"\s{2,}", name_city.strip())
    if len(parts) >= 2:
        name = "  ".join(parts[:-1]).strip(" -")
        city = re.sub(r"\s*\(.*?\)\s*", "", parts[-1]).strip()
    else:
        name, city = name_city.strip(), ""
    return name, city

def detect_year_and_round(pdf_path: Path) -> tuple[int | None, str | None]:
    fname = pdf_path.stem.lower()
    year = None
    m = YEAR_RE.search(fname)
    if m:
        year = int(m.group(0))
    if year is None:
        for parent in pdf_path.parents:
            m2 = YEAR_RE.search(parent.name)
            if m2:
                year = int(m2.group(0)); break
    round_guess = None
    for key, val in ROUND_MAP.items():
        if key in fname:
            round_guess = val; break
    return year, round_guess

def find_branch_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if re.search(r"\bBranch\b", str(c), flags=re.I):
            return c
    for c in list(df.columns)[2:8]:
        s = df[c].astype(str).str.lower()
        if (s.str.contains(r"\b(ai|cs|ec|ee|me|ce|cy|ds|bt|ar|et|ei|md|se|ae|au|cb|cd)\b", regex=True)).mean() > 0.05:
            return c
    return list(df.columns)[2]

def camelot_extract(pdf: Path) -> list[pd.DataFrame]:
    if camelot is None:
        return []
    # stream flavor works without Ghostscript; table areas vary per page, so we parse all and merge
    try:
        tables = camelot.read_pdf(str(pdf), pages="all", flavor="stream")
        return [t.df for t in tables] if tables else []
    except Exception:
        return []

def plumber_extract(pdf: Path) -> list[pd.DataFrame]:
    if pdfplumber is None:
        return []
    # fallback: basic extraction per page (lines), then split on 2+ spaces
    dfs = []
    try:
        with pdfplumber.open(pdf) as doc:
            for page in doc.pages:
                text = page.extract_text() or ""
                rows = []
                for line in text.splitlines():
                    line = re.sub(r"\u00a0", " ", line)
                    line = re.sub(r"\s+", " ", line).strip()
                    if not line:
                        continue
                    # crude split into "cells" by 2+ spaces (KCET PDFs use wide spacing)
                    cells = re.split(r"\s{2,}", line)
                    rows.append(cells)
                if not rows:
                    continue
                # normalize width
                maxw = max(len(r) for r in rows)
                rows = [r + [""]*(maxw-len(r)) for r in rows]
                df = pd.DataFrame(rows)
                dfs.append(df)
        return dfs
    except Exception:
        return []

def coerce_dataframe_list(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Try to build one big DataFrame that looks like the tabula output."""
    if not dfs:
        return pd.DataFrame()
    # Heuristic: if many frames with header row repeated, promote first row to header when it looks textual.
    cleaned = []
    for df in dfs:
        df = df.copy()
        df = df.replace({None: ""})
        # If first row has many strings and second row is numbers, set header
        if len(df) >= 2:
            s1 = (df.iloc[0].astype(str) != "").mean()
            s2 = pd.to_numeric(df.iloc[1], errors="coerce").isna().mean()
            # Keep as-is; we'll just concat and handle later
        cleaned.append(df)
    big = pd.concat(cleaned, ignore_index=True)
    # Create synthetic column names
    big.columns = [f"C{i}" for i in range(big.shape[1])]
    return big

def to_long(big: pd.DataFrame, year: int | None, round_guess: str | None, fname: str, write_dir: Path) -> tuple[Path, pd.DataFrame]:
    if big.empty:
        raise RuntimeError("No tables found (camelot/pdfplumber fallback).")

    # Build RowHeader as Col0 + Col1 (similar to tabula approach)
    big = big.rename(columns={big.columns[0]: "Col1"})
    if len(big.columns) > 1:
        big = big.rename(columns={big.columns[1]: "Col2"})
    big["Col1"] = big["Col1"].astype(str)
    big["Col2"] = big["Col2"].astype(str)
    big["RowHeader"] = (big["Col1"].fillna("") + "  " + big["Col2"].fillna("")).str.replace(r"\s+", " ", regex=True).str.strip()

    # Split CollegeCode & NameCity
    def split_code(s: str):
        m = CODE_RE.match(s)
        if not m:
            return pd.Series({"CollegeCode": None, "NameCity": s})
        return pd.Series({"CollegeCode": m.group(1), "NameCity": m.group(2)})

    tmp = big["RowHeader"].apply(split_code)
    big = pd.concat([big, tmp], axis=1)

    # Choose a branch column heuristically (use next column if available)
    branch_col = find_branch_col(big)
    helper = {"Col1", "Col2", "RowHeader", "CollegeCode", "NameCity", branch_col}
    cat_cols = [c for c in big.columns if c not in helper]

    # Keep rows that look like actual data
    wide = big.loc[big["CollegeCode"].notna(), ["CollegeCode", "NameCity", branch_col] + cat_cols].copy()
    wide = wide.rename(columns={branch_col: "Branch"})

    # Melt to long
    long = wide.melt(id_vars=["CollegeCode", "NameCity", "Branch"], var_name="Category", value_name="CutoffRank")

    # Clean ranks
    long["CutoffRank"] = (
        long["CutoffRank"].astype(str).str.strip().replace({"--": None, "nan": None, "NaN": None, "": None})
    )
    long["CutoffRankNum"] = pd.to_numeric(long["CutoffRank"], errors="coerce")

    # Split Name/City & canonicalize
    name_city = long["NameCity"].apply(lambda s: pd.Series({
        "CollegeName": split_name_city(s)[0],
        "CityRaw": split_name_city(s)[1]
    }))
    long = pd.concat([long.drop(columns=["NameCity"]), name_city], axis=1)
    long["City"] = long["CityRaw"].apply(canon_city)

    long["Year"] = year
    long["Round"] = round_guess

    out_csv = write_dir / f"{slugify(fname)}_LONG_WITH_CITY.csv"
    long[
        ["Year","Round","CollegeCode","CollegeName","CityRaw","City","Branch","Category","CutoffRank","CutoffRankNum"]
    ].to_csv(out_csv, index=False)

    return out_csv, long

def process_pdf(pdf: Path):
    year, round_guess = detect_year_and_round(pdf)
    write_dir = OUT_DIR / str(year) if (GROUP_OUT_BY_YEAR and year) else OUT_DIR
    write_dir.mkdir(exist_ok=True, parents=True)

    fname = pdf.stem
    dfs = camelot_extract(pdf)
    if not dfs:
        dfs = plumber_extract(pdf)
    big = coerce_dataframe_list(dfs)
    return to_long(big, year, round_guess, fname, write_dir)

def main():
    pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under: {DATA_DIR.resolve()}")
        sys.exit(1)

    all_frames = []
    ok = fail = 0
    for pdf in pdfs:
        try:
            csv_path, df_long = process_pdf(pdf)
            all_frames.append(df_long)
            ok += 1
            print(f"OK  → {pdf}  → {csv_path}")
        except Exception as e:
            fail += 1
            print(f"FAIL → {pdf} : {e}")

    if not all_frames:
        print("No tables parsed; aborting master build.")
        sys.exit(2)

    all_long = pd.concat(all_frames, ignore_index=True)

    def top1(s: pd.Series):
        vc = s.dropna().value_counts()
        return vc.index[0] if len(vc) else None

    master = (
        all_long.groupby("CollegeCode")
        .agg(
            CollegeName=("CollegeName", top1),
            City=("City", top1),
            CityRaw=("CityRaw", top1),
            FirstYear=("Year", "min"),
            LastYear=("Year", "max"),
        )
        .reset_index()
    )
    master["City"] = master["City"].fillna(master["CityRaw"])
    master["State"] = "Karnataka"
    master["District"] = None
    master["Latitude"] = None
    master["Longitude"] = None

    master_path = OUT_DIR / "KCET_College_Master_Locations.csv"
    master.to_csv(master_path, index=False)

    print("\nSummary:")
    print(f"  Parsed PDFs : {ok}")
    print(f"  Failed PDFs : {fail}")
    print(f"  Master saved → {master_path}")

if __name__ == "__main__":
    main()
