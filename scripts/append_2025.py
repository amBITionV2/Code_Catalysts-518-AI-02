# scripts/append_2025.py
import os
import re
import pandas as pd

DATA_DIR = "datasets/2025"
MASTER_FILE = "KCET_cleaned.csv"
OUTPUT_FILE = "KCET_cleaned_with_2025.csv"
LOG_FILE = "append_2025_failures.log"

ROUND_MAP = {
    "1st-2025.csv": "1st",
    "2nd-2025.csv": "2nd",
    "3rd-2025.csv": "3rd",
    "mock-2025.csv": "mock",
}

# ---------------- utils ----------------

def norm(s):
    return str(s).strip() if pd.notna(s) else ""

def clean_rank(x):
    """Convert ranks to int, remove outliers & garbage values."""
    if pd.isna(x): 
        return None
    s = str(x).strip()
    if s.lower() in {"", "na", "n/a", "nan", "nil", "none"}:
        return None
    if s in {"-", "—", "--"}:
        return None
    s = s.replace(",", "")
    # take integer part only
    m = re.match(r"^\d+", s)
    if not m:
        return None
    v = int(m.group(0))
    # drop outliers beyond 200k
    if v <= 0 or v > 200000:
        return None
    return v

def split_single_column(df: pd.DataFrame) -> pd.DataFrame:
    """If the CSV collapsed into one column, split on 2+ spaces / tabs / pipes."""
    if df.shape[1] == 1:
        series = df.iloc[:, 0].astype(str).str.replace(r"\t+", "  ", regex=True)
        wide = series.str.split(r"\s{2,}|\t+|\|", expand=True)
        return wide.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def row_has_any_value(vals):
    """Is there any non-empty value beyond the first column?"""
    for v in vals[1:]:
        if norm(v) not in {"", "None"}:
            return True
    return False

# ---------------- core parser ----------------

def parse_blockwise_2025(path, round_name):
    raw0 = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    raw = split_single_column(raw0).fillna("")

    records = []
    current_college = None
    category_headers = []
    expecting_header = False
    last_data_row_index = None

    i = 0
    while i < len(raw):
        row = [norm(x) for x in raw.iloc[i].tolist()]
        first = row[0].lower()

        if first.startswith("college:"):
            # Extract college name and code from format: "College: E001 College Name"
            college_info = row[0].split(":", 1)[1].strip()
            
            # Extract college code (E followed by digits)
            code_match = re.search(r'\bE\d+\b', college_info)
            current_college_code = code_match.group(0) if code_match else ""
            
            # Remove the code from the college name
            current_college = re.sub(r'\bE\d+\b', '', college_info).strip()
            
            # Store the current college code in a variable that will be used when creating records
            global current_college_code_global
            current_college_code_global = current_college_code
            
            category_headers = []
            expecting_header = True
            last_data_row_index = None
            i += 1
            continue

        if expecting_header and row[0].lower() == "course name":
            category_headers = [c for c in row[1:] if c]
            expecting_header = False
            last_data_row_index = None
            i += 1
            continue

        if expecting_header:
            i += 1
            continue

        if current_college and category_headers and row[0] and row[0].lower() != "course name":
            if row_has_any_value(row):
                course_name = row[0]
                for idx, cat in enumerate(category_headers, start=1):
                    if idx >= len(row):
                        continue
                    rank = clean_rank(row[idx])
                    if rank is None:
                        continue
                    # Use the global college code that was extracted from the College line
                    college_code = current_college_code_global if 'current_college_code_global' in globals() else ""
                    
                    records.append({
                        "Year": 2025,
                        "Round": round_name,
                        "College": current_college,
                        "College Code": college_code,
                        "Branch": course_name,
                        "Category": cat.strip().upper(),
                        "Closing Rank": rank,
                    })
                last_data_row_index = len(records) - 1
            else:
                continuation = row[0]
                if continuation and last_data_row_index is not None:
                    base_idx = last_data_row_index
                    if base_idx >= 0:
                        last_branch = records[base_idx]["Branch"]
                        new_branch = (last_branch + " " + continuation).strip()
                        j = base_idx
                        while j >= 0:
                            r = records[j]
                            if (
                                r["Year"] == 2025
                                and r["Round"] == round_name
                                and r["College"] == current_college
                                and r["Branch"] == last_branch
                            ):
                                r["Branch"] = new_branch
                                j -= 1
                            else:
                                break
            i += 1
            continue

        i += 1

    if not records:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[NO_RECORDS] {path}\n")
            f.write(raw.head(40).to_string(index=False) + "\n" + ("-"*80) + "\n")
        raise ValueError("No data rows parsed from 2025 file")

    df = pd.DataFrame.from_records(records)

    # ---------------- normalization ----------------
    # Normalize College
    df["College"] = (
        df["College"]
        .str.replace(r"\(.*?\)", "", regex=True)   # remove bracketed text
        .str.replace(r"\s+", " ", regex=True)        # normalize spaces
        .str.strip()
        .str.upper()
    )

    # Normalize Branch
    branch_map = {
        # AI/ML variations
        "AI & ML": "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING",
        "AIML": "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING",
        "AI/ML": "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING",
        "AI-ML": "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING",
        "AI & MACHINE LEARNING": "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING",
        
        # AI/DS variations
        "AI & DS": "ARTIFICIAL INTELLIGENCE & DATA SCIENCE",
        "AI/DS": "ARTIFICIAL INTELLIGENCE & DATA SCIENCE",
        "AIDS": "ARTIFICIAL INTELLIGENCE & DATA SCIENCE",
        "AI & DATA SCIENCE": "ARTIFICIAL INTELLIGENCE & DATA SCIENCE",
        
        # General AI and ML
        "AI": "ARTIFICIAL INTELLIGENCE",
        "ML": "MACHINE LEARNING",
        
        # Data Science
        "DS": "DATA SCIENCE",
        
        # Computer Science variations
        "CSE": "COMPUTER SCIENCE AND ENGINEERING",
        "CS": "COMPUTER SCIENCE AND ENGINEERING",
        "COMPUTER SCIENCE": "COMPUTER SCIENCE AND ENGINEERING",
        "COMPUTERS": "COMPUTER SCIENCE AND ENGINEERING",
        
        # Other common abbreviations
        "ISE": "INFORMATION SCIENCE AND ENGINEERING",
        "ECE": "ELECTRONICS AND COMMUNICATION ENGINEERING",
        "EEE": "ELECTRICAL AND ELECTRONICS ENGINEERING",
        "MECH": "MECHANICAL ENGINEERING",
        "CIVIL": "CIVIL ENGINEERING",
        "AERO": "AERONAUTICAL ENGINEERING",
    }
    
    # Clean up branch names
    df["Branch"] = (
        df["Branch"]
        .str.replace(r"\s+", " ", regex=True)  # Normalize spaces
        .str.strip()
        .str.upper()
        .replace(branch_map, regex=True)  # Apply replacements
    )
    
    # Clean up any remaining special characters
    df["Branch"] = df["Branch"].str.replace(r"[^A-Z0-9& ]", "", regex=True)

    # Ensure consistent column order
    return df[["Year", "Round", "College", "College Code", "Branch", "Category", "Closing Rank"]]

# ---------------- main ----------------

def main():
    # Initialize global variable for college code
    global current_college_code_global
    current_college_code_global = ""
    
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    frames = []
    for fname, rname in ROUND_MAP.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"⚠️ Missing {fname}, skipping")
            continue
        try:
            df = parse_blockwise_2025(fpath, rname)
            print(f"✅ Parsed {fname} -> {len(df)} rows")
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Failed {fname}: {e}")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[FAIL] {fpath} :: {e}\n")

    if not frames:
        print("❌ No 2025 data processed.")
        if os.path.exists(LOG_FILE):
            print(f"📝 See log: {LOG_FILE}")
        return

    df_2025 = pd.concat(frames, ignore_index=True)

    # Ensure College Code column exists and is string type
    df_2025["College Code"] = df_2025["College Code"].astype(str)
    
    # Append to master
    if os.path.exists(MASTER_FILE):
        master = pd.read_csv(MASTER_FILE)
        # Ensure College Code column exists in master
        if "College Code" not in master.columns:
            master["College Code"] = ""
        master["College Code"] = master["College Code"].astype(str)
        combined = pd.concat([master, df_2025], ignore_index=True)
    else:
        print(f"⚠️ {MASTER_FILE} not found; writing only 2025 rows.")
        combined = df_2025

    # Final cleanup: drop duplicates
    combined = combined.drop_duplicates()

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Appended 2025 data. Final rows = {len(combined)}")
    print(f"💾 Saved -> {OUTPUT_FILE}")
    if os.path.exists(LOG_FILE):
        print(f"📝 Any issues were logged to: {LOG_FILE}")

if __name__ == "__main__":
    main()
