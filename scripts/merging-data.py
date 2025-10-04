# ---------------------------------------------------------------------------------
# KCET seat allotment merger for 2015–2025
# Output:
#   - KCET_cleaned.csv   -> Year, Round, College, Branch, Category, Closing Rank
#   - branch_catalog.csv -> distinct Branch values with frequency
# Log:
#   - merge_failures.log
# ---------------------------------------------------------------------------------

import os
import re
import sys
import pandas as pd

BASE_DIR = "datasets"            # must contain 2015..2025 subfolders
OUTPUT   = "KCET_cleaned.csv"
CATALOG  = "branch_catalog.csv"
LOG_FAIL = "merge_failures.log"

# ======= CONFIG ==================================================================

# Keep alias map EMPTY by default to avoid over-collapsing branches.
# Add ONLY explicit synonyms you absolutely want merged.
BRANCH_ALIAS = {
    # Examples (enable if you want strict merging):
    # "CSE": "COMPUTER SCIENCE AND ENGINEERING",
    # "COMPUTERS": "COMPUTER SCIENCE AND ENGINEERING",
    # "AI & DS": "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
}

# Acceptable categories; parser also allows tokens that match regex patterns below.
CATEGORY_TOKENS_HINT = {
    "1G","1K","1R","2AG","2AK","2AR","2BG","2BK","2BR","3AG","3AK","3AR","3BG","3BK","3BR",
    "GM","GMK","GMR","SCG","SCK","SCR","STG","STK","STR",
    # Hyd-Kar 2017 variants:
    "1H","1KH","1RH","2AH","2AKH","2ARH","2BH","2BKH","2BRH",
    "3AH","3AKH","3ARH","3BH","3BKH","3BRH",
    "GMH","GMKH","GMRH","SCH","SCKH","SCRH","STH","STKH","STRH",
    # 2025 extras:
    "GMP","NRI","OPN","OTH",
}

ROUND_STD = {"1":"1st","2":"2nd","3":"3rd"}
RGX_COLLEGE_CODE = re.compile(r"^[A-Z]\d{3}$")  # e.g., E001

# Rank sanity limit (rejects corrupted numbers like 104469228525)
MAX_RANK = 200000

# ======= UTILITIES ===============================================================

def norm(x):
    return str(x).strip() if pd.notna(x) else ""

def is_nullish(x):
    if pd.isna(x): return True
    s = str(x).strip().lower()
    return s in {"", "na", "n/a", "nil", "none", "--", "—", "nan"}

def likely_category_token(x):
    s = norm(x).upper()
    if not s: return False
    if s in CATEGORY_TOKENS_HINT: 
        return True
    # Loose patterns (be conservative)
    if re.match(r"^[123](?:G|K|R)$", s):   # 1G, 2K, 3R
        return True
    if re.match(r"^[23]A[GRK]$", s):       # 2AG, 3AK ...
        return True
    if re.match(r"^2B[GRK]$", s):          # 2BG, 2BK, 2BR
        return True
    if re.match(r"^(SC|ST)[GKRH]$", s):    # SCG, SCKH, etc.
        return True
    if re.match(r"^(GMP|NRI|OPN|OTH)$", s):
        return True
    return False

def std_round_from_filename(fname: str) -> str:
    stem = os.path.splitext(os.path.basename(fname))[0].lower()
    if "mock" in stem: return "mock"
    m = re.search(r"(?:^|[^0-9])([123])(st|nd|rd)?(?:[^0-9]|$)", stem)
    if m:
        return ROUND_STD.get(m.group(1), stem)
    if any(k in stem for k in ["round1","r1","first"]): return "1st"
    if any(k in stem for k in ["round2","r2","second"]): return "2nd"
    if any(k in stem for k in ["round3","r3","third","ext","final"]): return "3rd"
    return stem  # fallback

def clean_rank_val(v):
    """Return a clean integer rank or None. Remove commas/decimals; drop huge outliers."""
    if is_nullish(v): return None
    s = str(v).replace(",", "").strip()
    m = re.match(r"^\d+", s)  # integer part only
    if not m:
        return None
    val = int(m.group(0))
    if not (0 < val <= MAX_RANK):
        return None
    return val

def split_single_column(df: pd.DataFrame) -> pd.DataFrame:
    """If file collapsed into one column, split by 2+ spaces / tabs / pipes."""
    if df.shape[1] == 1:
        series = df.iloc[:,0].astype(str).str.replace(r"\t+", "  ", regex=True)
        wide = series.str.split(r"\s{2,}|\t+|\|", expand=True)
        return wide.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def row_has_many_category_tokens(row_vals, min_count=6):
    return sum(1 for x in row_vals if likely_category_token(x)) >= min_count

def find_category_header_row(df, start_idx):
    """Find a row that looks like a category header near start_idx."""
    for i in range(start_idx, min(start_idx+40, len(df))):
        vals = [norm(x) for x in df.iloc[i].tolist()]
        if sum(1 for v in vals if v) < 4:
            continue
        if row_has_many_category_tokens(vals, min_count=6):
            return i
    return None

def detect_college_row(vals):
    """
    2015–2024 style college row:
      [S.No, E001, Long College Name, ...] -> return (name, code)
    """
    if len(vals) < 3: return (None, None)
    code = norm(vals[1])
    name = norm(vals[2])
    if RGX_COLLEGE_CODE.match(code) and len(name) >= 3:
        return (name, code)
    return (None, None)

def infer_lead_cols_from_header_row(df, header_idx):
    """Number of columns before first category token."""
    row = [norm(x) for x in df.iloc[header_idx].tolist()]
    for j, v in enumerate(row):
        if likely_category_token(v):
            return j
    return 2

def normalize_branch(name: str) -> str:
    """
    Normalize and standardize branch names.
    Converts to uppercase, removes extra spaces, and applies standard naming.
    Special handling for AI/ML and AI/DS branches.
    """
    if not name or pd.isna(name):
        return ""
    
    # Clean up the input first
    name = str(name).strip().upper()
    
    # Common branch mappings - ordered from most specific to most general
    branch_mapping = {
        # AI/ML specific variations
        'AI & ML': 'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING',
        'AIML': 'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING',
        'AI/ML': 'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING',
        'AI-ML': 'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING',
        'AI ML': 'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING',
        'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING': 'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING',
        'ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING': 'ARTIFICIAL INTELLIGENCE & MACHINE LEARNING',
        'CS IN AI & ML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CSE AI & ML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CSE IN AI & ML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CSE (AI & ML)': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CS (AI & ML)': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CS AI ML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CS IN AIML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CSE AIML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CS AIML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CS IN ARTIFICIAL INTELLIGENCE & MACHINE LEARNING': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        
        # AI/DS specific variations
        'AI & DS': 'ARTIFICIAL INTELLIGENCE & DATA SCIENCE',
        'AI/DS': 'ARTIFICIAL INTELLIGENCE & DATA SCIENCE',
        'AI-DS': 'ARTIFICIAL INTELLIGENCE & DATA SCIENCE',
        'AI DS': 'ARTIFICIAL INTELLIGENCE & DATA SCIENCE',
        'AIDS': 'ARTIFICIAL INTELLIGENCE & DATA SCIENCE',
        'ARTIFICIAL INTELLIGENCE & DATA SCIENCE': 'ARTIFICIAL INTELLIGENCE & DATA SCIENCE',
        'ARTIFICIAL INTELLIGENCE AND DATA SCIENCE': 'ARTIFICIAL INTELLIGENCE & DATA SCIENCE',
        'CS IN AI & DS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CSE AI & DS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CSE IN AI & DS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CSE (AI & DS)': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CS (AI & DS)': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CS AI DS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CS IN AIDS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CSE AIDS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CS AIDS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CS IN ARTIFICIAL INTELLIGENCE & DATA SCIENCE': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        
        # General AI variations
        'AI': 'ARTIFICIAL INTELLIGENCE',
        'ARTIFICIAL INTELLIGENCE': 'ARTIFICIAL INTELLIGENCE',
        'CS IN AI': 'COMPUTER SCIENCE (ARTIFICIAL INTELLIGENCE)',
        'CSE AI': 'COMPUTER SCIENCE (ARTIFICIAL INTELLIGENCE)',
        'CS AI': 'COMPUTER SCIENCE (ARTIFICIAL INTELLIGENCE)',
        
        # Data Science variations
        'DS': 'DATA SCIENCE',
        'DATA SCIENCE': 'DATA SCIENCE',
        'CS IN DS': 'COMPUTER SCIENCE (DATA SCIENCE)',
        'CSE DS': 'COMPUTER SCIENCE (DATA SCIENCE)',
        'CS (DS)': 'COMPUTER SCIENCE (DATA SCIENCE)',
        'CSE (DS)': 'COMPUTER SCIENCE (DATA SCIENCE)',
        'CS IN DATA SCIENCE': 'COMPUTER SCIENCE (DATA SCIENCE)',
        'CSE IN DATA SCIENCE': 'COMPUTER SCIENCE (DATA SCIENCE)',
        
        # Machine Learning variations
        'ML': 'MACHINE LEARNING',
        'MACHINE LEARNING': 'MACHINE LEARNING',
        'CS IN ML': 'COMPUTER SCIENCE (MACHINE LEARNING)',
        'CSE ML': 'COMPUTER SCIENCE (MACHINE LEARNING)',
        'CS ML': 'COMPUTER SCIENCE (MACHINE LEARNING)',
        'CS IN MACHINE LEARNING': 'COMPUTER SCIENCE (MACHINE LEARNING)',
        'CSE IN MACHINE LEARNING': 'COMPUTER SCIENCE (MACHINE LEARNING)',
        
        # Computer Science base variations
        'CS': 'COMPUTER SCIENCE AND ENGINEERING',
        'CSE': 'COMPUTER SCIENCE AND ENGINEERING',
        'COMPUTERS': 'COMPUTER SCIENCE AND ENGINEERING',
        'CS COMPUTERS': 'COMPUTER SCIENCE AND ENGINEERING',
        'COMPUTER SCIENCE': 'COMPUTER SCIENCE AND ENGINEERING',
        'COMPUTER SCIENCE AND ENGINEERING': 'COMPUTER SCIENCE AND ENGINEERING',
        
        # Other common branches
        'CE': 'CIVIL ENGINEERING',
        'CIVIL': 'CIVIL ENGINEERING',
        'CE CIVIL': 'CIVIL ENGINEERING',
        'CIVIL ENGG': 'CIVIL ENGINEERING',
        'ME': 'MECHANICAL ENGINEERING',
        'MECH': 'MECHANICAL ENGINEERING',
        'MECHANICAL': 'MECHANICAL ENGINEERING',
        'MECHANICAL ENGG': 'MECHANICAL ENGINEERING',
        'ECE': 'ELECTRONICS AND COMMUNICATION ENGINEERING',
        'EC': 'ELECTRONICS AND COMMUNICATION ENGINEERING',
        'EEE': 'ELECTRICAL AND ELECTRONICS ENGINEERING',
        'EE': 'ELECTRICAL AND ELECTRONICS ENGINEERING',
        'ISE': 'INFORMATION SCIENCE AND ENGINEERING',
        'IT': 'INFORMATION TECHNOLOGY',
        
        # Civil Engineering
        'CE': 'CIVIL ENGINEERING',
        'CIVIL': 'CIVIL ENGINEERING',
        'CE CIVIL': 'CIVIL ENGINEERING',
        'CIVIL ENGG': 'CIVIL ENGINEERING',
        
        # Mechanical Engineering
        'ME': 'MECHANICAL ENGINEERING',
        'MECH': 'MECHANICAL ENGINEERING',
        'MECHANICAL': 'MECHANICAL ENGINEERING',
        'MECHANICAL ENGG': 'MECHANICAL ENGINEERING',
        
        # Electronics and Communication Engineering
        'ECE': 'ELECTRONICS AND COMMUNICATION ENGINEERING',
        'EC': 'ELECTRONICS AND COMMUNICATION ENGINEERING',
        'ELECTRONICS': 'ELECTRONICS AND COMMUNICATION ENGINEERING',
        'ELECTRONICS ENGG': 'ELECTRONICS AND COMMUNICATION ENGINEERING',
        
        # Electrical and Electronics Engineering
        'EEE': 'ELECTRICAL AND ELECTRONICS ENGINEERING',
        'EE': 'ELECTRICAL AND ELECTRONICS ENGINEERING',
        'ELECTRICAL': 'ELECTRICAL AND ELECTRONICS ENGINEERING',
        'ELECTRICAL ENGG': 'ELECTRICAL AND ELECTRONICS ENGINEERING',
        
        # Information Science & Engineering
        'ISE': 'INFORMATION SCIENCE AND ENGINEERING',
        'IT': 'INFORMATION TECHNOLOGY',
        'INFT': 'INFORMATION TECHNOLOGY',
        'INFORMATION SCIENCE': 'INFORMATION SCIENCE AND ENGINEERING',
        
        # Other common branches
        'AERO': 'AERONAUTICAL ENGINEERING',
        'AERONAUTICAL': 'AERONAUTICAL ENGINEERING',
        'AUTO': 'AUTOMOBILE ENGINEERING',
        'AUTOMOBILE': 'AUTOMOBILE ENGINEERING',
        'BT': 'BIOTECHNOLOGY',
        'BIOTECH': 'BIOTECHNOLOGY',
        'BIOTECHNOLOGY': 'BIOTECHNOLOGY',
        'CHEM': 'CHEMICAL ENGINEERING',
        'CHEMICAL': 'CHEMICAL ENGINEERING',
        'CIVIL ENV': 'CIVIL AND ENVIRONMENTAL ENGINEERING',
        'CSE AIML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CSE AI&ML': 'COMPUTER SCIENCE (AI & MACHINE LEARNING)',
        'CSE DS': 'COMPUTER SCIENCE (DATA SCIENCE)',
        'CSE AI&DS': 'COMPUTER SCIENCE (AI & DATA SCIENCE)',
        'CSE CYBERSECURITY': 'COMPUTER SCIENCE (CYBERSECURITY)',
        'CSE IOT': 'COMPUTER SCIENCE (IOT)',
    }
    
    # Clean up the input
    name = str(name).strip().upper()
    
    # Remove any trailing numbers or special characters
    name = re.sub(r'\s*\d+\s*$', '', name)  # Remove trailing numbers
    name = re.sub(r'[^A-Z0-9\s&(),-]', ' ', name)  # Keep only alphanumeric, spaces, &, (), -
    name = re.sub(r'\s+', ' ', name).strip()  # Normalize spaces
    
    # Check if we have a direct mapping
    if name in branch_mapping:
        return branch_mapping[name]
    
    # Check for partial matches
    for key, value in branch_mapping.items():
        if key in name:
            return value
    
    # If no mapping found, return the cleaned name
    return re.sub(r"\s+", " ", name).strip()

def make_branch_name_from_leads(cells):
    """
    Build branch from leading columns (before categories) in matrix tables.
    Example: ["CS", "Computers"] -> "COMPUTERS"
    """
    if not cells:
        return ""
    
    # The first non-empty cell is the branch code (e.g., "CS", "CE")
    branch_code = next((c for c in cells if norm(c)), "")
    
    # If there's a second non-empty cell, it's the full branch name (e.g., "Computers", "Civil")
    branch_name = next((c for c in cells[1:] if norm(c)), "")
    
    # Prefer the full name if available, otherwise use the code
    if branch_name:
        return normalize_branch(branch_name)
    return normalize_branch(branch_code)

def row_is_all_empty(iterable):
    return all(not norm(x) for x in iterable)

def safe_year(folder_name: str) -> int:
    m = re.search(r"\d{4}", folder_name)
    return int(m.group(0)) if m else int(folder_name)

def preview(df: pd.DataFrame, n=8) -> str:
    try:
        return df.head(n).astype(str).to_string(index=False)
    except Exception:
        return "<preview unavailable>"

# ======= 2025 PARSER (block: 'College:' + 'Course Name' header) ==================

def parse_blockwise_2025(path, round_name, fail_log_acc):
    raw0 = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    # Skip splash-only files
    first_block = " ".join(str(x) for x in raw0.head(6).iloc[:,0].tolist()).lower()
    if "karnataka examinations authority" in first_block:
        fail_log_acc.append(f"[SKIP_NON_DATA] {path}")
        return []

    raw = split_single_column(raw0).fillna("")

    out = []
    current_college = None
    category_headers = []
    expecting_header = False

    # For stitching multi-line branch names:
    last_rows_idx = []   # indices in `out` for the last branch emitted
    last_branch    = ""  # last branch string we emitted

    i = 0
    while i < len(raw):
        row = [norm(x) for x in raw.iloc[i].tolist()]
        first_cell_up = row[0].upper()

        # Detect "College:" line
        if first_cell_up.startswith("COLLEGE:"):
            current_college = row[0].split(":", 1)[1].strip()
            category_headers = []
            expecting_header = True
            last_rows_idx = []
            last_branch = ""
            i += 1
            continue

        # Detect "Course Name" header line
        if expecting_header and first_cell_up == "COURSE NAME":
            category_headers = [norm(c).upper() for c in row[1:] if norm(c)]
            expecting_header = False
            last_rows_idx = []
            last_branch = ""
            i += 1
            continue

        # If still expecting header, skip filler rows
        if expecting_header:
            i += 1
            continue

        # Data rows (course lines) – require a college and a header
        if current_college and category_headers and first_cell_up and first_cell_up != "COURSE NAME":
            # Any non-empty value beyond first col?
            has_any = any(norm(v) not in {"", "NONE"} for v in row[1:])
            if has_any:
                course_name = normalize_branch(row[0])
                added = []
                for idx, cat in enumerate(category_headers, start=1):
                    if idx >= len(row):
                        continue
                    rank = clean_rank_val(row[idx])
                    if rank is None:
                        continue
                    # Extract college code from the current_college string if available
                    college_code = ""
                    college_name = re.sub(r"\s+", " ", current_college).strip()
                    
                    # Look for a pattern like "(Code: XXXX)" in the college name
                    code_match = re.search(r'\(Code:\s*([A-Z0-9]+)\)', college_name, re.IGNORECASE)
                    if code_match:
                        college_code = code_match.group(1)
                        # Remove the code from the college name
                        college_name = re.sub(r'\s*\(Code:\s*[A-Z0-9]+\)\s*', ' ', college_name).strip()
                    
                    out.append({
                        "Year": 2025,
                        "Round": round_name,
                        "College": college_name,
                        "College Code": college_code,
                        "Branch": course_name,
                        "Category": cat,
                        "Closing Rank": rank
                    })
                    added.append(len(out) - 1)
                last_rows_idx = added
                last_branch = course_name
            else:
                # Continuation line for the branch name
                cont = norm(row[0])
                if cont and last_rows_idx:
                    patched = normalize_branch(f"{last_branch} {cont}")
                    for idx_out in last_rows_idx:
                        out[idx_out]["Branch"] = patched
                    last_branch = patched

            i += 1
            continue

        i += 1

    return out

def parse_matrix_year_file(fpath, round_std, year_num, fail_log_acc):
    """
    Parse matrix-style files (2015-2024) where branches are in rows.
    Format:
        [S.No, College Code, College Name, ...]
        [Category Headers...]
        [Branch1, rank1, rank2, ...]
        [Branch2, rank1, rank2, ...]
    """
    try:
        raw0 = pd.read_csv(fpath, header=None, dtype=str, low_memory=False)
    except Exception as e:
        fail_log_acc.append(f"[READ_ERROR] {fpath}: {str(e)}")
        return []
        
    # Skip splash-only pages
    first_block = " ".join(str(x) for x in raw0.head(6).iloc[:,0].tolist()).lower()
    if "karnataka examinations authority" in first_block:
        fail_log_acc.append(f"[SKIP_NON_DATA] {fpath}")
        return []

    # Extract the round from the filename if not provided
    if isinstance(round_std, int) or (isinstance(round_std, str) and round_std.isdigit()):
        round_num = int(round_std)
        if 1 <= round_num <= 3:
            round_std = ["1st", "2nd", "3rd"][round_num - 1]
        else:
            round_std = f"Round {round_num}"
    
    raw = split_single_column(raw0)
    out_records = []
    current_college = None
    category_headers = None

    i = 0
    while i < len(raw):
        row = [norm(x) for x in raw.iloc[i].tolist()]
        
        # Detect college row (starts with a number and has a college code)
        if len(row) > 2 and row[0].isdigit() and RGX_COLLEGE_CODE.match(norm(row[1])):
            current_college = row[2]  # College name
            i += 1  # Move to the next row (category headers)
            
            # Get category headers (next row after college)
            if i < len(raw):
                category_headers = [norm(x) for x in raw.iloc[i].tolist()]
                i += 1  # Move to data rows
                
                # Process data rows until next college or end of file
                while i < len(raw):
                    data_row = [norm(x) for x in raw.iloc[i].tolist()]
                    
                    # Check if this is a new college or end of data
                    if len(data_row) > 0 and data_row[0].isdigit() and RGX_COLLEGE_CODE.match(norm(data_row[1])):
                        break
                    
                    # Skip empty rows or rows that don't have enough columns
                    if len(data_row) < 2 or not any(data_row):
                        i += 1
                        continue
                    
                    # First column is the branch name
                    branch_name = data_row[0]
                    
                    # Process each rank in the row (skipping the first column which is the branch name)
                    for j in range(1, min(len(data_row), len(category_headers))):
                        rank = clean_rank_val(data_row[j])
                        if rank is not None and category_headers[j]:  # Valid rank and category
                            category = category_headers[j]
                            
                            # Only process if it's a valid category
                            if likely_category_token(category):
                                # Extract college code from the row if available
                                college_code = row[1] if len(row) > 1 and RGX_COLLEGE_CODE.match(norm(row[1])) else ""
                    
                                out_records.append({
                                    "Year": year_num,
                                    "Round": round_std,
                                    "College": current_college,
                                    "College Code": college_code,
                                    "Branch": normalize_branch(branch_name),
                                    "Category": category,
                                    "Closing Rank": rank
                                })
                    i += 1
        else:
            i += 1

    return out_records

def main():
    all_data = []
    fail_log = []
    
    # Process each year's data
    for year in range(2015, 2026):  # 2015 to 2025
        year_dir = os.path.join(BASE_DIR, str(year))
        if not os.path.exists(year_dir):
            print(f"Skipping missing year: {year}")
            continue
            
        print(f"\nProcessing year: {year}")
        
        # Process each round file in the year directory
        for filename in os.listdir(year_dir):
            if not filename.lower().endswith('.csv'):
                continue
                
            filepath = os.path.join(year_dir, filename)
            print(f"  - {filename}")
            
            # Determine round from filename
            # First try to match patterns like '1st-2015.csv', '2nd-2015.csv', etc.
            round_match = re.search(r'^(\d+)(?:st|nd|rd|th)?[^0-9]', filename.lower())
            if not round_match:
                # If no match, try to find any number in the filename
                round_match = re.search(r'(\d+)', filename)
            
            round_num = round_match.group(1) if round_match else "1"
            
            # Convert round number to proper format (1st, 2nd, 3rd, etc.)
            try:
                round_num_int = int(round_num)
                if round_num_int == 1:
                    round_std = "1st"
                elif round_num_int == 2:
                    round_std = "2nd"
                elif round_num_int == 3:
                    round_std = "3rd"
                else:
                    round_std = f"Round {round_num_int}"
            except (ValueError, TypeError):
                # If we can't convert to int, use the original string
                round_std = str(round_num)
            
            try:
                if year == 2025:
                    # Use blockwise parser for 2025
                    records = parse_blockwise_2025(filepath, round_std, fail_log)
                else:
                    # Use matrix parser for 2015-2024
                    records = parse_matrix_year_file(filepath, round_std, year, fail_log)
                
                if records:
                    all_data.extend(records)
            except Exception as e:
                fail_log.append(f"[ERROR] {filepath}: {str(e)}")
    
    # Create DataFrame and clean up
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Ensure consistent column order with College Code
        df = df[['Year', 'Round', 'College', 'College Code', 'Branch', 'Category', 'Closing Rank']]
        
        # Sort the data
        df = df.sort_values(by=['Year', 'Round', 'College', 'Branch', 'Category'])
        
        # Save the cleaned data
        df.to_csv(OUTPUT, index=False)
        print(f"\n✅ Saved {len(df):,} records to {OUTPUT}")
        
        # Generate branch catalog
        branch_counts = df['Branch'].value_counts().reset_index()
        branch_counts.columns = ['Branch', 'Count']
        branch_counts = branch_counts.sort_values('Branch')
        branch_counts.to_csv(CATALOG, index=False)
        print(f"✅ Saved branch catalog to {CATALOG}")
    
    # Log any failures
    if fail_log:
        with open(LOG_FAIL, 'w', encoding='utf-8') as f:
            f.write("\n".join(fail_log))
        print(f"⚠️  Some files had issues. See {LOG_FAIL}")

if __name__ == "__main__":
    main()
