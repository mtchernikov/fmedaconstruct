import streamlit as st
import pandas as pd
import re, io, csv, json

# ---------------------------- App config ----------------------------
st.set_page_config(page_title="FMEDA Builder (KiCad + Failure DB)", layout="wide")
OPENAI_MODEL = "gpt-4o-mini"  # optional; only used when toggled

# Optional OpenAI client (needs st.secrets["OPENAI_API_KEY"])
try:
    from openai import OpenAI
    OAI = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
except Exception:
    OAI = None


# ---------------------------- helpers ----------------------------
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (str(s) if s is not None else "").strip().lower())


# ============================ KiCad 9 schematic parser ============================
REF_PREFIX = {
    "R":"Resistor","C":"Capacitor","L":"Inductor","D":"Diode","Z":"Zener",
    "Q":"MOSFET","T":"BJT","U":"IC_Analog","A":"OpAmp","K":"Relay","F":"Fuse",
    "J":"Connector","X":"Connector",
}

def detect_type_from_ref(ref: str):
    m = re.match(r'^([A-Za-z]+)', ref or "")
    if not m: return None
    pref = m.group(1).upper()
    for p in sorted(REF_PREFIX, key=lambda x: -len(x)):
        if pref.startswith(p): return REF_PREFIX[p]
    return None

def detect_type_from_lib(lib_id: str):
    s = (lib_id or "").lower()
    if "opamp" in s or "amplifier_operational" in s: return "OpAmp"
    if ":cp" in s or "electroly" in s: return "Capacitor_Polarized"
    if ":c" in s or "capacitor" in s: return "Capacitor"
    if ":r" in s or "resistor" in s: return "Resistor"
    if "lm393" in s or "comparator" in s: return "Comparator"
    if "mos" in s or "nmos" in s or "pmos" in s or "irf" in s: return "MOSFET"
    if "diode" in s: return "Diode"
    if "relay" in s: return "Relay"
    if "fuse" in s: return "Fuse"
    return None

def detect_type_from_value(val: str):
    t = (val or "").lower()
    if re.search(r'(^|\s)\d+(\.\d+)?(r|k|m)?(\s*ohm|$)', t): return "Resistor"
    if re.search(r'(^|\s)\d+(\.\d+)?(n|u|µ|p|f)(\s*|$)', t): return "Capacitor"
    if "ne5532" in t or "lm358" in t: return "OpAmp"
    if "irf" in t or "irfp" in t: return "MOSFET"
    return None

def _iter_blocks(text: str, tag: str):
    needle = f"({tag}"
    i = text.find(needle)
    n = len(text)
    while i != -1:
        depth = 0; start = i; j = i
        while j < n:
            ch = text[j]
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    yield text[start:j+1]; break
            j += 1
        i = text.find(needle, j)

def parse_kicad_sch_components(sch_text: str) -> pd.DataFrame:
    rows = []
    for blk in _iter_blocks(sch_text, "symbol"):
        m_lib = re.search(r'\(lib_id\s+"([^"]+)"\)', blk)
        m_ref = re.search(r'\(property\s+"Reference"\s+"([^"]*)"', blk)
        m_val = re.search(r'\(property\s+"Value"\s+"([^"]*)"', blk)
        lib_id = m_lib.group(1) if m_lib else ""
        ref = (m_ref.group(1) if m_ref else "").strip()
        val = (m_val.group(1) if m_val else "").strip()

        t  = detect_type_from_ref(ref)
        t2 = detect_type_from_lib(lib_id) or detect_type_from_value(val)
        if t == "IC_Analog" and t2: t = t2
        elif t2 and t != t2:
            if t in ("IC_Analog","Other") or (t == "Capacitor" and t2 == "Capacitor_Polarized"): t = t2

        ctype = t or "Other"
        if not ref: ref = "?"
        rows.append({"RefDes": ref, "ComponentType": ctype})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["RefDes","ComponentType","ct_norm"])
    df.loc[df["RefDes"] == "?", "ComponentType"] = "unknown component"
    df["ct_norm"] = df["ComponentType"].map(_norm)
    return df[["RefDes","ComponentType","ct_norm"]]


# ============================ Failure DB parsing ============================
_num_pat = re.compile(r'([-+]?\d*[\.,]?\d+(?:[eE][-+]?\d+)?)')

def parse_number_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    v = s.astype(str).replace({"None":"", "none":"", "nan":""})
    v = v.str.extract(_num_pat, expand=True)[0]
    v = v.str.replace(",", ".", regex=False)
    return pd.to_numeric(v, errors="coerce")

def parse_share_series(s: pd.Series) -> pd.Series:
    raw  = s.astype(str).replace({"None":"", "none":"", "nan":""})
    nums = parse_number_series(raw)
    is_pct = raw.str.contains("%")
    nums = nums.where(~is_pct, nums / 100.0)
    maxv = pd.to_numeric(nums, errors="coerce").max(skipna=True)
    if pd.notna(maxv) and maxv > 1.5:
        nums = nums / 100.0
    return nums

def auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-map Probability/Share and FIT variants to canonical names."""
    ren = {}
    have_share = any(_norm(c) == "share" for c in df.columns)
    have_fit   = any(_norm(c) == "fit"   for c in df.columns)

    for c in df.columns:
        n = _norm(c)  # e.g. "probability(%)" -> "probability"
        if not have_share:
            if ("probabil" in n) or (n in {"probability","probabilitypct","probabilitypercent"}) or ("share" in n):
                ren[c] = "Share"
                have_share = True
        if not have_fit:
            if n in {"fit","basefit","fitbasefit","lambdafit","lambda","failurerate","rate"}:
                ren[c] = "FIT"
                have_fit = True
    return df.rename(columns=ren) if ren else df

def parse_failure_csv_with_mapping(upload):
    """Return: (normalized_table, raw_dataframe, share_col_name, fit_col_name)"""
    raw_bytes = upload.read(); upload.seek(0)

    # delimiter sniff
    sample = raw_bytes[:4096].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        sep_guess = dialect.delimiter
    except Exception:
        sep_guess = None

    def _read(sep):
        return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, engine="python", dtype=str)

    if sep_guess:
        raw = _read(sep_guess)
    else:
        try:
            raw = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", dtype=str)
        except Exception:
            raw = _read(",")

    # header packed by ';'?
    if raw.shape[1] == 1 and ";" in raw.columns[0]:
        cols = [c.strip() for c in raw.columns[0].split(";")]
        tmp = raw.iloc[:,0].str.split(";", expand=True)
        if tmp.shape[1] == len(cols):
            tmp.columns = cols; raw = tmp

    # auto-rename common variants
    raw = auto_rename_columns(raw)
    st.write("CSV columns (auto-renamed if needed):", list(raw.columns))

    # column detection (prefer exact names now that we renamed)
    FIT_ALIASES   = ["FIT", "FIT (Base FIT)", "Base FIT"]
    SHARE_ALIASES = ["Share", "Share (Probability)", "Probability"]

    def pick_by_alias(aliases, cols):
        for a in aliases:
            if a in cols:
                return a
        return None

    norm = {c: _norm(c) for c in raw.columns}
    def find_col(cands):
        for orig, n in norm.items():
            if n in cands:
                return orig
        return None

    c_fit   = pick_by_alias(FIT_ALIASES,   list(raw.columns)) or find_col({"fit","lambda","rate"})
    c_share = pick_by_alias(SHARE_ALIASES, list(raw.columns)) or find_col({"share","percent","distribution"})
    c_type  = find_col({"componenttype","type","class","category","parttype","compclass"})
    c_mode  = find_col({"failuremode","mode","fm"})
    c_dc    = find_col({"dc","diagnosticcoverage","coverage","diagcoverage"})
    c_det   = find_col({"detectable","detected","isdetected"})
    c_dnm   = find_col({"diagnosticname","diag","diagnostic"})

    missing = [n for n,c in {"Component Type":c_type,"Failure Mode":c_mode,"Mode Share":c_share,"FIT":c_fit}.items() if c is None]
    if missing:
        st.warning("Map missing CSV columns:")
        cols = list(raw.columns)
        c_type = st.selectbox("Component Type column", cols, index=0 if c_type is None else cols.index(c_type))
        c_mode = st.selectbox("Failure Mode column",  cols, index=0 if c_mode is None else cols.index(c_mode))
        c_share= st.selectbox("Mode Share column",    cols, index=0 if c_share is None else cols.index(c_share))
        c_fit  = st.selectbox("FIT column",           cols, index=0 if c_fit is None else cols.index(c_fit))
        c_dc   = st.selectbox("Diagnostic Coverage (optional)", ["<none>"]+cols, index=0)
        c_det  = st.selectbox("Detectable (optional)",          ["<none>"]+cols, index=0)
        c_dnm  = st.text_input("Diagnostic Name (optional)", value=c_dnm or "")
        c_dc=None if c_dc=="<none>" else c_dc
        c_det=None if c_det=="<none>" else c_det
        c_dnm=None if not c_dnm else c_dnm

    # canonicalize ComponentType like "Resistor;Passive;Fixed …"
    def canon_type_cell(s: str) -> str:
        parts = [p.strip() for p in str(s).split(";") if p.strip()]
        if not parts: return ""
        base = {"resistor","capacitor","capacitor_polarized","inductor","diode","zener",
                "bjt","mosfet","opamp","comparator","relay","fuse","connector",
                "voltageregulator","microcontroller"}
        for p in parts:
            if _norm(p) in base: return p.capitalize()
        return parts[0]

    out = pd.DataFrame()
    out["ComponentType"] = raw[c_type].astype(str).fillna("").map(canon_type_cell)
    out["FailureMode"]   = raw[c_mode].astype(str).fillna("").str.strip()
    out["Share"]         = parse_share_series(raw[c_share])
    out["FIT"]           = parse_number_series(raw[c_fit])
    out["DC"]            = parse_number_series(raw[c_dc]).fillna(0.0).clip(0,1) if c_dc else 0.0
    if c_det:
        v = raw[c_det].astype(str).str.strip().str.lower()
        out["Detectable"] = v.isin(["1","true","yes","y","t"])
    else:
        out["Detectable"] = False
    out["DiagnosticName"] = raw[c_dnm].astype(str) if (c_dnm and c_dnm in raw.columns) else ""
    out["ct_norm"]        = out["ComponentType"].map(_norm)

    st.success("Failure DB parsed.")
    st.dataframe(out.head(20), height=240)

    return out, raw, c_share, c_fit


# --------------------------- Debug helper ---------------------------
def debug_numeric_preview(raw_df: pd.DataFrame, col_share: str, col_fit: str, n: int = 12):
    rs = raw_df[col_share].astype(str)
    rf = raw_df[col_fit].astype(str)
    ps = parse_share_series(rs)
    pf = parse_number_series(rf)

    mask = rs.replace({"None":"", "nan":""}).str.strip().ne("") | \
           rf.replace({"None":"", "nan":""}).str.strip().ne("")
    idx = raw_df.index[mask][:n]

    dbg = pd.DataFrame({
        "Share_raw":  rs.loc[idx].values,
        "Share_parsed": ps.loc[idx].values,
        "FIT_raw":    rf.loc[idx].values,
        "FIT_parsed": pf.loc[idx].values,
    }, index=idx)
    st.write("🔎 Parsing samples (first non-empty rows):")
    st.dataframe(dbg, height=240)


# ============================ FMEDA expand (keeps RefDes first) ============================
def expand_fmeda(components_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    if "ct_norm" not in components_df.columns:
        components_df["ct_norm"] = components_df["ComponentType"].map(_norm)
    if "ct_norm" not in failure_df.columns:
        failure_df["ct_norm"] = failure_df["ComponentType"].map(_norm)

    merged = components_df.merge(failure_df, how="left", on="ct_norm", suffixes=("_sch","_db"))
    unknown = merged["ComponentType_db"].isna() | (merged["ComponentType_sch"].str.lower()=="unknown component")

    out = pd.DataFrame({
        "RefDes":        merged["RefDes"],
        "ComponentType": merged["ComponentType_sch"].mask(unknown, "unknown component"),
        "FailureMode":   merged["FailureMode"].where(~unknown, pd.NA),
        "Share":         merged["Share"].where(~unknown, pd.NA),
        "FIT":           merged["FIT"].where(~unknown, pd.NA),
        "DC":            merged["DC"].where(~unknown, pd.NA),
        "Detectable":    merged["Detectable"].where(~unknown, False),
        "DiagnosticName":merged["DiagnosticName"].where(~unknown, "")
    })
    return out.sort_values(["RefDes","ComponentType"], kind="stable").reset_index(drop=True)


# ============================ Optional LLM ============================
def llm_classify_row(row: pd.Series, safety_goal: str):
    if OAI is None:
        return "NEEDS_REVIEW", "No OpenAI key configured."
    prompt = f"""You are an FMEDA safety expert.
Safety Goal: {safety_goal}
RefDes: {row['RefDes']}
Type: {row['ComponentType']}
Failure Mode: {row['FailureMode']}
FIT: {row['FIT']}
DC: {row['DC']}
Return JSON only: {{"label":"SAFE|UNSAFE|NEEDS_REVIEW","reason":"<short>"}}"""
    try:
        r = OAI.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        data = json.loads(r.choices[0].message.content)
        return data.get("label","NEEDS_REVIEW"), data.get("reason","")
    except Exception as e:
        return "NEEDS_REVIEW", f"LLM error: {e}"

def compute_spfm(fmeda: pd.DataFrame) -> float:
    d = fmeda.copy()
    if "FIT_eff" not in d.columns:
        d["FIT_eff"] = pd.to_numeric(d["FIT"], errors="coerce").fillna(0.0)
    lam_total = d["FIT_eff"].sum()
    lam_spf   = d.loc[d.get("Label","SAFE").str.upper()=="UNSAFE", "FIT_eff"].sum()
    return 1.0 if lam_total == 0 else 1.0 - lam_spf/lam_total


# ==================================== UI ====================================
st.title("FMEDA Builder — KiCad schematic × Failure DB")

safety_goal = st.text_input("Safety goal", "Prevent unintended output > 5 V")
use_llm     = st.toggle("Use LLM for SAFE/UNSAFE classification", value=False, help="Needs OPENAI_API_KEY in secrets")

left, right = st.columns([1,1])
with left:
    sch_file = st.file_uploader("Upload KiCad 9 schematic (.kicad_sch)", type=["kicad_sch"])
with right:
    fail_file = st.file_uploader("Upload Failure Rates & Modes CSV", type=["csv"])

if sch_file:
    sch_text = sch_file.read().decode("utf-8", errors="ignore")
    comp_df = parse_kicad_sch_components(sch_text)

    st.subheader("Detected components (edit if needed)")
    comp_df = st.data_editor(
        comp_df[["RefDes","ComponentType","ct_norm"]].rename(columns={"ct_norm":"_norm (read-only)"}),
        disabled=["_norm (read-only)"],
        height=320
    )
    comp_df["ct_norm"] = comp_df["ComponentType"].map(_norm)

    if fail_file:
        st.subheader("Failure DB preview & mapping")
        failure_df, raw_df_debug, col_share_name, col_fit_name = parse_failure_csv_with_mapping(fail_file)

        # Debug: raw vs parsed
        debug_numeric_preview(raw_df_debug, col_share_name, col_fit_name)

        st.divider()
        if st.button("Build FMEDA table"):
            fmeda = expand_fmeda(comp_df, failure_df)

            # Fill missing Share inside each RefDes group (equal split)
            fmeda["Share"] = pd.to_numeric(fmeda["Share"], errors="coerce")
            mask_na = fmeda["Share"].isna()
            if mask_na.any():
                eq = (
                    fmeda[mask_na]
                    .groupby(["RefDes","ComponentType"], dropna=False)["Share"]
                    .transform(lambda s: 1.0 / len(s) if len(s) else 1.0)
                )
                fmeda.loc[mask_na, "Share"] = eq

            # DC defaults to 0
            fmeda["DC"] = pd.to_numeric(fmeda["DC"], errors="coerce").fillna(0.0).clip(0,1)

            # Effective FIT (per failure mode)
            fmeda["FIT_eff"] = (
                pd.to_numeric(fmeda["FIT"], errors="coerce").fillna(0.0)
                * fmeda["Share"].astype(float)
                * (1.0 - fmeda["DC"].astype(float))
            )

            if use_llm:
                labels=[]; reasons=[]
                for _, row in fmeda.iterrows():
                    if pd.isna(row["FIT"]):
                        labels.append("UNASSESSED"); reasons.append("Unknown type / NaN FIT")
                    else:
                        lab, rsn = llm_classify_row(row, safety_goal)
                        labels.append(lab); reasons.append(rsn)
                fmeda["Label"]=labels; fmeda["Reason"]=reasons
                spfm = compute_spfm(fmeda)
                st.metric("SPFM (approx.)", f"{spfm*100:.2f}%")

            st.subheader("FMEDA result (RefDes first)")
            st.dataframe(fmeda, height=420)

            st.download_button("Download FMEDA (CSV)",
                               fmeda.to_csv(index=False).encode("utf-8"),
                               "fmeda_results.csv","text/csv")

            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
                fmeda.to_excel(xw, index=False, sheet_name="FMEDA")
                pd.DataFrame([{"SafetyGoal":safety_goal, "LLM": use_llm}]).to_excel(xw, index=False, sheet_name="Summary")
            st.download_button("Download FMEDA (XLSX)",
                               xbuf.getvalue(),
                               "fmeda_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload a KiCad 9 schematic to start.")
