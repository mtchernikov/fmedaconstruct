import streamlit as st
import pandas as pd
import re, io, csv, json, uuid

st.set_page_config(page_title="FMEDA Builder (KiCad + Failure DB)", layout="wide")

# --- Optional LLM (kept off by default; enable if you have a key) ---
OPENAI_MODEL = "gpt-4o"
try:
    from openai import OpenAI
    OAI = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
except Exception:
    OAI = None

def new_uuid(): return str(uuid.uuid4())
def _norm(s:str)->str: return re.sub(r'[^a-z0-9]', '', (str(s) if s is not None else "").strip().lower())

# --------- KiCad 9 schematic parser (balanced S-expr) ----------
REF_PREFIX = {
    "R":"Resistor","C":"Capacitor","L":"Inductor","D":"Diode","Z":"Zener",
    "Q":"MOSFET","T":"BJT","U":"IC_Analog","A":"OpAmp","K":"Relay","F":"Fuse",
    "J":"Connector","X":"Connector",
}
def detect_type_from_ref(ref:str):
    m=re.match(r'^([A-Za-z]+)', ref or "")
    if not m: return None
    pref=m.group(1).upper()
    for p in sorted(REF_PREFIX,key=lambda x:-len(x)):
        if pref.startswith(p): return REF_PREFIX[p]
    return None
def detect_type_from_lib(lib_id:str):
    s=(lib_id or "").lower()
    if "opamp" in s or "amplifier_operational" in s: return "OpAmp"
    if ":cp" in s or "electroly" in s: return "Capacitor_Polarized"
    if ":c"  in s or "capacitor" in s: return "Capacitor"
    if ":r"  in s or "resistor"  in s: return "Resistor"
    if "lm393" in s or "comparator" in s: return "Comparator"
    if "mos" in s or "nmos" in s or "pmos" in s or "irf" in s: return "MOSFET"
    if "diode" in s: return "Diode"
    if "relay" in s: return "Relay"
    if "fuse" in s: return "Fuse"
    return None
def detect_type_from_value(val:str):
    t=(val or "").lower()
    if re.search(r'(^|\s)\d+(\.\d+)?(r|k|m)?(\s*ohm|$)', t): return "Resistor"
    if re.search(r'(^|\s)\d+(\.\d+)?(n|u|µ|p|f)(\s*|$)', t): return "Capacitor"
    if "ne5532" in t or "lm358" in t: return "OpAmp"
    if "irf" in t or "irfp" in t: return "MOSFET"
    return None

def _iter_blocks(text:str, tag:str):
    needle=f"({tag}"
    i=text.find(needle)
    n=len(text)
    while i!=-1:
        depth=0; start=i; j=i
        while j<n:
            ch=text[j]
            if ch=="(": depth+=1
            elif ch==")":
                depth-=1
                if depth==0:
                    yield text[start:j+1]; break
            j+=1
        i=text.find(needle, j)

def parse_kicad_sch_components(sch_text:str)->pd.DataFrame:
    rows=[]
    for blk in _iter_blocks(sch_text,"symbol"):
        m_lib=re.search(r'\(lib_id\s+"([^"]+)"\)', blk)
        m_ref=re.search(r'\(property\s+"Reference"\s+"([^"]*)"', blk)
        m_val=re.search(r'\(property\s+"Value"\s+"([^"]*)"', blk)
        lib_id=m_lib.group(1) if m_lib else ""
        ref=(m_ref.group(1) if m_ref else "").strip()
        val=(m_val.group(1) if m_val else "").strip()

        t = detect_type_from_ref(ref)
        t2= detect_type_from_lib(lib_id) or detect_type_from_value(val)
        if t=="IC_Analog" and t2: t=t2
        elif t2 and t!=t2:
            if t in ("IC_Analog","Other") or (t=="Capacitor" and t2=="Capacitor_Polarized"): t=t2
        ctype=t or "Other"
        if not ref: ref="?"

        rows.append({"RefDes": ref, "ComponentType": ctype})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["RefDes","ComponentType"])
    # Explicit unknowns
    df.loc[df["RefDes"]=="?","ComponentType"]="unknown component"
    df["ct_norm"]=df["ComponentType"].map(_norm)
    return df[["RefDes","ComponentType","ct_norm"]]

# --------- Failure DB parser (delimiter auto-detect + header mapping + numeric) ----------
def parse_failure_csv_with_mapping(upload) -> pd.DataFrame:
    import csv, io, re
    import pandas as pd

    raw_bytes = upload.read()
    upload.seek(0)

    # --- delimiter sniffing ---
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

    # header packed with ';' ?
    if raw.shape[1] == 1 and ";" in raw.columns[0]:
        cols = [c.strip() for c in raw.columns[0].split(";")]
        tmp = raw.iloc[:,0].str.split(";", expand=True)
        if tmp.shape[1] == len(cols):
            tmp.columns = cols
            raw = tmp

    st.write("CSV columns:", list(raw.columns))

    norm = {c: re.sub(r'[^a-z0-9]', '', c.strip().lower()) for c in raw.columns}
    def find_col(cands):
        for orig, n in norm.items():
            if n in cands:
                return orig
        return None

    c_type = find_col({"componenttype","type","class","category","parttype","compclass"})
    c_mode = find_col({"failuremode","mode","fm"})
    c_share= find_col({"share","modeshare","distribution","percent","modesharepercent","modesharepct"})
    c_fit  = find_col({"fit","lambda","rate","fitrate","lambdafit","lambdafitper1e9h"})
    c_dc   = find_col({"dc","diagnosticcoverage","coverage","diagcoverage"})
    c_det  = find_col({"detectable","detected","isdetected"})
    c_dnm  = find_col({"diagnosticname","diag","diagnostic"})

    missing = [name for name,col in {
        "Component Type": c_type, "Failure Mode": c_mode, "Mode Share": c_share, "FIT": c_fit
    }.items() if col is None]
    if missing:
        st.warning(f"Please map missing columns: {', '.join(missing)}")
        cols = list(raw.columns)
        c_type = st.selectbox("Component Type column", cols, index=0 if c_type is None else cols.index(c_type))
        c_mode = st.selectbox("Failure Mode column",  cols, index=0 if c_mode is None else cols.index(c_mode))
        c_share= st.selectbox("Mode Share column",    cols, index=0 if c_share is None else cols.index(c_share))
        c_fit  = st.selectbox("FIT column",           cols, index=0 if c_fit is None else cols.index(c_fit))
        c_dc   = st.selectbox("Diagnostic Coverage (optional)", ["<none>"]+cols, index=0)
        c_det  = st.selectbox("Detectable (optional)",          ["<none>"]+cols, index=0)
        c_dnm  = st.text_input("Diagnostic Name (optional)", value=c_dnm or "")
        c_dc = None if c_dc == "<none>" else c_dc
        c_det= None if c_det == "<none>" else c_det
        c_dnm= None if not c_dnm else c_dnm

    # -------- Series-safe numeric parser --------
    def to_num_series(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Component type canonicalization (handles hierarchical strings with ';')
    def canonicalize_type_cell(s: str) -> str:
        parts = [p.strip() for p in str(s).split(";") if p.strip()]
        if not parts: return ""
        base = {"resistor","capacitor","capacitor_polarized","inductor","diode","zener",
                "bjt","mosfet","opamp","comparator","relay","fuse","connector"}
        for p in parts:
            if re.sub(r'[^a-z0-9]', '', p.lower()) in base:
                return p.capitalize()
        return parts[0]

    out = pd.DataFrame()
    out["ComponentType"] = raw[c_type].astype(str).fillna("").map(canonicalize_type_cell)
    out["FailureMode"]   = raw[c_mode].astype(str).fillna("").str.strip()

    share = to_num_series(raw[c_share]).fillna(0.0)
    if share.max() > 1.5:  # treat as percent if >150%
        share = share / 100.0
    out["Share"] = share.clip(lower=0)

    out["FIT"] = to_num_series(raw[c_fit])

    out["DC"]  = to_num_series(raw[c_dc]).fillna(0.0).clip(0,1) if c_dc else 0.0
    if c_det:
        v = raw[c_det].astype(str).str.strip().str.lower()
        out["Detectable"] = v.isin(["1","true","yes","y","t"])
    else:
        out["Detectable"] = False
    out["DiagnosticName"] = raw[c_dnm].astype(str) if (c_dnm and c_dnm in raw.columns) else ""

    # normalized join key
    out["ct_norm"] = out["ComponentType"].apply(lambda s: re.sub(r'[^a-z0-9]', '', s.strip().lower()))
    st.success("Failure DB parsed.")
    st.dataframe(out.head(20), height=240)
    return out


# --------- Expand components × failure modes (cross-merge by type) ----------
def expand_fmeda(components_df:pd.DataFrame, failure_df:pd.DataFrame)->pd.DataFrame:
    # Cross merge by normalized type
    merged = components_df.merge(failure_df, how="left", on="ct_norm", suffixes=("",""))
    # If no match OR component was explicitly unknown → mark as unknown line
    unknown_mask = merged["ComponentType_y"].isna() | (components_df["ct_norm"]=="unknowncomponent")
    merged.loc[unknown_mask, ["ComponentType_x","FailureMode","Share","FIT","DC","Detectable","DiagnosticName"]] = [
        "unknown component", pd.NA, pd.NA, pd.NA, pd.NA, False, ""
    ]
    # Choose display columns and fix names
    merged["ComponentType"] = merged["ComponentType_x"].where(~unknown_mask, "unknown component").fillna(merged["ComponentType_y"])
    fmeda = merged[["RefDes","ComponentType","FailureMode","Share","FIT","DC","Detectable","DiagnosticName"]].copy()
    # Sort: RefDes first as requested
    fmeda = fmeda.sort_values(["RefDes","ComponentType"], kind="stable").reset_index(drop=True)
    return fmeda

# --------- Optional LLM classification ----------
def llm_classify_row(row, safety_goal):
    if OAI is None:
        return "NEEDS_REVIEW", "No OpenAI key configured."
    prompt = f"""You are an FMEDA safety expert.
Safety Goal: {safety_goal}
RefDes: {row['RefDes']}
Type: {row['ComponentType']}
Failure Mode: {row['FailureMode']}
FIT: {row['FIT']}
DC: {row['DC']}

Return compact JSON only:
{{"label":"SAFE|UNSAFE|NEEDS_REVIEW","reason":"<short one line>"}}"""
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

def compute_spfm(fmeda:pd.DataFrame)->float:
    d=fmeda.dropna(subset=["FIT"])
    if d.empty: return 1.0
    lam_total=d["FIT"].sum()
    lam_spf=d.loc[d["Label"].str.upper()=="UNSAFE","FIT"].sum() if "Label" in d else 0.0
    return 1.0 if lam_total==0 else 1.0 - lam_spf/lam_total

# =============================== UI ===============================
st.title("FMEDA Builder — KiCad schematic × Failure DB")

col1,col2 = st.columns([1,1])
with col1:
    sch_file = st.file_uploader("Upload KiCad 9 schematic (.kicad_sch)", type=["kicad_sch"])
with col2:
    fail_file = st.file_uploader("Upload Failure Rates & Modes CSV", type=["csv"])

safety_goal = st.text_input("Safety goal", "Prevent unintended output > 5 V")
use_llm = st.toggle("Use LLM for SAFE/UNSAFE classification", value=False, help="Requires OPENAI_API_KEY in secrets")

if sch_file:
    sch_text = sch_file.read().decode("utf-8", errors="ignore")
    components_df = parse_kicad_sch_components(sch_text)
    st.subheader("Detected components (edit if needed)")
    components_df = st.data_editor(
        components_df[["RefDes","ComponentType","ct_norm"]].rename(columns={"ct_norm":"_norm (read-only)"}),
        disabled=["_norm (read-only)"],
        height=320
    )
    # Recompute normalization if user edited types
    components_df["ct_norm"] = components_df["ComponentType"].map(_norm)

    if fail_file:
        st.subheader("Failure DB preview & mapping")
        failure_df = parse_failure_csv_with_mapping(fail_file)

        st.divider()
        if st.button("Build FMEDA table"):
            fmeda = expand_fmeda(components_df, failure_df)

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

            st.subheader("FMEDA result")
            st.dataframe(fmeda, height=420)

            st.download_button("Download FMEDA (CSV)",
                               fmeda.to_csv(index=False).encode("utf-8"),
                               "fmeda_results.csv","text/csv")

            xbuf=io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
                fmeda.to_excel(xw, index=False, sheet_name="FMEDA")
                meta=pd.DataFrame([{"SafetyGoal":safety_goal, "LLM": use_llm}])
                meta.to_excel(xw, index=False, sheet_name="Summary")
            st.download_button("Download FMEDA (XLSX)",
                               xbuf.getvalue(),
                               "fmeda_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Upload a KiCad 9 schematic to start.")

