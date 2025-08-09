import streamlit as st
import pandas as pd
import re, io, json, uuid

# ---------- CONFIG ----------
st.set_page_config(page_title="FMEDA from KiCad with LLM Classification", layout="wide")

# Optional: change to your preferred model
OPENAI_MODEL = "gpt-4o-mini"

# ---------- LLM (OpenAI) ----------
try:
    from openai import OpenAI
    _OPENAI_CLIENT = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    _OPENAI_CLIENT = None  # Will show a warning when classification is requested

def new_uuid(): return str(uuid.uuid4())

# ---------- TYPE DETECTION HELPERS ----------
REF_PREFIX = {
    "R":"Resistor","C":"Capacitor","L":"Inductor","D":"Diode","Z":"Zener",
    "Q":"MOSFET","T":"BJT","U":"IC_Analog","A":"OpAmp","K":"Relay","F":"Fuse",
    "J":"Connector","X":"Connector",
}

def detect_type_from_ref(ref: str):
    if not ref: return None
    m = re.match(r"^([A-Za-z]+)", ref)
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
    if "connector" in s or "audiojack" in s: return "Connector"
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

# ---------- ROBUST KICAD 9 PARSER ----------
def _iter_sexpr_blocks(text: str, tag: str):
    """Yield full balanced blocks that start with '(tag'."""
    needle = f"({tag}"
    n = len(text)
    start = text.find(needle)
    while start != -1:
        i = start
        depth = 0
        started = False
        while i < n:
            ch = text[i]
            if ch == "(":
                depth += 1
                started = True
            elif ch == ")":
                depth -= 1
                if started and depth == 0:
                    yield text[start:i+1]
                    break
            i += 1
        start = text.find(needle, i)

def parse_kicad_sch_components(sch_text: str) -> pd.DataFrame:
    rows = []
    for block in _iter_sexpr_blocks(sch_text, "symbol"):
        m_lib = re.search(r'\(lib_id\s+"([^"]+)"\)', block)
        lib_id = m_lib.group(1) if m_lib else ""
        m_ref = re.search(r'\(property\s+"Reference"\s+"([^"]*)"', block)
        m_val = re.search(r'\(property\s+"Value"\s+"([^"]*)"', block)
        ref = (m_ref.group(1) if m_ref else "").strip()
        val = (m_val.group(1) if m_val else "").strip()

        t = detect_type_from_ref(ref)
        t2 = detect_type_from_lib(lib_id) or detect_type_from_value(val)
        if t == "IC_Analog" and t2:
            t = t2
        elif t2 and t != t2:
            if t in ("IC_Analog", "Other") or (t == "Capacitor" and t2 == "Capacitor_Polarized"):
                t = t2
        ctype = t or "Other"
        if not ref: ref = "?"

        rows.append({"RefDes": ref, "ComponentType": ctype, "Value": val, "LibId": lib_id})

    if not rows:
        return pd.DataFrame(columns=["RefDes", "ComponentType"])

    df = pd.DataFrame(rows)[["RefDes","ComponentType"]]
    df.loc[df["RefDes"] == "?", "ComponentType"] = "unknown component"
    df.loc[df["ComponentType"].isna(), "ComponentType"] = "unknown component"
    return df

# ---------- FAILURE CSV (robust detection + optional UI mapping) ----------
def parse_failure_csv_with_mapping(upload) -> pd.DataFrame:
    import csv
    import io
    import pandas as pd
    import re

    raw_bytes = upload.read()
    upload.seek(0)

    # 1) Try automatic delimiter detection
    sample = raw_bytes[:4096].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        sep_guess = dialect.delimiter
    except Exception:
        sep_guess = None

    # 2) Read with best guess; fallbacks if we still get 1 column
    def _read_with(sep):
        return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, engine="python", decimal=",", dtype=str)

    if sep_guess:
        raw = _read_with(sep_guess)
    else:
        # pandas autodetect
        try:
            raw = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", decimal=",", dtype=str)
        except Exception:
            raw = _read_with(",")

    if raw.shape[1] == 1:
        # brute-force fallbacks
        for s in [";", "\t", "|", ","]:
            raw = _read_with(s)
            if raw.shape[1] > 1:
                break

    # 3) If header line itself is packed (e.g., 'ComponentType;FailureMode;Share;FIT')
    if raw.shape[1] == 1 and ";" in raw.columns[0]:
        cols = [c.strip() for c in raw.columns[0].split(";")]
        tmp = raw.iloc[:,0].str.split(";", expand=True)
        if tmp.shape[1] == len(cols):
            tmp.columns = cols
            raw = tmp

    # Normalize header names for matching
    norm = {c: re.sub(r"[^a-z0-9]", "", c.strip().lower()) for c in raw.columns}
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

    # 4) If still missing, ask user to map once
    missing = [name for name,col in {
        "Component Type": c_type, "Failure Mode": c_mode, "Mode Share": c_share, "FIT": c_fit
    }.items() if col is None]
    if missing:
        st.warning(f"Please map missing columns: {', '.join(missing)}")
        cols = list(raw.columns)
        c_type = st.selectbox("Column for Component Type", cols, index=0 if c_type is None else cols.index(c_type))
        c_mode = st.selectbox("Column for Failure Mode", cols, index=0 if c_mode is None else cols.index(c_mode))
        c_share= st.selectbox("Column for Mode Share", cols, index=0 if c_share is None else cols.index(c_share))
        c_fit  = st.selectbox("Column for FIT", cols, index=0 if c_fit is None else cols.index(c_fit))
        c_dc   = st.selectbox("Column for Diagnostic Coverage (optional)", ["<none>"]+cols, index=0)
        c_det  = st.selectbox("Column for Detectable (optional)", ["<none>"]+cols, index=0)
        c_dnm  = st.text_input("Column for Diagnostic Name (optional)", value=c_dnm or "")
        c_dc = None if c_dc == "<none>" else c_dc
        c_det= None if c_det == "<none>" else c_det
        c_dnm= None if not c_dnm else c_dnm

    # 5) Build normalized table
    out = pd.DataFrame()
    out["ComponentType"] = raw[c_type].astype(str).fillna("").str.strip()
    out["FailureMode"]   = raw[c_mode].astype(str).fillna("").str.strip()

    # If a cell carries a whole hierarchy: 'Resistor;Passive;Fixed Resistor'
    def canonicalize_type_cell(s: str) -> str:
        parts = [p.strip() for p in str(s).split(";") if p.strip()]
        if not parts: return ""
        # Prefer a known base class if present
        base_set = {"resistor","capacitor","capacitor_polarized","inductor","diode","zener",
                    "bjt","mosfet","opamp","comparator","relay","fuse","connector"}
        for p in parts:
            p0 = re.sub(r"[^a-z0-9]", "", p.lower())
            if p0 in base_set:
                return p.capitalize()
        return parts[0]  # fallback to first token

    out["ComponentType"] = out["ComponentType"].map(canonicalize_type_cell)

    # Shares and FITs
    share = pd.to_numeric(raw[c_share].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)
    if share.max() > 1.5: share = share / 100.0
    out["Share"] = share.clip(lower=0)

    fit = pd.to_numeric(raw[c_fit].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out["FIT"] = fit

    out["DC"]  = pd.to_numeric(raw[c_dc], errors="coerce").fillna(0.0).clip(0,1) if c_dc else 0.0
    if c_det:
        v = raw[c_det].astype(str).str.strip().str.lower()
        out["Detectable"] = v.isin(["1","true","yes","y","t"])
    else:
        out["Detectable"] = False
    out["DiagnosticName"] = raw[c_dnm].astype(str) if (c_dnm and c_dnm in raw.columns) else ""

    # Final sanity: show preview
    st.success("Failure DB parsed.")
    st.dataframe(out.head(20), height=240)
    return out


# ---------- EXPAND: schematic × failure modes ----------
def merge_with_failure_modes(components_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    # find actual ComponentType column just in case
    comp_col = next((c for c in failure_df.columns if c.strip().lower() == "componenttype"), None)
    if comp_col is None:
        st.error("Failure Rates CSV must contain a 'ComponentType' column (after mapping).")
        st.stop()

    rows = []
    for _, comp in components_df.iterrows():
        comp_type = comp["ComponentType"]
        match_df = failure_df[failure_df[comp_col].astype(str).str.strip().str.lower() == comp_type.lower()]

        if match_df.empty or comp_type == "unknown component":
            rows.append({
                "RefDes": comp["RefDes"],
                "ComponentType": "unknown component",
                "FailureMode": "unknown",
                "Share": float("nan"),
                "FIT": float("nan"),
                "DC": float("nan"),
                "Detectable": False,
                "DiagnosticName": ""
            })
        else:
            for _, fm in match_df.iterrows():
                rows.append({
                    "RefDes": comp["RefDes"],
                    "ComponentType": comp_type,
                    "FailureMode": fm.get("FailureMode"),
                    "Share": fm.get("Share"),
                    "FIT": fm.get("FIT"),
                    "DC": fm.get("DC"),
                    "Detectable": fm.get("Detectable"),
                    "DiagnosticName": fm.get("DiagnosticName")
                })
    return pd.DataFrame(rows)

# ---------- LLM classification ----------
def llm_classify_row(row: pd.Series, safety_goal: str) -> tuple[str,str]:
    if _OPENAI_CLIENT is None:
        return "NEEDS_REVIEW", "No OpenAI API key in st.secrets[\"OPENAI_API_KEY\"]."
    prompt = f"""
You are an expert in FMEDA for electronics safety.
Safety Goal: {safety_goal}

Component:
  RefDes: {row.get('RefDes')}
  Type: {row.get('ComponentType')}
  Failure Mode: {row.get('FailureMode')}
  FIT: {row.get('FIT')}
  Diagnostic Coverage: {row.get('DC')}

Decide if this failure mode is SAFE or UNSAFE with respect to the Safety Goal.
If insufficient context, return NEEDS_REVIEW.

Return ONLY JSON:
{{"label":"SAFE|UNSAFE|NEEDS_REVIEW","reason":"<short>"}}
"""
    try:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        return data.get("label","NEEDS_REVIEW"), data.get("reason","")
    except Exception as e:
        return "NEEDS_REVIEW", f"LLM error: {e}"

# ---------- METRICS ----------
def compute_spfm(fmeda: pd.DataFrame) -> float:
    df = fmeda.dropna(subset=["FIT"])
    if df.empty: return 1.0
    lam_total = df["FIT"].sum()
    lam_spf   = df.loc[df["Label"].str.upper() == "UNSAFE", "FIT"].sum()
    return 1.0 if lam_total == 0 else 1.0 - (lam_spf / lam_total)

# ========================= UI =========================
st.title("FMEDA from KiCad + LLM Safety Classification")

safety_goal = st.text_input("Safety Goal", "Prevent unintended output > 5 V")
sch_file    = st.file_uploader("Upload KiCad schematic (.kicad_sch)", type=["kicad_sch"])
fail_file   = st.file_uploader("Upload Failure Rates & Modes CSV", type=["csv"])

if sch_file:
    sch_text = sch_file.read().decode("utf-8", errors="ignore")
    comp_df = parse_kicad_sch_components(sch_text)

    st.subheader("Detected Components (edit if necessary)")
    st.caption("Ensure ComponentType matches your Failure Rates CSV. Use common names like Resistor, Capacitor, OpAmp, MOSFET, Diode, etc.")
    comp_df = st.data_editor(comp_df, num_rows="dynamic", height=420)

    if fail_file:
        st.subheader("Failure Rates & Modes")
        failure_df = parse_failure_csv_with_mapping(fail_file)
        st.dataframe(failure_df.head(20), height=300)

        if st.button("Run FMEDA + LLM Classification"):
            with st.spinner("Merging and classifying..."):
                fmeda_df = merge_with_failure_modes(comp_df, failure_df)

                labels, reasons = [], []
                for _, row in fmeda_df.iterrows():
                    if pd.isna(row["FIT"]):
                        labels.append("UNASSESSED"); reasons.append("Unknown component / NaN FIT")
                    else:
                        lab, rea = llm_classify_row(row, safety_goal)
                        labels.append(lab); reasons.append(rea)
                fmeda_df["Label"] = labels
                fmeda_df["Reason"] = reasons

                spfm = compute_spfm(fmeda_df)

            st.subheader("FMEDA (classified)")
            st.dataframe(fmeda_df, height=420)
            st.metric("SPFM (approx.)", f"{spfm*100:.2f}%")
            st.caption("LFM not included in this step.")

            # Exports
            st.download_button("Download FMEDA (CSV)",
                fmeda_df.to_csv(index=False).encode("utf-8"),
                file_name="fmeda_results.csv",
                mime="text/csv"
            )

            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
                fmeda_df.to_excel(xw, index=False, sheet_name="FMEDA")
                pd.DataFrame([{"SPFM": spfm, "SafetyGoal": safety_goal}]).to_excel(xw, index=False, sheet_name="Summary")
            st.download_button("Download FMEDA (XLSX)",
                xbuf.getvalue(),
                file_name="fmeda_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    st.info("Upload a KiCad 9 schematic (.kicad_sch) to begin.")

