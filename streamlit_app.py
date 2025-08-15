import streamlit as st
import pandas as pd
import re, io, csv, json, math
import xml.etree.ElementTree as ET
from collections import defaultdict

# ============================ App config ============================
st.set_page_config(page_title="FMEDA Builder (KiCad sch+net)", layout="wide")
OPENAI_MODEL = "gpt-4o"  # optional only if toggled

# Optional OpenAI client (keine Pflicht)
try:
    from openai import OpenAI
    OAI = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
except Exception:
    OAI = None

# Optional Excel Export
try:
    import xlsxwriter  # noqa: F401
    EXCEL_ENABLED = True
except Exception:
    EXCEL_ENABLED = False

# ============================ Helpers ============================
_num_pat = re.compile(r'([-+]?\d*[\.,]?\d+(?:[eE][-+]?\d+)?)')
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (str(s) if s is not None else "").strip().lower())

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

# ============================ KiCad sch parser (S-Expression) ============================
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
    if "comparator" in s: return "Comparator"
    if "mos" in s or "nmos" in s or "pmos" in s or "irf" in s: return "MOSFET"
    if "diode" in s: return "Diode"
    if "relay" in s: return "Relay"
    if "fuse" in s: return "Fuse"
    return None

def detect_type_from_value(val: str):
    t = (val or "").lower()
    if re.search(r'(^|\s)\d+(\.\d+)?(r|k|m)?(\s*ohm|$)', t): return "Resistor"
    if re.search(r'(^|\s)\d+(\.\d+)?(n|u|µ|p|f)(\s*|$)', t): return "Capacitor"
    if any(x in t for x in ["lm358","ne5532","tl072"]): return "OpAmp"
    if "irf" in t or "irfp" in t: return "MOSFET"
    return None

def _iter_blocks(text: str, tag: str):
    needle = f"({tag}"
    i = text.find(needle); n = len(text)
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

# ============================ KiCad netlist parser (.net XML) ============================
def parse_kicad_net(xml_bytes: bytes):
    """
    Returns:
      nets: dict[str, set[(ref,pin)]]
      comp_pins: dict[ref] -> set[pin_numbers_as_str]
    """
    nets = defaultdict(set)
    comp_pins = defaultdict(set)
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return {}, {}
    # typical structure: /export/netlist/nets/net (name/code) -> node ref/pin
    # and components under /export/components/comp
    for net in root.findall(".//net"):
        name = net.get("name") or ""
        for node in net.findall(".//node"):
            ref = node.get("ref") or ""
            pin = node.get("pin") or ""
            if ref and pin:
                nets[name].add((ref, pin))
                comp_pins[ref].add(pin)
    return dict(nets), {k: set(v) for k, v in comp_pins.items()}

# ============================ Failure DB parsing & mapping ============================
def auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    have_share = any(_norm(c) == "share" for c in df.columns)
    have_fit   = any(_norm(c) == "fit"   for c in df.columns)
    for c in df.columns:
        n = _norm(c)
        if not have_share and (("probabil" in n) or (n in {"probability","probabilitypct","probabilitypercent"}) or ("share" in n) or ("distribution" in n)):
            ren[c] = "Share"; have_share = True
        if not have_fit and (n in {"fit","basefit","fitbasefit","lambdafit","lambda","failurerate","rate"}):
            ren[c] = "FIT"; have_fit = True
    return df.rename(columns=ren) if ren else df

def parse_failure_csv_with_mapping(upload):
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

    raw = auto_rename_columns(raw)
    st.write("CSV columns (auto-renamed if needed):", list(raw.columns))

    FIT_ALIASES   = ["FIT","FIT (Base FIT)","Base FIT"]
    SHARE_ALIASES = ["Share","Share (Probability)","Probability"]

    def pick_by_alias(aliases, cols):
        for a in aliases:
            if a in cols: return a
        return None

    norm = {c: _norm(c) for c in raw.columns}
    def find_col(cands):
        for orig, n in norm.items():
            if n in cands: return orig
        return None

    c_fit   = pick_by_alias(FIT_ALIASES, list(raw.columns)) or find_col({"fit","lambda","rate"})
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

def debug_numeric_preview(raw_df: pd.DataFrame, col_share: str, col_fit: str, n: int = 12):
    rs = raw_df[col_share].astype(str)
    rf = raw_df[col_fit].astype(str)
    ps = parse_share_series(rs)
    pf = parse_number_series(rf)
    mask = rs.replace({"None":"", "nan":""}).str.strip().ne("") | rf.replace({"None":"", "nan":""}).str.strip().ne("")
    idx = raw_df.index[mask][:n]
    dbg = pd.DataFrame({
        "Share_raw":  rs.loc[idx].values,
        "Share_parsed": ps.loc[idx].values,
        "FIT_raw":    rf.loc[idx].values,
        "FIT_parsed": pf.loc[idx].values,
    }, index=idx)
    st.write("🔎 Parsing samples (first non-empty rows):")
    st.dataframe(dbg, height=240)

# ============================ FMEDA expand (joins sch with DB) ============================
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

# ============================ Supply / Target parsing ============================
SUPPLY_DEFAULTS = {
    "vcc": 5.0, "vdd": 5.0, "vin": 12.0, "vbatt": 12.0, "vbat": 12.0,
    "+5v": 5.0, "5v": 5.0, "3v3": 3.3, "3.3v": 3.3, "+3v3": 3.3, "+12v": 12.0, "12v": 12.0,
    "avcc": 5.0, "dvcc": 5.0
}
def voltage_from_netname(name: str) -> float | None:
    if not name: return None
    n = _norm(name)
    if n in SUPPLY_DEFAULTS: return SUPPLY_DEFAULTS[n]
    m = re.search(r'(\d+(?:\.\d+)?)v', n)  # e.g. 5v, 3v3 -> first "3"
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # 3v3 common: try "3v3"
    m2 = re.search(r'(\d)v(\d)', n)
    if m2:
        return float(f"{m2.group(1)}.{m2.group(2)}")
    return None

def extract_threshold_from_goal(text: str, default: float = 5.0) -> float:
    if not text: return default
    m = re.search(r'(\d+(?:\.\d+)?)\s*v', text.lower())
    return float(m.group(1)) if m else default

# ============================ Propagation Engine (single-fault) ============================
def nets_touch_component(nets: dict[str,set[tuple]], ref: str):
    """Return set of net names that have at least one node for given ref."""
    out = set()
    for n, nodes in nets.items():
        for (r,_p) in nodes:
            if r == ref:
                out.add(n); break
    return out

def component_is_two_terminal(net_pins: set[str]) -> bool:
    return len(net_pins) == 2

def failure_creates_short(failure_mode: str) -> bool:
    if not isinstance(failure_mode, str): return False
    fm = failure_mode.lower()
    return any(k in fm for k in ["short", "s/c", "sc"])

def failure_is_open(failure_mode: str) -> bool:
    if not isinstance(failure_mode, str): return False
    fm = failure_mode.lower()
    return any(k in fm for k in ["open", "o/c", "oc"])

def failure_short_to_named(failure_mode: str) -> str | None:
    """Detect 'short to VCC/GND/...' pattern."""
    if not isinstance(failure_mode, str): return None
    fm = failure_mode.lower()
    m = re.search(r'short\s*(to|2)\s*([a-z0-9\+\-\.]+)', fm)
    if m:
        return m.group(2)
    return None

def label_violation_for_row(row, nets: dict[str,set[tuple]], comp_pins_map: dict[str,set[str]],
                            target_net: str, hazard_threshold_v: float):
    """
    Conservative single-fault rule:
      - Only shorts can inject high potential directly.
      - We check if the component connects from any 'supply-like' net with voltage >= threshold
        onto the target net by its short failure.
      - Opens are SAFE for an overvoltage-goal.
      - Unknown multi-pin shorts are NEEDS_REVIEW if they touch both a high-supply net and target net.
    """
    ref = str(row["RefDes"])
    fm  = str(row.get("FailureMode","") or "")
    ctype = str(row.get("ComponentType","") or "")
    if ref not in comp_pins_map:
        return "UNASSESSED", "No pin/nets info in .net"

    # Target net present?
    if target_net not in nets:
        return "UNASSESSED", f"Target net '{target_net}' not found"

    # Which nets this component touches?
    nets_of_comp = nets_touch_component(nets, ref)
    if not nets_of_comp:
        return "UNASSESSED", "Component not placed in any net"

    # classify failure
    if failure_is_open(fm):
        return "SAFE", "Open fault cannot raise voltage at target in this simplified model"

    if not failure_creates_short(fm):
        # leakage, drift, param shift -> not modeled -> needs review
        return "NEEDS_REVIEW", "Non-short failure not modeled for overvoltage; manual assessment"

    # Named short target? e.g. "short to VCC"
    named = failure_short_to_named(fm)
    # Identify supply-like nets among component nets
    supply_candidates = []
    for n in nets_of_comp:
        v = voltage_from_netname(n)
        if v is not None and v >= hazard_threshold_v - 1e-9:
            supply_candidates.append((n, v))

    # If explicit 'short to <NAME>' and that name resolves to a supply >= threshold:
    if named:
        # try to resolve alias directly or via voltage parsing
        named_v = voltage_from_netname(named)
        if named_v is None:
            # if 'gnd' -> 0 V
            if _norm(named) in {"gnd","ground","0v"}:
                named_v = 0.0
        if named_v is not None and named_v >= hazard_threshold_v - 1e-9:
            # Does the component also touch the target net?
            if target_net in nets_of_comp:
                return "UNSAFE", f"'{fm}' injects {named_v:.2f}V to target via {ref}"
            # Or: explicit short between the target net name string and named supply exists?
        # fall through to generic

    # Generic short: If the component is two-terminal and one side is supply>=threshold and the other side is target, hazard.
    if component_is_two_terminal(comp_pins_map[ref]):
        # two nets only
        comp_nets = list(nets_of_comp)
        if len(comp_nets) == 2:
            a, b = comp_nets
            # if either side is target and the other a high supply
            if target_net in (a, b):
                other = b if target_net == a else a
                v_other = voltage_from_netname(other)
                if v_other is not None and v_other >= hazard_threshold_v - 1e-9:
                    return "UNSAFE", f"Short connects {other} (≈{v_other:.2f}V) to {target_net}"
                else:
                    return "SAFE", f"No high supply on opposite net ({other})"
        # more than 2 nets in .net (unusual for two-terminal) -> conservative review
        return "NEEDS_REVIEW", "Two-terminal part touches >2 nets in .net; review"

    # Multi-pin parts (ICs, MOSFETs etc.): conservative rule
    touches_target = target_net in nets_of_comp
    touches_high = len(supply_candidates) > 0
    if touches_target and touches_high:
        vs = ", ".join(f"{n}({v:.1f}V)" for n,v in supply_candidates[:3])
        return "UNSAFE", f"Short in multi-pin {ctype} may connect {vs} to {target_net}"
    if touches_high:
        return "NEEDS_REVIEW", f"Short could bridge high supply {supply_candidates[0][0]} to neighbor nets; target not directly on this part"
    return "SAFE", "No high supply among nets of this component"

# ============================ LLM (optional) ============================
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

# ============================ Metrics ============================
def compute_spfm(fmeda: pd.DataFrame) -> float | None:
    d = fmeda.copy()
    # nur gültige Zahlen
    d["FIT"]   = pd.to_numeric(d["FIT"], errors="coerce")
    d["Share"] = pd.to_numeric(d["Share"], errors="coerce")
    d["DC"]    = pd.to_numeric(d["DC"], errors="coerce").fillna(0.0).clip(0,1)
    # Effektiver Beitrag pro FM
    d["FIT_eff"] = (d["FIT"] * d["Share"] * (1.0 - d["DC"]))
    # Ungültige (NaN) ignorieren
    valid = d["FIT_eff"].notna()
    if not valid.any():
        return None
    lam_total = d.loc[valid, "FIT_eff"].sum()
    lam_spf   = d.loc[valid & (d.get("Label","SAFE").str.upper()=="UNSAFE"), "FIT_eff"].sum()
    if lam_total <= 0:
        return None
    return 1.0 - (lam_spf / lam_total)

# ============================ UI ============================
st.title("FMEDA Builder — KiCad sch + net × Failure DB")

safety_goal = st.text_input("Safety goal (free text)", "Prevent unintended output > 5 V at OUT")
target_net  = st.text_input("Target net name (exact as in netlist)", "OUT")
use_llm     = st.toggle("Use LLM (optional) to refine SAFE/UNSAFE", value=False)

left, mid, right = st.columns([1,1,1])
with left:
    sch_file = st.file_uploader("Upload KiCad 9 schematic (.kicad_sch)", type=["kicad_sch"])
with mid:
    net_file = st.file_uploader("Upload KiCad netlist (.net XML)", type=["net","xml"])
with right:
    fail_file = st.file_uploader("Upload Failure Rates & Modes (CSV)", type=["csv"])

if sch_file:
    sch_text = sch_file.read().decode("utf-8", errors="ignore")
    comp_df  = parse_kicad_sch_components(sch_text)

    st.subheader("Detected components (editable)")
    comp_df = st.data_editor(
        comp_df[["RefDes","ComponentType","ct_norm"]].rename(columns={"ct_norm":"_norm (read-only)"}),
        disabled=["_norm (read-only)"],
        height=320
    )
    comp_df["ct_norm"] = comp_df["ComponentType"].map(_norm)

    # Netlist parse (optional but recommended)
    nets, comp_pins_map = ({}, {})
    if net_file:
        nets, comp_pins_map = parse_kicad_net(net_file.read())
        st.success(f"Netlist parsed: {len(nets)} nets, {len(comp_pins_map)} components with pin mapping.")
        # kleine Vorschau
        if nets:
            some = list(nets.items())[:5]
            st.write({k: list(v)[:4] for k,v in some})

    if fail_file:
        st.subheader("Failure DB preview & mapping")
        failure_df, raw_df_debug, col_share_name, col_fit_name = parse_failure_csv_with_mapping(fail_file)
        debug_numeric_preview(raw_df_debug, col_share_name, col_fit_name)
        st.divider()

        if st.button("Build FMEDA table"):
            fmeda = expand_fmeda(comp_df, failure_df)

            # fehlende Share pro (RefDes,ComponentType) gleichmäßig verteilen
            fmeda["Share"] = pd.to_numeric(fmeda["Share"], errors="coerce")
            mask_na = fmeda["Share"].isna()
            if mask_na.any():
                eq = (
                    fmeda[mask_na]
                    .groupby(["RefDes","ComponentType"], dropna=False)["Share"]
                    .transform(lambda s: 1.0 / len(s) if len(s) else 1.0)
                )
                fmeda.loc[mask_na, "Share"] = eq

            # DC defaults 0
            fmeda["DC"] = pd.to_numeric(fmeda["DC"], errors="coerce").fillna(0.0).clip(0,1)

            # einfache Propagation gegen Safety Goal
            threshold_v = extract_threshold_from_goal(safety_goal, default=5.0)
            labels, reasons = [], []
            for _, row in fmeda.iterrows():
                if pd.isna(row["FIT"]) or pd.isna(row["Share"]):
                    labels.append("UNASSESSED"); reasons.append("Missing FIT/Share or unknown type")
                    continue
                if not nets or not comp_pins_map:
                    labels.append("NEEDS_REVIEW"); reasons.append("No netlist; cannot trace propagation")
                    continue
                lab, rsn = label_violation_for_row(row, nets, comp_pins_map, target_net, threshold_v)
                labels.append(lab); reasons.append(rsn)

            fmeda["Label"]  = labels
            fmeda["Reason"] = reasons

            # Effektiver FIT pro FM
            fmeda["FIT_eff"] = (
                pd.to_numeric(fmeda["FIT"], errors="coerce").fillna(pd.NA)
                * pd.to_numeric(fmeda["Share"], errors="coerce").fillna(pd.NA)
                * (1.0 - pd.to_numeric(fmeda["DC"], errors="coerce").fillna(0.0))
            )

            spfm = compute_spfm(fmeda)
            if spfm is not None:
                st.metric("SPFM (single point fault metric, approx.)", f"{spfm*100:.2f}%")
            else:
                st.info("SPFM nicht berechenbar (keine gültigen FIT×Share Werte).")

            st.subheader("FMEDA result")
            st.dataframe(fmeda, height=480, use_container_width=True)

            # CSV Download
            st.download_button("Download FMEDA (CSV)",
                               fmeda.to_csv(index=False).encode("utf-8"),
                               "fmeda_results.csv","text/csv")

            # Excel Download
            if EXCEL_ENABLED:
                xbuf = io.BytesIO()
                with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
                    fmeda.to_excel(xw, index=False, sheet_name="FMEDA")
                    pd.DataFrame([{
                        "SafetyGoal": safety_goal,
                        "TargetNet": target_net,
                        "LLM": use_llm
                    }]).to_excel(xw, index=False, sheet_name="Summary")
                st.download_button("Download FMEDA (XLSX)",
                                   xbuf.getvalue(),
                                   "fmeda_results.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Excel-Export deaktiviert (installiere `xlsxwriter`).")
else:
    st.info("Bitte .kicad_sch hochladen, dann .net und Failure-DB (CSV).")
