# streamlit_app.py
import streamlit as st
import pandas as pd
import re, io, csv, json, time
from collections import defaultdict

# ============================ App config ============================
st.set_page_config(page_title="Safety Co-Pilot — FMEDA + Sensitivity (LLM)", layout="wide")
OPENAI_MODEL = "gpt-4o"  # requires OPENAI_API_KEY in Streamlit secrets

# Optional OpenAI client
try:
    from openai import OpenAI
    OAI = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
except Exception:
    OAI = None

# Optional Excel export
try:
    import xlsxwriter  # noqa: F401
    EXCEL_ENABLED = True
except Exception:
    EXCEL_ENABLED = False


# ============================ Session state (sticky build) ============================
for k, v in {
    "built": False,
    "fmeda": None,
    "base_spfm": None,
    "nets": None,
    "comp_pins_map": None,
    "pin_name_map": None,
    "node2net": None,
    "target_net": None,
    "comp_df_snapshot": None,
    "fail_df_snapshot": None,
    "include_hints": False,
}.items():
    st.session_state.setdefault(k, v)


# ============================ Utilities ============================
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


# ============================ KiCad .net parsing (S-exp + XML) ============================
def iter_blocks_exact(text: str, tag: str):
    pat = re.compile(r'\(' + re.escape(tag) + r'(?=[\s\()])')
    for m in pat.finditer(text):
        i = m.start()
        depth = 0; j = i; n = len(text)
        while j < n:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    yield text[i:j+1]
                    break
            j += 1

def _token_alt(key):
    return rf'\({key}\s+("([^"]+)"|[^\s\)]+)\)'

def _pick(m, quoted_idx=2, any_idx=1):
    if m is None:
        return ""
    try: val_q = m.group(quoted_idx)
    except IndexError: val_q = None
    try: val_a = m.group(any_idx)
    except IndexError: val_a = None
    val = val_q if val_q is not None else val_a
    if val is None: return ""
    return val.strip('"')

REF_PREFIX = {
    "R":"Resistor","C":"Capacitor","L":"Inductor","D":"Diode","Z":"Zener",
    "Q":"Transistor","T":"Transistor","U":"IC","A":"OpAmp","K":"Relay","F":"Fuse",
    "J":"Connector","X":"Connector",
}

def infer_type(ref: str, value: str) -> str:
    m = re.match(r'^([A-Za-z]+)', ref or "")
    base = "Other"
    if m:
        pref = m.group(1).upper()
        for p in sorted(REF_PREFIX, key=lambda x: -len(x)):
            if pref.startswith(p): base = REF_PREFIX[p]; break
    v = (value or "").lower()
    if base=="Transistor":
        if "mos" in v or "nmos" in v or "pmos" in v: return "MOSFET"
        if "npn" in v or "pnp" in v: return "BJT"
        return "MOSFET" if (ref or "").upper().startswith("Q") else "BJT"
    if base=="IC" and ("opamp" in v or "lm" in v): return "OpAmp"
    if "diode" in v or v.startswith("1n") or v.startswith("sb"): return "Diode"
    return base

def parse_kicad_net(net_bytes: bytes):
    text = net_bytes.decode("utf-8", errors="ignore").strip()

    # XML
    if text.startswith("<"):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
        nets = defaultdict(set)
        comp_pins = defaultdict(set)
        pin_name_map = {}
        ref_to_libpart = {}
        ref_types = {}

        for comp in root.findall(".//components/comp"):
            ref = comp.get("ref") or ""
            if not ref: continue
            val = (comp.findtext("./value") or "")
            ref_types[ref] = infer_type(ref, val)
            ls = comp.find("./libsource")
            if ls is not None:
                ref_to_libpart[ref] = (ls.get("lib") or "", ls.get("part") or "")

        libpart_pins = {}
        for lp in root.findall(".//libparts/libpart"):
            lib = lp.get("lib") or ""
            part= lp.get("part") or ""
            pins = {}
            for p in lp.findall("./pins/pin"):
                num  = p.get("num") or ""
                name = p.get("name") or ""
                ptype= p.get("type") or ""
                if num: pins[str(num)] = {"name": name, "type": ptype}
            libpart_pins[(lib,part)] = pins

        for net in root.findall(".//nets/net"):
            name = net.get("name") or ""
            for node in net.findall("./node"):
                ref = node.get("ref") or ""
                pin = node.get("pin") or ""
                if ref and pin:
                    nets[name].add((ref, pin))
                    comp_pins[ref].add(pin)
                    if ref in ref_to_libpart:
                        pinfo = libpart_pins.get(ref_to_libpart[ref], {}).get(str(pin))
                        if pinfo: pin_name_map[(ref, str(pin))] = pinfo

        return dict(nets), {k:set(v) for k,v in comp_pins.items()}, pin_name_map, ref_types

    # S-expression
    nets = defaultdict(set)
    comp_pins = defaultdict(set)
    pin_name_map = {}
    ref_values = {}

    for comp_blk in iter_blocks_exact(text, "comp"):
        m_ref = re.search(_token_alt("ref"), comp_blk)
        if not m_ref: continue
        ref = _pick(m_ref)
        m_val = re.search(r'\(value\s+("([^"]+)"|[^\s\)]+)\)', comp_blk)
        val = _pick(m_val) if m_val else ""
        ref_values[ref] = val

    node_re = re.compile(
        rf'\(node\s+{_token_alt("ref")}\s+{_token_alt("pin")}'
        r'(?:\s+\(pinfunction\s+("([^"]+)"|[^\s\)]+)\))?'
        r'(?:\s+\(pintype\s+("([^"]+)"|[^\s\)]+)\))?'
        r'\s*\)'
    )
    for nb in iter_blocks_exact(text, "net"):
        m_name = re.search(r'\(name\s+("([^"]+)"|[^\s\)]+)\)', nb)
        net_name = _pick(m_name) if m_name else ""
        for nm in node_re.finditer(nb):
            ref = _pick(nm, quoted_idx=2, any_idx=1)
            pin = _pick(nm, quoted_idx=4, any_idx=3)
            if not ref or not pin: continue
            nets[net_name].add((ref, pin))
            comp_pins[ref].add(pin)
            pfunc = _pick(nm, quoted_idx=6, any_idx=5)
            ptype = _pick(nm, quoted_idx=8, any_idx=7)
            if pfunc or ptype:
                pin_name_map[(ref, pin)] = {"name": pfunc, "type": ptype}

    ref_types = {ref: infer_type(ref, ref_values.get(ref,"")) for ref in comp_pins.keys()}
    return dict(nets), {k:set(v) for k,v in comp_pins.items()}, pin_name_map, ref_types

def invert_node_to_net(nets):
    node2net = {}
    for net_name, nodes in nets.items():
        for node in nodes:
            node2net[node] = net_name
    return node2net


# ============================ Failure DB parsing ============================
def auto_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    have_share = any(_norm(c) == "share" for c in df.columns)
    have_fit   = any(_norm(c) == "fit"   for c in df.columns)
    for c in df.columns:
        n = _norm(c)
        if not have_share and (("probabil" in n) or ("share" in n) or ("distribution" in n) or n.startswith("probability")):
            ren[c] = "Share"; have_share = True
        if not have_fit and (n in {"fit","basefit","fitbasefit","lambdafit","lambda","failurerate","rate"}):
            ren[c] = "FIT"; have_fit = True
    return df.rename(columns=ren) if ren else df

def parse_failure_csv_with_mapping(upload):
    raw_bytes = upload.read(); upload.seek(0)
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

    if any(x is None for x in [c_type, c_mode, c_share, c_fit]):
        st.warning("Map missing CSV columns:")
        cols = list(raw.columns)
        c_type = st.selectbox("Component Type column", cols, index=cols.index(c_type) if c_type else 0)
        c_mode = st.selectbox("Failure Mode column",  cols, index=cols.index(c_mode) if c_mode else 0)
        c_share= st.selectbox("Mode Share column",    cols, index=cols.index(c_share) if c_share else 0)
        c_fit  = st.selectbox("FIT column",           cols, index=cols.index(c_fit) if c_fit else 0)
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
                "voltageregulator","microcontroller","ic","other"}
        for p in parts:
            if _norm(p) in base: return p.capitalize()
        return parts[0]

    out = pd.DataFrame()
    out["ComponentType"] = raw[c_type].astype(str).fillna("").map(canon_type_cell)
    out["FailureMode"]   = raw[c_mode].astype(str).fillna("").str.strip()
    out["Share"]         = parse_share_series(raw[c_share])
    out["FIT"]           = parse_number_series(raw[c_fit])
    out["DC"]            = parse_share_series(raw[c_dc]).clip(0,1) if c_dc else 0.0
    if c_det:
        v = raw[c_det].astype(str).str.strip().str.lower()
        out["Detectable"] = v.isin(["1","true","yes","y","t"])
    else:
        out["Detectable"] = False
    out["DiagnosticName"] = raw[c_dnm].astype(str) if (c_dnm and c_dnm in raw.columns) else ""
    out["ct_norm"]        = out["ComponentType"].map(_norm)

    st.success("Failure DB parsed.")
    st.dataframe(out.head(20), height=240)
    return out


# ============================ Build components from NET ============================
def build_component_table_from_net(comp_pins_map: dict, ref_types: dict) -> pd.DataFrame:
    rows = []
    def ref_sort_key(r: str):
        m = re.search(r'(\d+)$', r)
        return (re.sub(r'\d+$', '', r), int(m.group(1)) if m else 0)
    for ref in sorted(comp_pins_map.keys(), key=ref_sort_key):
        rows.append({"RefDes": ref, "ComponentType": ref_types.get(ref, "Other")})
    df = pd.DataFrame(rows, columns=["RefDes","ComponentType"])
    df["ct_norm"] = df["ComponentType"].map(_norm)
    return df


# ============================ LLM label (goal used verbatim) ============================
def label_llm(row: pd.Series, safety_goal_text: str, target_net: str,
              nets, comp_pins_map, pin_name_map, node2net,
              include_voltage_hints: bool = False, **kwargs):
    """
    include_voltage_hints: set True to pass simple rail hints based on net names.
    Also accepts legacy alias 'include_hints' via kwargs.
    """
    if "include_hints" in kwargs and kwargs["include_hints"] is not None:
        include_voltage_hints = bool(kwargs["include_hints"])

    if OAI is None:
        return "NEEDS_REVIEW", "LLM not configured (set OPENAI_API_KEY)."

    ref = str(row["RefDes"])
    pins = sorted(list(comp_pins_map.get(ref, [])))
    pinctx = []
    for p in pins:
        net = node2net.get((ref, p))
        info = pin_name_map.get((ref, p), {})
        pinctx.append({"pin": p, "name": info.get("name",""), "net": net})

    # Optional hints from net names
    supply_candidates = []
    if include_voltage_hints:
        def guess_v(name: str):
            if not name: return None
            n = name.lower()
            if n in {"/b-","/gnd","gnd","ground","0v"}: return 0.0
            if "12v" in n or n in {"/b+"}: return 12.0
            m = re.search(r'(\d+(?:\.\d+)?)v', n)
            try: return float(m.group(1)) if m else None
            except: return None
        for n in {x["net"] for x in pinctx if x.get("net")}:
            v = guess_v(n)
            if v is not None:
                supply_candidates.append({"net": n, "voltage_hint": v})

    prompt = f"""
You are an FMEDA safety expert. Perform single-fault propagation for the EXACT goal below.
Use the goal verbatim.

SAFETY_GOAL: {safety_goal_text}
TARGET_NET: {target_net}

Component:
  RefDes: {ref}
  Type: {row.get('ComponentType')}
  Failure Mode: {row.get('FailureMode')}

Pins from KiCad netlist:
{json.dumps(pinctx, ensure_ascii=False)}

Voltage hints (optional):
{json.dumps(supply_candidates, ensure_ascii=False)}

Decide if THIS SINGLE FAILURE MODE can violate the goal.
Return STRICT JSON:
{{"label":"SAFE|UNSAFE|NEEDS_REVIEW","reason":"<=140 chars"}}
"""
    try:
        try:
            r = OAI.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                response_format={"type":"json_object"}
            )
            content = r.choices[0].message.content
        except Exception:
            r = OAI.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            content = r.choices[0].message.content

        data = json.loads(content)
        lab = str(data.get("label","NEEDS_REVIEW")).upper()
        rsn = str(data.get("reason",""))
        if lab not in {"SAFE","UNSAFE","NEEDS_REVIEW"}:
            return "NEEDS_REVIEW", f"Invalid label: {lab}"
        return lab, rsn
    except Exception as e:
        return "NEEDS_REVIEW", f"LLM error: {e}"


# ============================ SPFM & Sensitivity helpers ============================
def compute_fit_eff(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["FIT"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["Share"], errors="coerce").fillna(0.0)
        * (1.0 - pd.to_numeric(df["DC"], errors="coerce").fillna(0.0).clip(0,1))
    )

def compute_spfm(fmeda: pd.DataFrame, label_col: str = "Label_llm") -> float | None:
    d = fmeda.copy()
    d["FIT_eff"] = compute_fit_eff(d)
    lam_tot = d["FIT_eff"].sum()
    lam_haz = d.loc[d[label_col].str.upper()=="UNSAFE", "FIT_eff"].sum()
    if lam_tot <= 0: return None
    return 1.0 - (lam_haz / lam_tot)

def top_contributors(df: pd.DataFrame, label_col="Label_llm", top_n=10, include_nr=False):
    d = df.copy()
    d["FIT_eff"] = compute_fit_eff(d)
    if include_nr:
        mask = d[label_col].str.upper().isin(["UNSAFE","NEEDS_REVIEW"])
    else:
        mask = d[label_col].str.upper()=="UNSAFE"
    d = d[mask].copy()
    d["hazard_lambda"] = d["FIT_eff"]
    d = d.sort_values("hazard_lambda", ascending=False)
    return d.head(top_n)

def measures_for_row(row, goal_text: str):
    comp = (row.get("ComponentType") or "").lower()
    fm   = (row.get("FailureMode") or "").lower()
    ideas = []
    if "short" in fm:
        ideas += [
            "Add window comparator at OUT with latched shutdown",
            "Series impedance/limiter (R/PTC/fuse) in hazardous path",
            "Clamp (TVS/zener) to hold node on safe side",
            "Rail supervision (OV/UV) forces safe state",
        ]
        if "mosfet" in comp:
            ideas += ["Back-to-back MOSFETs in series (block body diode)",
                      "Gate clamp + proper gate resistors"]
        if "diode" in comp:
            ideas += ["Dual series diodes or add series R before OUT"]
        if "connector" in comp:
            ideas += ["Pin re-ordering/spacing, guarded GND pins, harness fuse"]
    elif "open" in fm:
        ideas += [
            "Fail-safe bias (pull-up/pull-down) so open drives safe state",
            "Loss-of-drive detection + shutdown",
        ]
        if "resistor" in comp:
            ideas += ["Split into series pair; one open keeps safe bias"]
    else:
        ideas += [
            "Self-diagnostics (plausibility), watchdog on driver",
            "Thermal margin or higher-quality grade device",
        ]
    return list(dict.fromkeys(ideas))


# ============================ LLM Sensitivity & Mitigation (Top-K) ============================
def build_hazard_context(df: pd.DataFrame, label_col="Label_llm", include_nr=False, max_rows=20):
    d = df.copy()
    d["FIT"] = pd.to_numeric(d["FIT"], errors="coerce")
    d["Share"] = pd.to_numeric(d["Share"], errors="coerce")
    d["DC"] = pd.to_numeric(d["DC"], errors="coerce").clip(0,1)
    d["FIT_eff"] = d["FIT"].fillna(0.0) * d["Share"].fillna(0.0) * (1.0 - d["DC"].fillna(0.0))
    lam_total = d["FIT_eff"].sum()

    labels = d[label_col].astype(str).str.upper()
    hazard_mask = labels.eq("UNSAFE") | (include_nr & labels.eq("NEEDS_REVIEW"))
    h = d.loc[hazard_mask, ["RefDes","ComponentType","FailureMode","FIT","Share","DC","FIT_eff",label_col,"Reason_llm"]].copy()
    h = h.sort_values("FIT_eff", ascending=False).head(max_rows)

    rows_ctx = []
    for _, r in h.iterrows():
        rows_ctx.append({
            "RefDes": str(r["RefDes"]),
            "ComponentType": str(r["ComponentType"]),
            "FailureMode": str(r["FailureMode"]),
            "Label": str(r[label_col]),
            "Reason": str(r.get("Reason_llm","")),
            "FIT": float(r["FIT"]) if pd.notna(r["FIT"]) else 0.0,
            "Share": float(r["Share"]) if pd.notna(r["Share"]) else 0.0,
            "DC": float(r["DC"]) if pd.notna(r["DC"]) else 0.0,
            "FIT_eff": float(r["FIT_eff"]) if pd.notna(r["FIT_eff"]) else 0.0
        })
    return lam_total, rows_ctx, h

def llm_sensitivity_proposals(goal_text: str, target_net: str, lam_total: float, rows_ctx: list[dict], top_k: int = 5):
    if OAI is None:
        return {"error": "LLM not configured (OPENAI_API_KEY missing)."}, None

    schema_hint = {
      "proposals": [
        {
          "rank": 1,
          "RefDes": "D1",
          "FailureMode": "Short",
          "action": "IncreaseDC",
          "change": {"delta_DC": 0.4},
          "delta_spfm_est": 0.0123,
          "rationale": "why this is effective",
          "measures": ["specific 1","specific 2","specific 3"],
          "effort": "low",
          "side_effects": "short note"
        }
      ],
      "notes": "assumptions and calculation steps"
    }

    prompt = f"""
You are a functional safety engineer. Perform LLM-only sensitivity analysis for SPFM.

SAFETY_GOAL (verbatim): {goal_text}
TARGET_NET: {target_net}

Math (use exactly):
  λ_i = FIT_i * Share_i * (1 - DC_i)
  Λ_tot = sum(FIT_i * Share_i * (1 - DC_i)) over all rows
  Baseline SPFM = 1 - (sum λ_i for UNSAFE rows) / Λ_tot

Small-change estimates for a single row i:
  Increase DC by ΔDC_i:      ΔSPFM ≈ (λ_i / Λ_tot) * ΔDC_i
  Reduce FIT by fraction a:  ΔSPFM ≈ (λ_i / Λ_tot) * a
  Block propagation (make SAFE): ΔSPFM ≈ (λ_i / Λ_tot)
  Redundancy that eliminates the single-point hazard: same as blocking.

Data (JSON):
- Λ_tot: {lam_total:.12g}
- Candidate rows (UNSAFE and optionally NEEDS_REVIEW):
{json.dumps(rows_ctx, ensure_ascii=False)}

Task:
1) Propose the TOP {top_k} actions maximizing ΔSPFM with realistic effort.
2) For each action provide:
   RefDes, FailureMode, action ∈ [IncreaseDC, ReduceFIT, BlockPropagation, Redundancy],
   change (ΔDC or fit_reduction_frac or block:true),
   delta_spfm_est (0..1),
   3–6 concrete design measures, effort, side_effects.
3) Sort desc by delta_spfm_est and rank 1..{top_k}. Max {top_k} items.
4) Return STRICT JSON with keys like this schema:
{json.dumps(schema_hint, ensure_ascii=False)}
"""
    try:
        try:
            r = OAI.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                response_format={"type":"json_object"}
            )
            content = r.choices[0].message.content
        except Exception:
            r = OAI.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            content = r.choices[0].message.content

        data = json.loads(content)
        return data, None
    except Exception as e:
        return None, f"LLM error: {e}"

def verify_proposals_math(proposals: list[dict], rows_ctx: list[dict], lam_total: float):
    idx = {(r["RefDes"], r["FailureMode"]): float(r.get("FIT_eff", 0.0)) for r in rows_ctx}
    checked = []
    for p in proposals:
        key = (str(p.get("RefDes","")), str(p.get("FailureMode","")))
        lam_i = idx.get(key, 0.0)
        change = p.get("change", {}) or {}
        est = None
        if change.get("block") is True:
            est = lam_i / lam_total if lam_total > 0 else 0.0
        elif "delta_DC" in change:
            try: est = (lam_i / lam_total) * float(change["delta_DC"])
            except Exception: est = None
        elif "fit_reduction_frac" in change:
            try: est = (lam_i / lam_total) * float(change["fit_reduction_frac"])
            except Exception: est = None
        checked.append({
            "RefDes": p.get("RefDes",""),
            "FailureMode": p.get("FailureMode",""),
            "action": p.get("action",""),
            "LLM_delta_spfm_est": p.get("delta_spfm_est", None),
            "Our_delta_spfm_est": est,
            "Measures": p.get("measures", []),
            "effort": p.get("effort",""),
            "side_effects": p.get("side_effects",""),
        })
    return pd.DataFrame(checked)

def apply_proposals_to_scenario(df: pd.DataFrame, proposals: list[dict], label_col="Label_llm"):
    scen = df.copy()
    scen["FIT"] = pd.to_numeric(scen["FIT"], errors="coerce")
    scen["Share"] = pd.to_numeric(scen["Share"], errors="coerce")
    scen["DC"] = pd.to_numeric(scen["DC"], errors="coerce").clip(0,1)
    scen[label_col] = scen[label_col].astype(str).str.upper()

    for p in proposals:
        ref = str(p.get("RefDes",""))
        fm  = str(p.get("FailureMode",""))
        change = p.get("change", {}) or {}
        mask = (scen["RefDes"].astype(str)==ref) & (scen["FailureMode"].astype(str)==fm)
        if not mask.any(): continue
        if change.get("block") is True:
            scen.loc[mask, label_col] = "SAFE"
        if "delta_DC" in change:
            scen.loc[mask, "DC"] = (scen.loc[mask, "DC"] + float(change["delta_DC"])).clip(0,1)
        if "fit_reduction_frac" in change:
            scen.loc[mask, "FIT"] = scen.loc[mask, "FIT"] * (1.0 - float(change["fit_reduction_frac"]))

    scen["FIT_eff"] = scen["FIT"].fillna(0.0) * scen["Share"].fillna(0.0) * (1.0 - scen["DC"].fillna(0.0))
    scen_spfm = compute_spfm(scen, label_col)
    return scen, scen_spfm


# ============================ UI ============================
st.title("Safety Co-Pilot — FMEDA + Sensitivity (LLM only)")

safety_goal = st.text_input("Safety Goal (verbatim, used as-is by the LLM)",
                            "Prevent unintended >5 V at OUT")
st.session_state["include_hints"] = st.toggle("Include voltage hints from net names",
                                              value=st.session_state.get("include_hints", False),
                                              key="include_hints_toggle")

left, right = st.columns([1,1])
with left:
    net_file = st.file_uploader("Upload KiCad netlist (.net — S-expression or XML)", type=["net","xml"])
with right:
    fail_file = st.file_uploader("Upload Failure Rates & Modes (CSV)", type=["csv"])

# ---- Parse inputs (store to state so reruns don't lose them) ----
if net_file is not None:
    net_bytes = net_file.read()
    try:
        nets, comp_pins_map, pin_name_map, ref_types = parse_kicad_net(net_bytes)
    except Exception as e:
        st.error(f"Failed to parse .net: {e}")
        st.stop()
    node2net = invert_node_to_net(nets)
    st.success(f"Parsed NET: {len(nets)} nets • {len(comp_pins_map)} components.")
    st.session_state.nets = nets
    st.session_state.comp_pins_map = comp_pins_map
    st.session_state.pin_name_map = pin_name_map
    st.session_state.node2net = node2net

if fail_file is not None:
    fail_df = parse_failure_csv_with_mapping(fail_file)
    st.session_state.fail_df_snapshot = fail_df

# ---- If we have a net, let user edit types and choose target net ----
if st.session_state.nets:
    # quick type guess by ref prefix; user can edit
    comp_df = build_component_table_from_net(st.session_state.comp_pins_map,
                                             ref_types={r:"Other" for r in st.session_state.comp_pins_map})
    infer_types = []
    for ref in comp_df["RefDes"]:
        infer_types.append(infer_type(ref, ""))
    comp_df["ComponentType"] = infer_types
    comp_df["ct_norm"] = comp_df["ComponentType"].map(_norm)

    st.subheader("NET components (edit types if needed)")
    comp_df = st.data_editor(
        comp_df[["RefDes","ComponentType","ct_norm"]].rename(columns={"ct_norm":"_norm (read-only)"}),
        disabled=["_norm (read-only)"],
        height=280,
        key="comp_editor"
    )
    comp_df["ct_norm"] = comp_df["ComponentType"].map(_norm)

    net_names = sorted(st.session_state.nets.keys(), key=lambda x: x.lower())
    default_idx = 0
    for i, n in enumerate(net_names):
        if "out" in n.lower():
            default_idx = i; break
    target_net = st.selectbox("Target net (from NET)", net_names, index=default_idx if net_names else 0, key="target_net_sel")

else:
    comp_df = None
    target_net = None

# ---- Build button ----
st.divider()
if st.button("Build FMEDA with LLM", type="primary", use_container_width=True):
    if not (st.session_state.nets and st.session_state.fail_df_snapshot is not None and comp_df is not None):
        st.error("Please upload both .net and Failure DB CSV first.")
        st.stop()

    # Merge by normalized ComponentType
    failure_df = st.session_state.fail_df_snapshot.copy()
    fmeda = comp_df.merge(failure_df, how="left", on="ct_norm", suffixes=("_comp","_db"))
    if "ComponentType_comp" in fmeda.columns:
        fmeda = fmeda.rename(columns={"ComponentType_comp":"ComponentType"})
    fmeda = fmeda[["RefDes","ComponentType","FailureMode","Share","FIT","DC","Detectable","DiagnosticName"]]

    # Fill Share inside each RefDes×Type group if missing
    fmeda["Share"] = pd.to_numeric(fmeda["Share"], errors="coerce")
    mask_na = fmeda["Share"].isna()
    if mask_na.any():
        eq = (
            fmeda[mask_na]
            .groupby(["RefDes","ComponentType"], dropna=False)["Share"]
            .transform(lambda s: 1.0 / len(s) if len(s) else 1.0)
        )
        fmeda.loc[mask_na, "Share"] = eq

    fmeda["DC"] = pd.to_numeric(fmeda["DC"], errors="coerce").fillna(0.0).clip(0,1)
    fmeda["FIT_eff"] = compute_fit_eff(fmeda)

    # LLM labels
    if OAI is None:
        st.error("LLM not configured. Add OPENAI_API_KEY to Streamlit secrets.")
        st.stop()

    nets = st.session_state.nets
    comp_pins_map = st.session_state.comp_pins_map
    pin_name_map = st.session_state.pin_name_map
    node2net = st.session_state.node2net
    selected_target = st.session_state.get("target_net_sel")

    labels, reasons = [], []
    prog = st.progress(0.0, text="Classifying with LLM…")
    total = len(fmeda)
    for idx, (_, row) in enumerate(fmeda.iterrows(), start=1):
        lab, rsn = label_llm(
            row, safety_goal, selected_target,
            nets, comp_pins_map, pin_name_map, node2net,
            include_voltage_hints=st.session_state.get("include_hints", False)
        )
        labels.append(lab); reasons.append(rsn)
        prog.progress(idx/total, text=f"Classifying with LLM… {idx}/{total}")
        time.sleep(0.01)
    fmeda["Label_llm"]  = labels
    fmeda["Reason_llm"] = reasons

    base_spfm = compute_spfm(fmeda, "Label_llm")

    # Save to state
    st.session_state.fmeda = fmeda
    st.session_state.base_spfm = base_spfm
    st.session_state.target_net = selected_target
    st.session_state.comp_df_snapshot = comp_df.copy()
    st.session_state.built = True

# ============================ RESULTS (sticky) ============================
if st.session_state.built and isinstance(st.session_state.fmeda, pd.DataFrame):
    fmeda = st.session_state.fmeda.copy()
    base_spfm = st.session_state.base_spfm
    target_net = st.session_state.target_net
    nets = st.session_state.nets
    comp_pins_map = st.session_state.comp_pins_map
    pin_name_map = st.session_state.pin_name_map
    node2net = st.session_state.node2net

    st.metric("Baseline SPFM (LLM)", f"{(base_spfm*100):.2f}%" if base_spfm is not None else "n/a")
    st.subheader("FMEDA — LLM result")
    st.dataframe(
        fmeda[["RefDes","ComponentType","FailureMode","Share","FIT","DC","Label_llm","Reason_llm","FIT_eff"]],
        height=480, use_container_width=True
    )
    csv_bytes = fmeda.to_csv(index=False).encode("utf-8")
    st.download_button("Download FMEDA (CSV)", csv_bytes, "fmeda_llm_only.csv","text/csv")

    # ---------------------- Manual Sensitivity (optional) ----------------------
    st.header("Sensitivity Analysis (manual what-ifs)")
    with st.expander("How it works", expanded=False):
        st.markdown(
            "- **FIT_eff = FIT × Share × (1 − DC)** per row.\n"
            "- Rows labeled **UNSAFE** by the LLM contribute to the hazardous sum.\n"
            "- Adjust **DC**, **FIT**, or **block propagation** for top contributors; we recompute SPFM.\n"
        )
    include_nr = st.toggle("Treat NEEDS_REVIEW rows as hazardous in sensitivity", value=False, key="sens_inc_nr")
    top_n = st.slider("Top contributors to adjust", min_value=3, max_value=30, value=10, step=1, key="sens_topn")

    top = top_contributors(fmeda, "Label_llm", top_n=top_n, include_nr=include_nr)
    if top.empty:
        st.info("No contributors selected. If most rows are SAFE, widen Top N or include NEEDS_REVIEW.")
    else:
        st.write("Top contributors by hazardous λ (descending):")
        st.dataframe(top[["RefDes","ComponentType","FailureMode","FIT","Share","DC","Label_llm","hazard_lambda"]],
                     use_container_width=True, height=320)

        st.subheader("Scenario knobs")
        colA, colB, colC = st.columns(3)
        with colA:
            global_dc_up = st.slider("Global DC increase on selected rows", 0.0, 0.9, 0.0, 0.05, key="sens_glob_dc")
        with colB:
            global_fit_down_pct = st.slider("Global FIT reduction (%) on selected rows", 0, 90, 0, 5, key="sens_glob_fit")
        with colC:
            block_all = st.checkbox("Block propagation for ALL selected rows (make SAFE)", key="sens_block_all")

        per_row_changes = {}
        st.caption("Optional per-row overrides:")
        for i, r in top.reset_index(drop=True).iterrows():
            with st.expander(f"{r.RefDes} — {r.ComponentType} — {r.FailureMode}", expanded=False):
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    dc_up = st.slider(f"ΔDC for {r.RefDes}", 0.0, 0.9, global_dc_up, 0.05, key=f"dc_{i}")
                with c2:
                    fit_down = st.slider(f"FIT ↓ (%) for {r.RefDes}", 0, 90, global_fit_down_pct, 5, key=f"fit_{i}")
                with c3:
                    block = st.checkbox(f"Block propagation ({r.RefDes})", value=block_all, key=f"blk_{i}")
                per_row_changes[(r.RefDes, r.FailureMode)] = {"dc_up": dc_up, "fit_down_pct": fit_down, "block": block}

        if st.button("Recompute scenario SPFM", key="sens_recompute"):
            scen = fmeda.copy()
            scen["FIT"] = pd.to_numeric(scen["FIT"], errors="coerce")
            scen["Share"] = pd.to_numeric(scen["Share"], errors="coerce")
            scen["DC"] = pd.to_numeric(scen["DC"], errors="coerce").clip(0,1)
            scen["Label_llm"] = scen["Label_llm"].astype(str).str.upper()

            for (ref, fm), ch in per_row_changes.items():
                mask = (scen["RefDes"]==ref) & (scen["FailureMode"]==fm)
                if ch["block"]:
                    scen.loc[mask, "Label_llm"] = "SAFE"
                scen.loc[mask, "DC"] = (scen.loc[mask, "DC"] + ch["dc_up"]).clip(0,1)
                k = 1.0 - (ch["fit_down_pct"]/100.0)
                scen.loc[mask, "FIT"] = scen.loc[mask, "FIT"] * k

            scen["FIT_eff"] = compute_fit_eff(scen)
            scen_spfm = compute_spfm(scen, "Label_llm")

            st.metric("Scenario SPFM (LLM)", f"{(scen_spfm*100):.2f}%" if scen_spfm is not None else "n/a",
                      delta=f"{((scen_spfm - base_spfm)*100):+.2f}%" if (scen_spfm is not None and base_spfm is not None) else None)

            st.write("Changed rows:")
            changed = scen.merge(fmeda, on=["RefDes","FailureMode"], suffixes=("_scen","_base"))
            changed = changed[
                (changed["DC_scen"]!=changed["DC_base"]) |
                (changed["FIT_scen"]!=changed["FIT_base"]) |
                (changed["Label_llm_scen"]!=changed["Label_llm_base"])
            ][["RefDes","ComponentType_base","FailureMode",
               "FIT_base","FIT_scen","DC_base","DC_scen","Label_llm_base","Label_llm_scen","FIT_eff_base","FIT_eff_scen"]]
            st.dataframe(changed, use_container_width=True, height=360)
            st.download_button("Download scenario FMEDA (CSV)",
                               scen.to_csv(index=False).encode("utf-8"),
                               "fmeda_scenario.csv","text/csv")

    # ---------------------- Design Measures (advice only) ----------------------
    st.header("Design Measures")
    st.caption("Target the few top contributors. Mix detection (DC↑), path blocking, and FIT reduction.")
    top_for_measures = top_contributors(fmeda, "Label_llm", top_n=8, include_nr=False)

    fallback = []
    for _, r in top_for_measures.iterrows():
        fallback.append({
            "RefDes": r.RefDes,
            "FailureMode": r.FailureMode,
            "Measures": measures_for_row(r, safety_goal)
        })

    status_msg = st.empty()
    use_llm_measures = st.toggle("Ask LLM for tailored measures",
                                 value=st.session_state.get("use_llm_measures", False),
                                 key="use_llm_measures",
                                 help="Generates text advice only; no SPFM change.")

    final_measures = []
    if use_llm_measures and OAI is not None and not top_for_measures.empty:
        rows_json = []
        for _, r in top_for_measures.iterrows():
            rows_json.append({
                "RefDes": r.RefDes,
                "ComponentType": r.ComponentType,
                "FailureMode": r.FailureMode,
                "HazardLambda": float(r.hazard_lambda),
                "ReasonLLM": r.get("Reason_llm","")
            })
        try:
            prompt = f"""You are a safety engineer. For each of these rows, propose 3–6 concise, realistic design measures.
SAFETY_GOAL: {safety_goal}
ROWS: {json.dumps(rows_json, ensure_ascii=False)}
Return JSON list of {{"RefDes":"","FailureMode":"","Measures":["..."]}} only."""
            try:
                rrr = OAI.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0,
                    response_format={"type":"json_object"}
                )
                content = rrr.choices[0].message.content
            except Exception:
                rrr = OAI.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0
                )
                content = rrr.choices[0].message.content
            items = json.loads(content)
            used = 0
            if isinstance(items, list):
                by_key = {(d.get("RefDes",""), d.get("FailureMode","")): d for d in items}
                for f in fallback:
                    k = (f["RefDes"], f["FailureMode"])
                    if k in by_key and by_key[k].get("Measures"):
                        final_measures.append(by_key[k]); used += 1
                    else:
                        final_measures.append(f)
                status_msg.info(f"LLM measures applied for {used}/{len(fallback)} rows.")
            else:
                final_measures = fallback
                status_msg.warning("LLM returned no valid list; showing fallback.")
        except Exception:
            final_measures = fallback
            status_msg.warning("LLM error; showing fallback.")
    else:
        final_measures = fallback
        if use_llm_measures and OAI is None:
            status_msg.warning("OPENAI_API_KEY not configured; showing fallback.")

    meas_df = pd.DataFrame([{"RefDes":m["RefDes"], "FailureMode":m["FailureMode"],
                             "Measures": " • ".join(m["Measures"])} for m in final_measures])
    st.dataframe(meas_df, use_container_width=True, height=320)
    st.download_button("Download measures (CSV)",
                       meas_df.to_csv(index=False).encode("utf-8"),
                       "design_measures.csv","text/csv")

    # ---------------------- LLM Sensitivity & Mitigation Proposals (Top-K by ΔSPFM) ----------------------
    st.header("LLM Sensitivity & Mitigation Proposals (Top-K by effectiveness)")

    include_nr_for_llm = st.toggle("Allow LLM to consider NEEDS_REVIEW rows as hazardous", value=False, key="llm_inc_nr")
    max_candidates = st.slider("Rows to provide to LLM (ranked by λ)", 5, 40, 20, 1, key="llm_rows")
    top_k_actions  = st.slider("Top actions to request from LLM", 3, 10, 5, 1, key="llm_topk")

    lam_total, rows_ctx, _top_df = build_hazard_context(
        fmeda, label_col="Label_llm", include_nr=include_nr_for_llm, max_rows=max_candidates
    )

    if OAI is None:
        st.warning("LLM not configured. Set OPENAI_API_KEY in secrets to enable this feature.")
    else:
        if st.button("Ask LLM for Top actions", key="llm_ask_actions"):
            llm_out, err = llm_sensitivity_proposals(safety_goal, target_net, lam_total, rows_ctx, top_k=top_k_actions)
            if err:
                st.error(err)
            elif not llm_out or "proposals" not in llm_out:
                st.error("LLM returned no proposals or invalid JSON.")
            else:
                props = llm_out.get("proposals", [])
                st.subheader("LLM Top actions (raw JSON)")
                st.json(llm_out, expanded=False)

                vdf = verify_proposals_math(props, rows_ctx, lam_total)
                vdf["LLM_delta_spfm_est_%"] = pd.to_numeric(vdf["LLM_delta_spfm_est"], errors="coerce").fillna(0.0) * 100.0
                vdf["Our_delta_spfm_est_%"] = pd.to_numeric(vdf["Our_delta_spfm_est"], errors="coerce").fillna(0.0) * 100.0
                st.subheader("Checked actions (ΔSPFM estimates)")
                st.dataframe(
                    vdf[["RefDes","FailureMode","action","LLM_delta_spfm_est_%","Our_delta_spfm_est_%","effort","side_effects","Measures"]],
                    use_container_width=True, height=360
                )

                if st.button("Apply Top actions and recompute SPFM", key="llm_apply_actions"):
                    scen, scen_spfm = apply_proposals_to_scenario(fmeda, props, label_col="Label_llm")
                    st.metric(
                        "Scenario SPFM (after applying LLM actions)",
                        f"{(scen_spfm*100):.2f}%" if scen_spfm is not None else "n/a",
                        delta=f"{((scen_spfm - base_spfm)*100):+.2f}%" if (scen_spfm is not None and base_spfm is not None) else None
                    )
                    st.dataframe(
                        scen[["RefDes","ComponentType","FailureMode","FIT","Share","DC","Label_llm","FIT_eff"]],
                        use_container_width=True, height=360
                    )
                    st.download_button("Download scenario FMEDA (CSV)",
                                       scen.to_csv(index=False).encode("utf-8"),
                                       "fmeda_llm_scenario.csv","text/csv")

else:
    st.info("Upload a KiCad .net and a Failure DB CSV to start.")
