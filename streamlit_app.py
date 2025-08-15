import streamlit as st
import pandas as pd
import re, io, csv, json
from collections import defaultdict

# ============================ App config ============================
st.set_page_config(page_title="FMEDA (KiCad .net) — Rules vs LLM", layout="wide")
OPENAI_MODEL = "gpt-4o"  # only used if LLM comparison is enabled

# Optional OpenAI client (LLM is optional)
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

# ============================ Small utils ============================
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

# ============================ S-expression helpers ============================
def _iter_blocks(text: str, tag: str):
    """Yield s-expr blocks starting with '({tag}' at balanced parens depth."""
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

# ============================ Type inference ============================
REF_PREFIX = {
    "R":"Resistor","C":"Capacitor","L":"Inductor","D":"Diode","Z":"Zener",
    "Q":"MOSFET","T":"BJT","U":"IC","A":"OpAmp","K":"Relay","F":"Fuse",
    "J":"Connector","X":"Connector",
}
def infer_type_from_ref_or_value(ref: str, value: str) -> str:
    m = re.match(r'^([A-Za-z]+)', ref or "")
    if m:
        pref = m.group(1).upper()
        for p in sorted(REF_PREFIX, key=lambda x: -len(x)):
            if pref.startswith(p): return REF_PREFIX[p]
    t = (value or "").lower()
    if re.search(r'(^|\s)\d+(\.\d+)?(r|k|m)?(\s*ohm|$)', t): return "Resistor"
    if re.search(r'(^|\s)\d+(\.\d+)?(n|u|µ|p|f)(\s*|$)', t): return "Capacitor"
    if "opamp" in t: return "OpAmp"
    if "comparator" in t or "comp" in t: return "Comparator"
    if "mos" in t or "nmos" in t or "pmos" in t: return "MOSFET"
    if "diode" in t or t.startswith("1n") or t.startswith("sb"): return "Diode"
    return "Other"

# ============================ NET parser (S-expression + XML fallback; quoted-safe) ============================
def _token_alt(key):
    # matches (key "X") or (key X)
    return rf'\({key}\s+("([^"]+)"|[^\s\)]+)\)'

def _pick(m, quoted_idx=2, any_idx=1):
    """
    Safely pick a capture from a regex Match:
    - prefer the quoted capture (e.g., group 2) else the unquoted token (group 1)
    - return "" if neither exists (avoids calling .strip on None)
    """
    if m is None:
        return ""
    try:
        val_q = m.group(quoted_idx)
    except IndexError:
        val_q = None
    try:
        val_a = m.group(any_idx)
    except IndexError:
        val_a = None
    val = val_q if val_q is not None else val_a
    if val is None:
        return ""
    return val.strip('"')

def parse_kicad_net(net_bytes: bytes):
    """
    Supports KiCad 6/7/8/9 .net in S-expression and XML.
    Returns:
      nets:          dict[str, set[(ref,pin)]]
      comp_pins_map: dict[ref] -> set(pin)
      pin_name_map:  dict[(ref,pin)] -> {"name": str, "type": str}
      ref_types:     dict[ref] -> inferred component type
    """
    text = net_bytes.decode("utf-8", errors="ignore").strip()

    # ---------- XML path ----------
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
            val = (comp.findtext("./value") or "").strip().lower()
            ref_types[ref] = infer_type_from_ref_or_value(ref, val)
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
                if num:
                    pins[str(num)] = {"name": name, "type": ptype}
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

    # ---------- S-expression path (quoted-safe) ----------
    nets = defaultdict(set)
    comp_pins = defaultdict(set)
    pin_name_map = {}
    ref_types = {}

    # components: references + values (for type inference)
    for comp_blk in _iter_blocks(text, "comp"):
        m_ref = re.search(_token_alt("ref"), comp_blk)
        if not m_ref: continue
        ref = _pick(m_ref)
        m_val = re.search(r'\(value\s+("([^"]+)"|[^\s\)]+)\)', comp_blk)
        val = _pick(m_val) if m_val else ""
        ref_types[ref] = infer_type_from_ref_or_value(ref, val)

    # nets → nodes (capture optional pinfunction/pintype)
    node_re = re.compile(
        rf'\(node\s+{_token_alt("ref")}\s+{_token_alt("pin")}'
        r'(?:\s+\(pinfunction\s+("([^"]+)"|[^\s\)]+)\))?'
        r'(?:\s+\(pintype\s+("([^"]+)"|[^\s\)]+)\))?'
        r'\s*\)'
    )
    for net_blk in _iter_blocks(text, "net"):
        m_name = re.search(r'\(name\s+("([^"]+)"|[^\s\)]+)\)', net_blk)
        net_name = _pick(m_name) if m_name else ""
        for nm in node_re.finditer(net_blk):
            ref = _pick(nm, quoted_idx=2, any_idx=1)
            pin = _pick(nm, quoted_idx=4, any_idx=3)
            if not ref or not pin:
                continue
            nets[net_name].add((ref, pin))
            comp_pins[ref].add(pin)
            pfunc = _pick(nm, quoted_idx=6, any_idx=5)
            ptype = _pick(nm, quoted_idx=8, any_idx=7)
            if pfunc or ptype:
                pin_name_map[(ref, pin)] = {"name": pfunc, "type": ptype}

    return dict(nets), {k:set(v) for k,v in comp_pins.items()}, pin_name_map, ref_types

def invert_node_to_net(nets: dict[str,set[tuple]]):
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
        if not have_share and (("probabil" in n) or (n in {"probability","probabilitypct","probabilitypercent"}) or ("share" in n) or ("distribution" in n)):
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

# ============================ Supplies & thresholds ============================
SUPPLY_DEFAULTS = {
    "vcc": 5.0, "vdd": 5.0, "vin": 12.0, "vbatt": 12.0, "vbat": 12.0,
    "+5v": 5.0, "5v": 5.0, "3v3": 3.3, "3.3v": 3.3, "+3v3": 3.3, "+12v": 12.0, "12v": 12.0,
    "avcc": 5.0, "dvcc": 5.0
}
def voltage_from_netname(name: str) -> float | None:
    if not name: return None
    n = _norm(name)
    if n in SUPPLY_DEFAULTS: return SUPPLY_DEFAULTS[n]
    m = re.search(r'(\d+(?:\.\d+)?)v', n)  # 5v, 12v
    if m:
        try: return float(m.group(1))
        except Exception: pass
    m2 = re.search(r'(\d)v(\d)', n)  # 3v3
    if m2:
        return float(f"{m2.group(1)}.{m2.group(2)}")
    return None

def extract_threshold_from_goal(text: str, default: float = 5.0) -> float:
    if not text: return default
    m = re.search(r'(\d+(?:\.\d+)?)\s*v', text.lower())
    return float(m.group(1)) if m else default

# ============================ Pin helpers ============================
def pin_name(pinfo):  return (pinfo or {}).get("name","")
def net_of(node2net, ref, pin): return node2net.get((ref, str(pin)))

def find_by_pin_names(pin_name_map, ref, names_like):
    want = {s.lower() for s in names_like}
    out = set()
    for (r,p), info in pin_name_map.items():
        if r != ref: continue
        nm = pin_name(info).lower()
        if nm in want or any(nm.startswith(w) for w in want):
            out.add(p)
    return out

def diode_pins(pin_name_map, ref):
    a = find_by_pin_names(pin_name_map, ref, ["a","anode"])
    k = find_by_pin_names(pin_name_map, ref, ["k","cathode"])
    return a, k

def mosfet_pins(pin_name_map, ref):
    d = find_by_pin_names(pin_name_map, ref, ["d","drain"])
    s = find_by_pin_names(pin_name_map, ref, ["s","source"])
    g = find_by_pin_names(pin_name_map, ref, ["g","gate"])
    return d, s, g

def bjt_pins(pin_name_map, ref):
    c = find_by_pin_names(pin_name_map, ref, ["c","collector"])
    e = find_by_pin_names(pin_name_map, ref, ["e","emitter"])
    b = find_by_pin_names(pin_name_map, ref, ["b","base"])
    return c, e, b

def supply_pins(pin_name_map, ref):
    tokens = ["vcc","vdd","vss","vee","v+","v-","avcc","dvcc","vref","vref+","vref-","vbat"]
    return find_by_pin_names(pin_name_map, ref, tokens)

def output_pins(pin_name_map, ref):
    return find_by_pin_names(pin_name_map, ref, ["out","vo","vout"])

# ============================ Deterministic propagation (NET-only) ============================
def failure_creates_short(failure_mode: str) -> bool:
    if not isinstance(failure_mode, str): return False
    fm = failure_mode.lower()
    return any(k in fm for k in ["short", "s/c", "sc"])

def failure_is_open(failure_mode: str) -> bool:
    if not isinstance(failure_mode, str): return False
    fm = failure_mode.lower()
    return any(k in fm for k in ["open", "o/c", "oc"])

def failure_short_to_named(failure_mode: str) -> str | None:
    if not isinstance(failure_mode, str): return None
    fm = failure_mode.lower()
    m = re.search(r'short\s*(to|2)\s*([a-z0-9\+\-\.]+)', fm)
    return m.group(2) if m else None

def comp_is_two_terminal(comp_pins_map, ref) -> bool:
    return len(comp_pins_map.get(ref, set())) == 2

def label_rules(row, nets, comp_pins_map, pin_name_map, node2net,
                target_net: str, hazard_v: float):
    ref   = str(row["RefDes"])
    ctype = (row.get("ComponentType") or "").lower()
    fm    = (row.get("FailureMode") or "").lower()

    if ref not in comp_pins_map:
        return "UNASSESSED", "Component not in netlist (NET-only mode)"
    if target_net not in nets:
        return "UNASSESSED", f"Target net '{target_net}' not found"

    if failure_is_open(fm):
        return "SAFE", "Open fault cannot raise target voltage (overvoltage goal)"
    if not failure_creates_short(fm):
        return "NEEDS_REVIEW", "Non-short failure not modeled; manual review"

    def pins_bridge_high_to_target(pinsA, pinsB):
        for pa in pinsA:
            na = net_of(node2net, ref, pa)
            if not na: continue
            va = voltage_from_netname(na)
            if va is None or va < hazard_v: continue
            for pb in pinsB:
                nb = net_of(node2net, ref, pb)
                if nb == target_net:
                    return True, f"Short connects {na} ({va:.2f}V) to {target_net}"
        return False, ""

    named = failure_short_to_named(fm)
    if named:
        v_named = voltage_from_netname(named)
        if v_named is None and _norm(named) in {"gnd","ground","0v"}:
            v_named = 0.0
        if v_named is not None and v_named >= hazard_v:
            if any(net_of(node2net, ref, p) == target_net for p in comp_pins_map.get(ref, [])):
                return "UNSAFE", f"Explicit '{row['FailureMode']}' injects {v_named:.2f}V into {target_net}"

    if "diode" in ctype or "zener" in ctype:
        a, k = diode_pins(pin_name_map, ref)
        if a and k:
            hit, why = pins_bridge_high_to_target(a, k)
            if hit: return "UNSAFE", f"Diode A-K short: {why}"
            hit, why = pins_bridge_high_to_target(k, a)
            if hit: return "UNSAFE", f"Diode K-A short: {why}"
        if comp_is_two_terminal(comp_pins_map, ref):
            pins = list(comp_pins_map[ref])
            n0, n1 = net_of(node2net, ref, pins[0]), net_of(node2net, ref, pins[1])
            other  = n1 if n0 == target_net else n0
            v      = voltage_from_netname(other)
            if (n0 == target_net or n1 == target_net) and (v is not None and v >= hazard_v):
                return "UNSAFE", f"Two-terminal short bridges {v:.2f}V to {target_net}"
        return "SAFE", "No high-rail-to-target bridge for diode short"

    if "mosfet" in ctype:
        d, s, g = mosfet_pins(pin_name_map, ref)
        if d and s:
            hit, why = pins_bridge_high_to_target(d, s)
            if hit: return "UNSAFE", f"MOSFET D-S short: {why}"
            hit, why = pins_bridge_high_to_target(s, d)
            if hit: return "UNSAFE", f"MOSFET S-D short: {why}"
        return "NEEDS_REVIEW", "MOSFET short not clearly D-S; review"

    if "bjt" in ctype:
        c, e, b = bjt_pins(pin_name_map, ref)
        if c and e:
            hit, why = pins_bridge_high_to_target(c, e)
            if hit: return "UNSAFE", f"BJT C-E short: {why}"
            hit, why = pins_bridge_high_to_target(e, c)
            if hit: return "UNSAFE", f"BJT E-C short: {why}"
        return "NEEDS_REVIEW", "BJT short not clearly C-E; review"

    if "opamp" in ctype or "comparator" in ctype:
        outs = output_pins(pin_name_map, ref)
        rails= supply_pins(pin_name_map, ref)
        if outs:
            out_nets = {net_of(node2net, ref, p) for p in outs}
            if target_net in out_nets and rails:
                for rp in rails:
                    nr = net_of(node2net, ref, rp)
                    vr = voltage_from_netname(nr)
                    if vr is not None and vr >= hazard_v:
                        return "UNSAFE", f"Output short to rail {nr} ({vr:.2f}V) at {target_net}"
            if target_net in out_nets:
                return "NEEDS_REVIEW", "OpAmp output at target; generic short -> review"
        return "SAFE", "No output-to-high-rail bridge"

    if "connector" in ctype:
        pins = list(comp_pins_map[ref])
        tgt_pins = [p for p in pins if net_of(node2net, ref, p) == target_net]
        if tgt_pins:
            for p in pins:
                n = net_of(node2net, ref, p)
                v = voltage_from_netname(n)
                if n != target_net and v is not None and v >= hazard_v:
                    return "UNSAFE", f"Connector pin short from {n} ({v:.2f}V) to {target_net}"
        return "NEEDS_REVIEW", "Connector short unclear; review harness"

    if comp_is_two_terminal(comp_pins_map, ref):
        pins = list(comp_pins_map[ref])
        n0, n1 = net_of(node2net, ref, pins[0]), net_of(node2net, ref, pins[1])
        if n0 == target_net or n1 == target_net:
            other = n1 if n0 == target_net else n0
            v_other = voltage_from_netname(other)
            if v_other is not None and v_other >= hazard_v:
                return "UNSAFE", f"Short connects {other} ({v_other:.2f}V) to {target_net}"
            return "SAFE", f"No high supply on opposite net ({other})"
        return "NEEDS_REVIEW", "Two-terminal short not on target"

    # Conservative multipin fallback
    part_pins = comp_pins_map.get(ref, set())
    part_nets = {net_of(node2net, ref, p) for p in part_pins}
    touches_tgt = target_net in part_nets
    touches_hi  = any((voltage_from_netname(n) or -1) >= hazard_v for n in part_nets if n)
    if touches_tgt and touches_hi:
        return "UNSAFE", "Multi-pin short may connect high rail to target (conservative)"
    if touches_hi:
        return "NEEDS_REVIEW", "Touches high rail but not target; path unclear"
    return "SAFE", "No high rail on part nets"

# ============================ LLM propagation (optional) ============================
def label_llm(row, safety_goal: str, target_net: str,
              nets, comp_pins_map, pin_name_map, node2net, hazard_v: float):
    if OAI is None:
        return "NEEDS_REVIEW", "LLM not configured"
    ref = str(row["RefDes"])
    pins = sorted(list(comp_pins_map.get(ref, [])))
    pinctx = []
    for p in pins:
        net = node2net.get((ref, p))
        info = pin_name_map.get((ref, p), {})
        pinctx.append({"pin": p, "name": info.get("name",""), "net": net})
    supply_candidates = []
    for n in {x["net"] for x in pinctx if x.get("net")}:
        v = voltage_from_netname(n)
        if v is not None and v >= hazard_v:
            supply_candidates.append({"net": n, "voltage": v})

    prompt = f"""
You are an FMEDA safety expert performing single-fault propagation for an OVERVOLTAGE goal.

Safety Goal: {safety_goal}
Target net: {target_net}
Hazard threshold (V): {hazard_v}

Component:
  RefDes: {ref}
  Type: {row.get('ComponentType')}
  Failure Mode: {row.get('FailureMode')}

Pins (from KiCad .net):
{json.dumps(pinctx, ensure_ascii=False)}

Candidate high rails (parsed from net names):
{json.dumps(supply_candidates, ensure_ascii=False)}

Task: Decide whether THIS SINGLE FAILURE MODE can violate the goal
(i.e., cause target net to exceed threshold), using conservative hardware reasoning.
Return strict JSON only:
  {{"label":"SAFE|UNSAFE|NEEDS_REVIEW","reason":"short explanation"}}
"""
    try:
        r = OAI.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        data = json.loads(r.choices[0].message.content)
        lab = str(data.get("label","NEEDS_REVIEW")).upper()
        rsn = data.get("reason","")
        if lab not in {"SAFE","UNSAFE","NEEDS_REVIEW"}:
            lab = "NEEDS_REVIEW"
        return lab, rsn
    except Exception as e:
        return "NEEDS_REVIEW", f"LLM error: {e}"

# ============================ FMEDA expansion (NET-only) ============================
def build_component_table_from_net(comp_pins_map: dict, ref_types: dict) -> pd.DataFrame:
    rows = []
    def ref_sort_key(r: str):
        m = re.search(r'(\d+)$', r)
        return (re.sub(r'\d+$', '', r), int(m.group(1)) if m else 0)
    for ref in sorted(comp_pins_map.keys(), key=ref_sort_key):
        rows.append({"RefDes": ref, "ComponentType": ref_types.get(ref, infer_type_from_ref_or_value(ref, ""))})
    df = pd.DataFrame(rows, columns=["RefDes","ComponentType"])
    df["ct_norm"] = df["ComponentType"].map(_norm)
    return df

def expand_fmeda(components_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    if "ct_norm" not in components_df.columns:
        components_df["ct_norm"] = components_df["ComponentType"].map(_norm)
    if "ct_norm" not in failure_df.columns:
        failure_df["ct_norm"] = failure_df["ComponentType"].map(_norm)
    merged = components_df.merge(failure_df, how="left", on="ct_norm", suffixes=("_comp","_db"))
    out = pd.DataFrame({
        "RefDes":        merged["RefDes"],
        "ComponentType": merged["ComponentType_comp"],
        "FailureMode":   merged["FailureMode"],
        "Share":         merged["Share"],
        "FIT":           merged["FIT"],
        "DC":            merged["DC"],
        "Detectable":    merged["Detectable"],
        "DiagnosticName":merged["DiagnosticName"]
    })
    return out.sort_values(["RefDes","ComponentType"], kind="stable").reset_index(drop=True)

# ============================ Metrics ============================
def compute_spfm(fmeda: pd.DataFrame, label_col: str) -> float | None:
    d = fmeda.copy()
    d["FIT"]   = pd.to_numeric(d["FIT"], errors="coerce")
    d["Share"] = pd.to_numeric(d["Share"], errors="coerce")
    d["DC"]    = pd.to_numeric(d["DC"], errors="coerce").fillna(0.0).clip(0,1)
    d["FIT_eff"] = d["FIT"] * d["Share"] * (1.0 - d["DC"])
    valid = d["FIT_eff"].notna()
    if not valid.any(): return None
    lam_total = d.loc[valid, "FIT_eff"].sum()
    lam_spf   = d.loc[valid & (d.get(label_col,"SAFE").str.upper()=="UNSAFE"), "FIT_eff"].sum()
    if lam_total <= 0: return None
    return 1.0 - (lam_spf / lam_total)

# ============================ UI ============================
st.title("FMEDA — KiCad .net (S-expression/XML) • Rules vs LLM")

safety_goal = st.text_input("Safety goal", "Prevent unintended output > 5 V at OUT")
threshold_v = extract_threshold_from_goal(safety_goal, default=5.0)

c0, c1 = st.columns([1,1])
with c0:
    net_file = st.file_uploader("Upload KiCad netlist (.net — S-expression or XML)", type=["net","xml"])
with c1:
    fail_file = st.file_uploader("Upload Failure Rates & Modes (CSV)", type=["csv"])

use_llm = st.toggle("Run LLM comparison (optional)", value=False, help="Needs OPENAI_API_KEY in Streamlit secrets.")

if net_file:
    try:
        nets, comp_pins_map, pin_name_map, ref_types = parse_kicad_net(net_file.read())
    except Exception as e:
        st.error(f"Failed to parse .net (S-expression/XML): {e}")
        st.stop()

    if not comp_pins_map:
        st.error("Parsed 0 components from the .net. Confirm this is a KiCad netlist (not SPICE) and try again.")
        st.stop()

    node2net = invert_node_to_net(nets)
    st.success(f"Parsed NET: {len(nets)} nets • {len(comp_pins_map)} components • {len(pin_name_map)} pin-name entries.")

    # Build components strictly from NET (no SCH)
    comp_df = build_component_table_from_net(comp_pins_map, ref_types)
    st.subheader("NET components (edit types if needed)")
    comp_df = st.data_editor(
        comp_df[["RefDes","ComponentType","ct_norm"]].rename(columns={"ct_norm":"_norm (read-only)"}),
        disabled=["_norm (read-only)"],
        height=320
    )
    comp_df["ct_norm"] = comp_df["ComponentType"].map(_norm)

    # Target net dropdown
    net_names = sorted(nets.keys(), key=lambda x: x.lower())
    default_idx = 0
    for i, n in enumerate(net_names):
        if "out" in n.lower():
            default_idx = i; break
    target_net = st.selectbox("Target net (from NET)", net_names, index=default_idx if net_names else 0)

    if fail_file:
        st.subheader("Failure DB preview & mapping")
        failure_df, raw_df_debug, col_share_name, col_fit_name = parse_failure_csv_with_mapping(fail_file)
        debug_numeric_preview(raw_df_debug, col_share_name, col_fit_name)
        st.divider()

        if st.button("Build FMEDA (NET-only)"):
            fmeda = expand_fmeda(comp_df, failure_df)

            # Fill missing Share equally per (RefDes, ComponentType)
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

            # Deterministic labels
            labels_rules, reasons_rules = [], []
            for _, row in fmeda.iterrows():
                if pd.isna(row["FIT"]) or pd.isna(row["Share"]):
                    labels_rules.append("UNASSESSED"); reasons_rules.append("Missing FIT/Share or unknown type")
                    continue
                lab, rsn = label_rules(row, nets, comp_pins_map, pin_name_map, node2net, target_net, threshold_v)
                labels_rules.append(lab); reasons_rules.append(rsn)
            fmeda["Label_rules"]  = labels_rules
            fmeda["Reason_rules"] = reasons_rules

            # LLM labels (optional)
            if use_llm:
                labels_llm, reasons_llm = [], []
                for _, row in fmeda.iterrows():
                    if pd.isna(row["FIT"]) or pd.isna(row["Share"]):
                        labels_llm.append("UNASSESSED"); reasons_llm.append("Missing FIT/Share or unknown type")
                        continue
                    lab, rsn = label_llm(row, safety_goal, target_net, nets, comp_pins_map, pin_name_map, node2net, threshold_v)
                    labels_llm.append(lab); reasons_llm.append(rsn)
                fmeda["Label_llm"]  = labels_llm
                fmeda["Reason_llm"] = reasons_llm
                fmeda["Label_diff"] = (fmeda["Label_rules"].astype(str).str.upper()
                                       != fmeda["Label_llm"].astype(str).str.upper())
            else:
                fmeda["Label_llm"] = ""
                fmeda["Reason_llm"] = ""
                fmeda["Label_diff"] = False

            # Effective FIT & SPFM
            fmeda["FIT_eff"] = (
                pd.to_numeric(fmeda["FIT"], errors="coerce").fillna(pd.NA)
                * pd.to_numeric(fmeda["Share"], errors="coerce").fillna(pd.NA)
                * (1.0 - pd.to_numeric(fmeda["DC"], errors="coerce").fillna(0.0))
            )

            spfm_rules = compute_spfm(fmeda, label_col="Label_rules")
            if spfm_rules is not None:
                st.metric("SPFM (Rules)", f"{spfm_rules*100:.2f}%")
            else:
                st.info("SPFM (Rules) not computable (no valid FIT×Share).")

            if use_llm:
                spfm_llm = compute_spfm(fmeda, label_col="Label_llm")
                if spfm_llm is not None:
                    st.metric("SPFM (LLM)", f"{spfm_llm*100:.2f}%")
                else:
                    st.info("SPFM (LLM) not computable (no valid FIT×Share).")

            st.subheader("FMEDA — NET-only • Rules vs LLM")
            st.dataframe(
                fmeda[[
                    "RefDes","ComponentType","FailureMode","Share","FIT","DC",
                    "Label_rules","Reason_rules","Label_llm","Reason_llm","Label_diff","FIT_eff"
                ]],
                height=560, use_container_width=True
            )

            st.download_button("Download FMEDA (CSV)",
                               fmeda.to_csv(index=False).encode("utf-8"),
                               "fmeda_results.csv","text/csv")

            if EXCEL_ENABLED:
                xbuf = io.BytesIO()
                with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
                    fmeda.to_excel(xw, index=False, sheet_name="FMEDA")
                    meta = {
                        "SafetyGoal": safety_goal,
                        "Threshold_V": threshold_v,
                        "TargetNet": target_net,
                        "LLM_Enabled": use_llm
                    }
                    pd.DataFrame([meta]).to_excel(xw, index=False, sheet_name="Summary")
                st.download_button("Download FMEDA (XLSX)",
                                   xbuf.getvalue(),
                                   "fmeda_results.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Excel export disabled (install `xlsxwriter`).")
    else:
        st.info("Upload the Failure Rates & Modes CSV to proceed.")
else:
    st.info("Upload a KiCad .net (S-expression/XML) and the Failure DB (CSV) to build FMEDA.")
