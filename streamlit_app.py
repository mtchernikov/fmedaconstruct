import streamlit as st
import pandas as pd
import re, io, csv, json
from collections import defaultdict
from time import sleep

# ============================ App config ============================
st.set_page_config(page_title="FMEDA (KiCad .net) — LLM only", layout="wide")
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


# ============================ Utils ============================
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
def iter_blocks_exact(text: str, tag: str):
    """Yield balanced s-expr blocks starting with '({tag} ' or '({tag}(' ."""
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
    # matches (key "X") or (key X)
    return rf'\({key}\s+("([^"]+)"|[^\s\)]+)\)'

def _pick(m, quoted_idx=2, any_idx=1):
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


# ============================ NET parser (S-exp + XML) ============================
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
    if base=="IC" and ("opamp" in v or "lm" in v):
        return "OpAmp"
    if "diode" in v or v.startswith("1n") or v.startswith("sb"):
        return "Diode"
    return base

def parse_kicad_net(net_bytes: bytes):
    text = net_bytes.decode("utf-8", errors="ignore").strip()

    # XML path
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

    # S-expression path
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
    return out, raw, c_share, c_fit


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


# ============================ LLM classification only ============================
def label_llm(row: pd.Series, safety_goal_text: str, target_net: str,
              nets, comp_pins_map, pin_name_map, node2net):
    """
    Returns (label, reason).
    Uses the *exact* safety_goal_text from the UI as the goal definition.
    """
    if OAI is None:
        return "NEEDS_REVIEW", "LLM not configured (set OPENAI_API_KEY in secrets)."

    ref = str(row["RefDes"])
    pins = sorted(list(comp_pins_map.get(ref, [])))
    pinctx = []
    for p in pins:
        net = node2net.get((ref, p))
        info = pin_name_map.get((ref, p), {})
        pinctx.append({"pin": p, "name": info.get("name",""), "net": net})

    # Small supply hints from net names (non-binding; goal text is authoritative)
    def guess_v(name: str):
        if not name: return None
        n = name.lower()
        if n in {"/b-","/gnd","gnd","ground","0v"}: return 0.0
        if "12v" in n or n in {"/b+"}: return 12.0
        m = re.search(r'(\d+(?:\.\d+)?)v', n)
        try: return float(m.group(1)) if m else None
        except: return None

    supply_candidates = []
    for n in {x["net"] for x in pinctx if x.get("net")}:
        v = guess_v(n)
        if v is not None:
            supply_candidates.append({"net": n, "voltage_hint": v})

    prompt = f"""
You are an FMEDA safety expert. Perform single-fault propagation for the EXACT goal given below.
Do not reinterpret or restate the goal: use it verbatim.

SAFETY_GOAL (verbatim): {safety_goal_text}
TARGET_NET: {target_net}

Component under analysis:
  RefDes: {ref}
  Type: {row.get('ComponentType')}
  Failure Mode: {row.get('FailureMode')}

Pins from KiCad netlist (net names and optional pin names):
{json.dumps(pinctx, ensure_ascii=False)}

Voltage hints inferred from net names (optional, may be incomplete):
{json.dumps(supply_candidates, ensure_ascii=False)}

Task:
Decide if THIS SINGLE FAILURE MODE can violate the given safety goal.
Be conservative when unsure.

Return STRICT JSON only:
{{
  "label":"SAFE|UNSAFE|NEEDS_REVIEW",
  "reason":"short, clear justification (<= 140 chars)"
}}
"""

    try:
        # Ask for JSON output; fall back robustly if API doesn't support response_format.
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
            lab, rsn = "NEEDS_REVIEW", f"Invalid label from LLM: {lab}"
        return lab, rsn
    except Exception as e:
        return "NEEDS_REVIEW", f"LLM error: {e}"


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
st.title("FMEDA — KiCad .net • LLM-only propagation")

safety_goal = st.text_input("Safety Goal (verbatim, used as-is by the LLM)",
                            "Prevent unintended >5 V at OUT")

c0, c1 = st.columns([1,1])
with c0:
    net_file = st.file_uploader("Upload KiCad netlist (.net — S-expression or XML)", type=["net","xml"])
with c1:
    fail_file = st.file_uploader("Upload Failure Rates & Modes (CSV)", type=["csv"])

if net_file:
    try:
        nets, comp_pins_map, pin_name_map, ref_types = parse_kicad_net(net_file.read())
    except Exception as e:
        st.error(f"Failed to parse .net (S-expression/XML): {e}")
        st.stop()

    if not comp_pins_map:
        st.error("Parsed 0 components from the .net. Confirm this is a KiCad netlist (not SPICE).")
        st.stop()

    node2net = invert_node_to_net(nets)
    st.success(f"Parsed NET: {len(nets)} nets • {len(comp_pins_map)} components • {len(pin_name_map)} pin-name entries.")

    comp_df = build_component_table_from_net(comp_pins_map, ref_types)
    st.subheader("NET components (edit types if needed)")
    comp_df = st.data_editor(
        comp_df[["RefDes","ComponentType","ct_norm"]].rename(columns={"ct_norm":"_norm (read-only)"}),
        disabled=["_norm (read-only)"],
        height=320
    )
    comp_df["ct_norm"] = comp_df["ComponentType"].map(_norm)

    # Target net selection
    net_names = sorted(nets.keys(), key=lambda x: x.lower())
    default_idx = 0
    for i, n in enumerate(net_names):
        if "out" in n.lower():
            default_idx = i; break
    target_net = st.selectbox("Target net (from NET)", net_names, index=default_idx if net_names else 0)

    if fail_file:
        st.subheader("Failure DB preview & mapping")
        failure_df, raw_df_debug, col_share_name, col_fit_name = parse_failure_csv_with_mapping(fail_file)

        # quick parse debug
        rs = raw_df_debug[col_share_name].astype(str)
        rf = raw_df_debug[col_fit_name].astype(str)
        preview = pd.DataFrame({
            "Share_raw": rs.head(8).tolist(),
            "FIT_raw":   rf.head(8).tolist(),
        })
        st.write("Parsing preview (first few rows):")
        st.dataframe(preview, height=180)

        st.divider()
        run = st.button("Build FMEDA with LLM")
        if run:
            if OAI is None:
                st.error("LLM not configured. Set OPENAI_API_KEY in Streamlit secrets.")
                st.stop()

            # Expand FMEDA
            fmeda = comp_df.merge(failure_df, how="left", on="ct_norm", suffixes=("_comp","_db"))
            fmeda = fmeda[["RefDes","ComponentType_comp","FailureMode","Share","FIT","DC","Detectable","DiagnosticName"]] \
                         .rename(columns={"ComponentType_comp":"ComponentType"})

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

            fmeda["DC"] = pd.to_numeric(fmeda["DC"], errors="coerce").fillna(0.0).clip(0,1)
            fmeda["FIT_eff"] = (
                pd.to_numeric(fmeda["FIT"], errors="coerce").fillna(0.0)
                * fmeda["Share"].astype(float)
                * (1.0 - fmeda["DC"].astype(float))
            )

            # LLM pass
            labels, reasons = [], []
            prog = st.progress(0.0, text="Classifying with LLM…")
            total = len(fmeda)
            for idx, (_, row) in enumerate(fmeda.iterrows(), start=1):
                lab, rsn = label_llm(row, safety_goal, target_net, nets, comp_pins_map, pin_name_map, node2net)
                labels.append(lab); reasons.append(rsn)
                prog.progress(idx/total, text=f"Classifying with LLM… {idx}/{total}")
                # tiny sleep to avoid rate spikes; adjust as needed
                sleep(0.02)
            fmeda["Label_llm"]  = labels
            fmeda["Reason_llm"] = reasons

            spfm_llm = compute_spfm(fmeda, label_col="Label_llm")
            if spfm_llm is not None:
                st.metric("SPFM (LLM)", f"{spfm_llm*100:.2f}%")
            else:
                st.info("SPFM (LLM) not computable (no valid FIT×Share).")

            st.subheader("FMEDA — LLM result")
            st.dataframe(
                fmeda[[
                    "RefDes","ComponentType","FailureMode","Share","FIT","DC",
                    "Label_llm","Reason_llm","FIT_eff"
                ]],
                height=560, use_container_width=True
            )

            # Downloads
            csv_bytes = fmeda.to_csv(index=False).encode("utf-8")
            st.download_button("Download FMEDA (CSV)", csv_bytes, "fmeda_llm_only.csv","text/csv")

            if EXCEL_ENABLED:
                xbuf = io.BytesIO()
                with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
                    fmeda.to_excel(xw, index=False, sheet_name="FMEDA")
                    meta = {
                        "SafetyGoal_verbatim": safety_goal,
                        "TargetNet": target_net,
                        "Model": OPENAI_MODEL,
                    }
                    pd.DataFrame([meta]).to_excel(xw, index=False, sheet_name="Summary")
                st.download_button("Download FMEDA (XLSX)",
                                   xbuf.getvalue(),
                                   "fmeda_llm_only.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Excel export disabled (install `xlsxwriter`).")
    else:
        st.info("Upload the Failure Rates & Modes CSV to proceed.")
else:
    st.info("Upload a KiCad .net (S-expression/XML) and the Failure DB (CSV) to build FMEDA.")
