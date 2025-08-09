import streamlit as st
import pandas as pd
import io, json, uuid, re, xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

st.set_page_config(page_title="Safety Co-Pilot – FMEDA + Proposals", layout="wide")
def new_uuid(): return str(uuid.uuid4())

# ------- Data models -------
@dataclass
class Component:
    ref: str; ctype: str; value: str; x: float; y: float; rot: int; lib_id: str
@dataclass
class GlobalLabel:
    name: str; x: float; y: float

# ------- KiCad .kicad_sch (instances + labels, not nets) -------
def parse_kicad_sch(text: str) -> Tuple[List[Component], List[GlobalLabel]]:
    comps, labels = [], []
    sym_re = re.compile(r'\(symbol\s+\(lib_id\s+"([^"]+)"\)[\s\S]*?\(at\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\)[\s\S]*?\(property\s+"Reference"\s+"([^"]+)"[\s\S]*?\(property\s+"Value"\s+"([^"]*)"', re.M)
    for m in sym_re.finditer(text):
        lib_id,xs,ys,rot,ref,val = m.groups()
        comps.append(Component(ref.strip(),"Unknown",(val or "").strip(),float(xs),float(ys),int(float(rot)),lib_id.strip()))
    gl_re = re.compile(r'\(global_label\s+"([^"]+)"[\s\S]*?\(at\s+([-\d.]+)\s+([-\d.]+)', re.M)
    for m in gl_re.finditer(text):
        name,xs,ys = m.groups(); labels.append(GlobalLabel(name.strip(),float(xs),float(ys)))
    return comps, labels

# ------- Optional KiCad XML netlist (exact connectivity) -------
# Export in KiCad: File → Export → Netlist… → KiCad XML
def parse_xml_netlist(xml_bytes: bytes):
    # returns: dict net->set(ref.pin), dict ref->{pin->net}
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return {}, {}
    ns = {}  # no namespace
    net2pins, partpins = {}, {}
    for net in root.findall(".//net", ns):
        netname = net.get("name","")
        items = set()
        for node in net.findall("./node", ns):
            ref = node.get("ref"); pin = node.get("pin")
            items.add(f"{ref}.{pin}")
            partpins.setdefault(ref, {})[pin] = netname
        if netname: net2pins.setdefault(netname,set()).update(items)
    return net2pins, partpins

# ------- Deterministic type detection (no LLM) -------
TYPE_SYNONYMS = {
    "Resistor":["resistor","res","r","shunt"],
    "Capacitor":["capacitor","cap","c"],
    "Capacitor_Polarized":["electrolytic","cp","cap_polarized","elcap"],
    "Inductor":["inductor","l","coil","choke"],
    "Diode":["diode","d"], "Zener":["zener","zd"],
    "BJT":["bjt","transistor","q","npn","pnp"],
    "MOSFET":["mosfet","nfet","pfet","irf","irfp"],
    "OpAmp":["opamp","amplifier_operational","op-amp","amp","ne5532","lm358"],
    "Comparator":["comparator","lm393","cmp"],
    "Connector":["connector","jack","terminal","hdr","audiojack","conn","j"],
    "Relay":["relay","k"], "Fuse":["fuse","f"], "IC_Analog":["ic_analog"], "IC_Digital":["ic_digital","logic","gate","74hc","cd40"],
    "Other":["other","unknown"]
}
REF_PREFIX = {"R":"Resistor","C":"Capacitor","L":"Inductor","D":"Diode","Z":"Zener","Q":"MOSFET","T":"BJT","U":"IC_Analog","A":"OpAmp","K":"Relay","F":"Fuse","J":"Connector","X":"Connector"}
def detect_type_from_ref(ref:str)->Optional[str]:
    m=re.match(r"^([A-Za-z]+)",ref or ""); 
    if not m: return None
    pref=m.group(1).upper()
    for p in sorted(REF_PREFIX,key=lambda x:-len(x)):
        if pref.startswith(p): return REF_PREFIX[p]
    return None
def detect_type_from_lib(lib_id:str)->Optional[str]:
    s=(lib_id or "").lower()
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
def detect_type_from_value(val:str)->Optional[str]:
    t=(val or "").lower()
    if re.search(r'(^|\s)\d+(\.\d+)?(r|k|m)?(\s*ohm|$)',t): return "Resistor"
    if re.search(r'(^|\s)\d+(\.\d+)?(n|u|µ|p|f)(\s*|$)',t): return "Capacitor"
    if "ne5532" in t or "lm358" in t: return "OpAmp"
    if "irf" in t or "irfp" in t: return "MOSFET"
    return None
def detect_component_type(ref,lib_id,value)->str:
    t=detect_type_from_ref(ref)
    t2=detect_type_from_lib(lib_id) or detect_type_from_value(value)
    if t=="IC_Analog" and t2: t=t2
    elif t2 and t!=t2:
        if t in ("IC_Analog","Other"): t=t2
        if t=="Capacitor" and t2=="Capacitor_Polarized": t=t2
    return t or "Other"

# ------- Failure DB (single CSV) -------
def parse_failure_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    # Normalize header strings once
    norm = {c: re.sub(r'[^a-z0-9]', '', c.strip().lower()) for c in raw.columns}

    # Helper: find column by a set of acceptable normalized names
    def find_col(candidates):
        for orig, n in norm.items():
            if n in candidates:
                return orig
        return None

    # Try broad synonym sets
    col_type = find_col({"componenttype","type","class","category","parttype","compclass"})
    col_mode = find_col({"failuremode","mode","fm"})
    col_share = find_col({"share","modeshare","distribution","percent","modesharepercent","modesharepct"})
    col_fit   = find_col({"fit","lambda","rate","fitrate","lambdaft"})
    col_dc    = find_col({"dc","diagnosticcoverage","coverage","diagcoverage"})
    col_det   = find_col({"detectable","detected","isdetected"})
    col_dname = find_col({"diagnosticname","diag","diagnostic"})

    # If any critical column missing, ask user to map interactively
    missing = []
    if col_type is None: missing.append("Component Type")
    if col_mode is None: missing.append("Failure Mode")
    if col_share is None: missing.append("Mode Share")
    if col_fit is None: missing.append("FIT")

    if missing:
        st.warning(f"Please map missing columns: {', '.join(missing)}")
        cols = list(raw.columns)
        col_type = st.selectbox("Column for Component Type", cols, index=0 if col_type is None else cols.index(col_type))
        col_mode = st.selectbox("Column for Failure Mode", cols, index=0 if col_mode is None else cols.index(col_mode))
        col_share = st.selectbox("Column for Mode Share", cols, index=0 if col_share is None else cols.index(col_share))
        col_fit   = st.selectbox("Column for FIT", cols, index=0 if col_fit is None else cols.index(col_fit))
        col_dc    = st.selectbox("Column for Diagnostic Coverage (optional)", ["<none>"]+cols, index=0)
        col_det   = st.selectbox("Column for Detectable (optional)", ["<none>"]+cols, index=0)
        col_dname = st.text_input("Column for Diagnostic Name (optional)", value=col_dname or "")

        col_dc = None if col_dc == "<none>" else col_dc
        col_det = None if col_det == "<none>" else col_det
        col_dname = None if not col_dname else col_dname

    # Build output with defaults
    out = pd.DataFrame()
    out["ComponentType"] = raw[col_type].astype(str).str.strip()
    out["FailureMode"] = raw[col_mode].astype(str).str.strip()
    # Share: accept % or fraction
    share_series = pd.to_numeric(raw[col_share], errors="coerce").fillna(0.0)
    if share_series.max() > 1.5:  # looks like percent
        share_series = share_series / 100.0
    out["Share"] = share_series.clip(lower=0.0)
    # FIT
    out["FIT"] = pd.to_numeric(raw[col_fit], errors="coerce").fillna(0.0)

    # Optionals
    if col_dc is not None:
        out["DC"] = pd.to_numeric(raw[col_dc], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        out["DC"] = 0.0
    if col_det is not None:
        # Treat non-empty/nonzero as True
        v = raw[col_det]
        out["Detectable"] = v.astype(str).str.strip().str.lower().isin(["1","true","yes","y","t"])
    else:
        out["Detectable"] = False
    if col_dname is not None and col_dname in raw.columns:
        out["DiagnosticName"] = raw[col_dname].astype(str).fillna("")
    else:
        out["DiagnosticName"] = ""

    # Canonicalize ComponentType to match our internal types
    def canonize(s: str) -> str:
        s = (s or "").strip().lower()
        for canon, syns in TYPE_SYNONYMS.items():
            if s == canon.lower() or s in syns:
                return canon
        return "Other"

    out["ComponentTypeNorm"] = out["ComponentType"].apply(canonize)

    # Final column order
    return out[["ComponentType","ComponentTypeNorm","FailureMode","Share","FIT","Detectable","DC","DiagnosticName"]]


# ------- Safety goal -------
def parse_safety_goal(text:str)->Dict[str,Any]:
    t=text.strip().lower(); goal={"raw":text,"metric":"max_voltage","limit_V":None,"type":"peak"}
    m=re.search(r'(\d+(\.\d+)?)\s*v', t); 
    if m: goal["limit_V"]=float(m.group(1))
    if "dc" in t: goal["metric"]="dc_offset"; goal["type"]="dc"
    if "rail" in t: goal["metric"]="rail_event"; goal["type"]="rail"
    if "vpp" in t or "peak" in t: goal["type"]="peak"
    return goal

# ------- Roles + applicable modes -------
def infer_role(c:Component)->str:
    lid=c.lib_id.lower(); val=(c.value or "").lower(); ctype=detect_component_type(c.ref,c.lib_id,c.value)
    if ctype in ("Capacitor","Capacitor_Polarized"):
        if "cp" in lid or "electroly" in lid or "u" in val: return "coupling_cap"
        return "cap"
    if ctype=="Resistor":
        try:
            if "k" in val: ohm=float(val.replace("k",""))*1e3
            elif "m" in val: ohm=float(val.replace("m",""))/1e3
            else: ohm=float(re.sub("[^0-9.]", "", val) or 0)
        except: ohm=None
        if ohm is not None and ohm<=150: return "series_out_res"
        return "res"
    if ctype=="OpAmp": return "opamp"
    return "other"
def applicable_modes(ctype:str)->List[str]:
    if ctype=="Resistor": return ["open","short"]
    if ctype in ("Capacitor","Capacitor_Polarized"): return ["open","short"]
    if ctype=="OpAmp": return ["output_stuck_high","output_stuck_low","open_feedback"]
    if ctype in ("MOSFET","BJT"): return ["short_drain_source","gate_open","stuck_on","stuck_off"]
    if ctype=="Diode": return ["open","short"]
    return ["open","short"]

# ------- SAFE/UNSAFE rule engine (conservative) -------
def rule_label(ctx:Dict[str,Any], goal:Dict[str,Any], outputs:List[str])->Tuple[str,str]:
    comp,fm,role=ctx["component"],ctx["failure_mode"],ctx["role"]
    if comp["ctype"]=="OpAmp" and fm in ("output_stuck_high","output_stuck_low"):
        return "UNSAFE","OpAmp output can hit rail without barrier."
    if role=="coupling_cap" and fm=="short":
        return "UNSAFE","Coupling capacitor short creates DC path to output."
    if role=="series_out_res" and fm=="open":
        return "SAFE","Series output resistor open disconnects output."
    if role=="coupling_cap" and fm=="open":
        return "SAFE","Coupling cap open blocks the signal."
    if comp["ctype"]=="Resistor" and ("feedback" in ctx.get("hints",[])) and fm=="short":
        return "UNSAFE","Feedback short can force amplifier saturation."
    return "NEEDS_REVIEW","Topology context insufficient; manual check advised."
def classify_fault(safety_goal_text:str, outputs:List[str], ctx:Dict[str,Any])->Dict[str,Any]:
    label,reason=rule_label(ctx, parse_safety_goal(safety_goal_text), outputs)
    return {"label":label,"reason":reason,"affected_outputs":outputs}

# ------- FMEDA + SPFM -------
def compute_fmeda(components:List[Component], failure_df:pd.DataFrame, safety_goal_text:str, outputs:List[str]):
    rows=[]; by_type={k:list(v.index) for k,v in failure_df.groupby("ComponentTypeNorm")}
    for c in components:
        det=detect_component_type(c.ref,c.lib_id,c.value); dbt = det if det in by_type else (list(by_type.keys()) or ["Other"])[0]
        role=infer_role(c)
        for idx in by_type.get(dbt,[]):
            r=failure_df.loc[idx]
            fm=str(r["FailureMode"])
            if fm not in applicable_modes(det): continue
            fit_mode=float(r["FIT"])*float(r["Share"])
            ctx={"component":{"ref":c.ref,"ctype":det,"value":c.value,"lib_id":c.lib_id},"failure_mode":fm,"role":role,"hints":[]}
            verdict=classify_fault(safety_goal_text, outputs, ctx)
            rows.append({"ComponentRef":c.ref,"DetectedType":det,"DBType":dbt,"Value":c.value,
                         "FailureMode":fm,"Share":float(r["Share"]),"FIT_Base":float(r["FIT"]),
                         "FIT_Mode":fit_mode,"DiagnosticName":r.get("DiagnosticName",""),
                         "Detectable":bool(r.get("Detectable",False)),"DC":float(r.get("DC",0.0)),
                         "LLM_Label":verdict["label"],"Reason":verdict["reason"],"AffectedOutputs":",".join(verdict["affected_outputs"])})
    out=pd.DataFrame(rows)
    if out.empty: return out,1.0,None
    lam_total=out["FIT_Mode"].sum(); lam_spf=out.loc[out["LLM_Label"]=="UNSAFE","FIT_Mode"].sum()
    spfm=1.0-(lam_spf/lam_total if lam_total>0 else 0.0)
    return out,spfm,None

# ------- Export: embedded .kicad_sch with mitigation region -------
def lib_symbol(name, ref_char, value, body):
    return f'''        (symbol "{name}" (pin_numbers hide) (pin_names hide) (in_bom yes) (on_board yes)
            (property "Reference" "{ref_char}" (at 0 2.5 0) (effects (font (size 1.27 1.27))))
            (property "Value" "{value}" (at 0 -2.5 0) (effects (font (size 1.27 1.27))))
            (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) (hide yes)))
            (property "Datasheet" "" (at 0 0 0) (effects (font (size 1.27 1.27)) (hide yes)))
            (symbol "{name}_1_1"
{body}            )
        )
'''
def pin(ptype, atx, aty, ang, length, name, number):
    return f'                (pin {ptype} line (at {atx} {aty} {ang}) (length {length}) (name "{name}" (effects (font (size 1 1)))) (number "{number}" (effects (font (size 1 1)))))\n'
def poly(points,w=0.25):
    pts=" ".join(f'(xy {x} {y})' for x,y in points)
    return f'                (polyline (pts {pts}) (stroke (width {w}) (type default)))\n'
def kicad_text(x,y,txt):
    return f'''    (text "{txt}" (at {x} {y} 0) (effects (font (size 1.5 1.5)) (justify left)) (uuid "{new_uuid()}"))\n'''
def export_embedded_with_mitigations(components, labels, proposals, outputs, title="Mitigation Proposal"):
    lib = ""
    lib += lib_symbol("R_Small","R","R_Small", '                (rectangle (start -2.5 1) (end 2.5 -1) (stroke (width 0.15) (type default)) (fill (type none)))\n'+ pin("passive",-5,0,0,2.5,"1","1")+pin("passive",5,0,180,2.5,"2","2"))
    lib += lib_symbol("C_Small","C","C_Small", poly([(-2,0),(2,0)],0.3)+poly([(-2,-1.2),(2,-1.2)],0.3)+ pin("passive",0,4,270,2.5,"1","1")+pin("passive",0,-4,90,2.5,"2","2"))
    lib += lib_symbol("CP_Small","C","CP_Small", poly([(-2,0),(2,0)],0.3)+poly([(-2,-1.2),(2,-1.2)],0.3)+ pin("passive",0,4,270,2.5,"+","1")+pin("passive",0,-4,90,2.5,"-","2"))
    lib += lib_symbol("OpAmp","U","OpAmp", poly([(-6,-5),(-6,5)],0.15)+poly([(-6,5),(6,0)],0.15)+poly([(6,0),(-6,-5)],0.15)+ pin("input",-8,2,0,2,"+","3")+pin("input",-8,-2,0,2,"-","2")+pin("output",8,0,180,2,"OUT","1"))
    buf = io.StringIO()
    buf.write(f'''(kicad_sch
    (version 20250114)
    (generator "eeschema") (generator_version "9.0")
    (uuid "{new_uuid()}") (paper "A4")
    (title_block (title "{title}") (rev "P") (company "Safety Co-Pilot"))
    (lib_symbols
{lib}    )
''')
    for gl in labels:
        buf.write(f'''    (global_label "{gl.name}" (shape input) (at {gl.x} {gl.y} 0) (fields_autoplaced yes) (effects (font (size 1.27 1.27)) (justify left)) (uuid "{new_uuid()}"))\n''')
    for c in components:
        # Re-emit instances (no destructive edits)
        dt=detect_component_type(c.ref,c.lib_id,c.value)
        libname="R_Small" if dt=="Resistor" else ("CP_Small" if dt in ("Capacitor_Polarized",) else ("C_Small" if dt=="Capacitor" else ("OpAmp" if dt=="OpAmp" else "R_Small")))
        buf.write(f'''    (symbol (lib_id "{libname}") (at {c.x} {c.y} {c.rot}) (unit 1) (in_bom yes) (on_board yes) (uuid "{new_uuid()}"))
''')
    y0=18.0; buf.write(kicad_text(8,y0,"Mitigation region (accepted): connects via Global Labels to your real nets"))
    y=y0+6
    for p in proposals:
        buf.write(kicad_text(10,y,f"- {p['action']} @ {p.get('target','')} ({p.get('details','')})")); y+=5
    # For each safety output, drop a small block with labels to connect
    x0=160.0; row=0
    for outnet in outputs:
        buf.write(f'''    (global_label "{outnet}" (shape input) (at {x0} {100+row*16} 0) (fields_autoplaced yes) (effects (font (size 1.27 1.27)) (justify left)) (uuid "{new_uuid()}"))\n''')
        # placeholder series R + CP chain as a visual anchor
        buf.write(f'''    (symbol (lib_id "R_Small") (at {x0+14} {100+row*16} 0) (unit 1) (in_bom yes) (on_board yes) (uuid "{new_uuid()}"))
    (symbol (lib_id "CP_Small") (at {x0+26} {100+row*16} 0) (unit 1) (in_bom yes) (on_board yes) (uuid "{new_uuid()}"))
''')
        row+=1
    buf.write(")\n"); return buf.getvalue().encode("utf-8")

# ------- UI -------
st.title("Safety Co-Pilot – FMEDA from KiCad (Propose → Accept)")

c1,c2 = st.columns([1,1])
with c1:
    sch_file = st.file_uploader("Upload KiCad schematic (.kicad_sch)", type=["kicad_sch"])
    xml_file = st.file_uploader("Optional: KiCad XML netlist (for exact connectivity)", type=["xml"])
with c2:
    failure_csv = st.file_uploader("Upload Failure Rates + Failure Modes (CSV)", type=["csv"])
    safety_goal_text = st.text_input("Safety Goal", value="Prevent unintended output bigger than 5 V")
    safety_outputs_raw = st.text_input("Safety outputs (comma-separated)", value="LEFT_OUT, RIGHT_OUT")
    safety_outputs = [s.strip() for s in safety_outputs_raw.split(",") if s.strip()]

run = st.button("Run FMEDA")
if run:
    if not sch_file or not failure_csv:
        st.error("Please upload the schematic and the failure table."); st.stop()
    sch_text = sch_file.read().decode("utf-8", errors="ignore")
    comps, labels = parse_kicad_sch(sch_text)
    comps = [Component(c.ref, detect_component_type(c.ref,c.lib_id,c.value), c.value, c.x,c.y,c.rot,c.lib_id) for c in comps]
    st.success(f"Parsed {len(comps)} components, {len(labels)} global labels.")
    st.dataframe(pd.DataFrame([c.__dict__ for c in comps]))

    # Optional exact graph
    net2pins, partpins = ({}, {})
    if xml_file:
        net2pins, partpins = parse_xml_netlist(xml_file.read())
        st.caption(f"Netlist loaded: {len(net2pins)} nets.")

    fail_df = parse_failure_csv(failure_csv)
    st.write("Failure DB (normalized):"); st.dataframe(fail_df.head(20))

    results_df, spfm, lfm = compute_fmeda(comps, fail_df, safety_goal_text, safety_outputs)
    st.subheader("FMEDA results"); st.dataframe(results_df)
    st.metric("SPFM (approx.)", f"{spfm*100:.2f}%"); st.caption("LFM omitted in this step.")

    st.session_state["_results"]=results_df
    st.session_state["_comps"]=comps
    st.session_state["_labels"]=labels
    st.session_state["_outputs"]=safety_outputs

if st.button("Propose mitigations"):
    if "_results" not in st.session_state: st.error("Run FMEDA first."); st.stop()
    df = st.session_state["_results"]; outs = st.session_state["_outputs"]
    props=[]
    if (df["LLM_Label"]=="UNSAFE").any():
        props.append({"action":"Add DC window + mute relay","target":",".join(outs),"details":"Detect DC offset & disconnect <100 ms"})
    if not any(infer_role(c)=="series_out_res" for c in st.session_state["_comps"]):
        props.append({"action":"Add series resistor","target":",".join(outs),"details":"~22–100 Ω to limit surge"})
    props.append({"action":"Add output pulldown","target":",".join(outs),"details":"~47–220 kΩ to reference"})
    st.session_state["_props"]=props
    st.subheader("Proposed changes (review)"); st.dataframe(pd.DataFrame(props))

if st.button("Accept → export .kicad_sch (embedded mitigation region)"):
    if "_props" not in st.session_state: st.error("Nothing to accept. Click 'Propose mitigations' first."); st.stop()
    blob = export_embedded_with_mitigations(st.session_state["_comps"], st.session_state["_labels"],
                                            st.session_state["_props"], st.session_state["_outputs"])
    st.download_button("Download proposed_update.kicad_sch", data=blob, file_name="proposed_update.kicad_sch", mime="text/plain")

# Always: FMEDA exports
if "_results" in st.session_state:
    df = st.session_state["_results"]
    st.download_button("Download FMEDA (CSV)", df.to_csv(index=False).encode("utf-8"), "fmeda_results.csv","text/csv")
    xbuf=io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="FMEDA")
        pd.DataFrame([{"SPFM": (df["FIT_Mode"].sum()-df.loc[df["LLM_Label"]=="UNSAFE","FIT_Mode"].sum())/max(df["FIT_Mode"].sum(),1e-12)}]).to_excel(xw, index=False, sheet_name="Summary")
    st.download_button("Download FMEDA (XLSX)", xbuf.getvalue(), "fmeda_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download FMEDA (JSON)", df.to_json(orient="records", indent=2).encode("utf-8"), "fmeda_results.json","application/json")

