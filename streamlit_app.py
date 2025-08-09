import re
import pandas as pd

# ---- helpers for type detection (reuse your existing ones if you have them) ----
REF_PREFIX = {
    "R":"Resistor","C":"Capacitor","L":"Inductor","D":"Diode","Z":"Zener",
    "Q":"MOSFET","T":"BJT","U":"IC_Analog","A":"OpAmp","K":"Relay","F":"Fuse",
    "J":"Connector","X":"Connector"
}
def detect_type_from_ref(ref: str):
    if not ref: return None
    m = re.match(r"^([A-Za-z]+)", ref)
    if not m: return None
    pref = m.group(1).upper()
    for p in sorted(REF_PREFIX, key=lambda x: -len(x)):
        if pref.startswith(p):
            return REF_PREFIX[p]
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

# ---- NEW robust S-expression parser for symbols ----
def _iter_sexpr_blocks(text: str, tag: str):
    """Yield full balanced blocks starting with '(tag' (e.g. 'symbol')."""
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
        # continue search after this block
        start = text.find(needle, i)

def parse_kicad_sch_components(sch_text: str) -> pd.DataFrame:
    """
    Robustly extract components from KiCad 9 .kicad_sch by scanning balanced
    '(symbol ...)' blocks and reading properties inside.
    Returns DataFrame with columns: RefDes, ComponentType, ComponentTypeNorm (optional later).
    """
    rows = []
    for block in _iter_sexpr_blocks(sch_text, "symbol"):
        # lib_id
        m_lib = re.search(r'\(lib_id\s+"([^"]+)"\)', block)
        lib_id = m_lib.group(1) if m_lib else ""

        # properties: Reference / Value
        # allow extra attributes inside (property "Reference" "R1" (at ...) (effects ...))
        m_ref = re.search(r'\(property\s+"Reference"\s+"([^"]*)"', block)
        m_val = re.search(r'\(property\s+"Value"\s+"([^"]*)"', block)
        ref = (m_ref.group(1) if m_ref else "").strip()
        val = (m_val.group(1) if m_val else "").strip()

        # determine type: ref prefix first, then lib_id, then value
        t = detect_type_from_ref(ref)
        t2 = detect_type_from_lib(lib_id) or detect_type_from_value(val)
        if t == "IC_Analog" and t2:
            t = t2
        elif t2 and t != t2:
            if t in ("IC_Analog", "Other") or (t == "Capacitor" and t2 == "Capacitor_Polarized"):
                t = t2
        ctype = t or "Other"

        # handle missing refs (shouldn’t happen normally, but be safe)
        ref = ref if ref else "?"

        rows.append({"RefDes": ref, "ComponentType": ctype, "Value": val, "LibId": lib_id})

    if not rows:
        # Empty schematic or pattern didn’t match
        return pd.DataFrame(columns=["RefDes", "ComponentType"])

    df = pd.DataFrame(rows)[["RefDes", "ComponentType"]]
    # map unknowns explicitly as you requested
    df.loc[df["RefDes"] == "?", "ComponentType"] = "unknown component"
    df.loc[df["ComponentType"].isna(), "ComponentType"] = "unknown component"
    return df
