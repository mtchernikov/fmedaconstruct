import streamlit as st
import pandas as pd
import re
import json
from openai import OpenAI

# Mapping from KiCad prefix to failure DB component types
COMPONENT_TYPE_MAP = {
    "R": "Resistor",
    "RES": "Resistor",
    "C": "Capacitor",
    "CAP": "Capacitor",
    "U": "OpAmp",
    "Q": "MOSFET",
    "M": "MOSFET",
    "D": "Diode",
    "L": "Inductor",
}

# Initialize OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def parse_kicad_sch_components(sch_text):
    """Extract components from KiCad 9 schematic text."""
    components = []
    matches = re.findall(r'\(symbol\s+("[^"]+")\s+(.*?)\)\s*\)', sch_text, re.DOTALL)

    for m in matches:
        body = m[1]
        ref_match = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', body)
        ref = ref_match.group(1) if ref_match else "?"

        val_match = re.search(r'\(property\s+"Value"\s+"([^"]+)"', body)
        value = val_match.group(1) if val_match else ""

        prefix = re.match(r"[A-Za-z]+", ref)
        if prefix:
            prefix = prefix.group(0).upper()
        comp_type = COMPONENT_TYPE_MAP.get(prefix, None)

        if comp_type is None:
            val_up = value.upper()
            if "RES" in val_up or re.match(r"\d+[kKmM]?", val_up):
                comp_type = "Resistor"
            elif "CAP" in val_up or re.match(r"\d+uF|\d+nF|\d+pF", val_up, re.IGNORECASE):
                comp_type = "Capacitor"
            else:
                comp_type = "unknown component"

        components.append({"RefDes": ref, "ComponentType": comp_type})

    return pd.DataFrame(components)

def merge_with_failure_modes(components_df, failure_rates_df):
    """Merge extracted components with Failure Rates/Modes table."""
    merged_rows = []

    for _, comp in components_df.iterrows():
        comp_type = comp["ComponentType"]
        match_df = failure_rates_df[
            failure_rates_df["ComponentType"].str.strip().str.lower() == comp_type.lower()
        ]

        if match_df.empty:
            merged_rows.append({
                "RefDes": comp["RefDes"],
                "ComponentType": comp_type,
                "FailureMode": "unknown",
                "Share": None,
                "FIT": None,
                "DC": None,
                "Detectable": None,
                "DiagnosticName": None
            })
        else:
            for _, fm in match_df.iterrows():
                merged_rows.append({
                    "RefDes": comp["RefDes"],
                    "ComponentType": comp_type,
                    "FailureMode": fm.get("FailureMode"),
                    "Share": fm.get("Share"),
                    "FIT": fm.get("FIT"),
                    "DC": fm.get("DC"),
                    "Detectable": fm.get("Detectable"),
                    "DiagnosticName": fm.get("DiagnosticName")
                })

    return pd.DataFrame(merged_rows)

def llm_classify_row(row, safety_goal):
    """Classify Safe/Unsafe using LLM."""
    prompt = f"""
    You are an expert in FMEDA for electronics safety.
    Safety Goal: {safety_goal}

    Component:
      RefDes: {row['RefDes']}
      Type: {row['ComponentType']}
      Failure Mode: {row['FailureMode']}
      FIT: {row['FIT']}
      Diagnostic Coverage: {row['DC']}

    Decide if this failure mode is SAFE or UNSAFE with respect to the Safety Goal.
    Only respond with JSON:
    {{
        "label": "SAFE" | "UNSAFE" | "NEEDS_REVIEW",
        "reason": "short explanation"
    }}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("label", "NEEDS_REVIEW"), data.get("reason", "")
    except Exception as e:
        return "NEEDS_REVIEW", f"LLM error: {e}"

# Streamlit UI
st.title("FMEDA Generator with LLM Safety Classification")

safety_goal = st.text_input("Enter Safety Goal", "Prevent unintended output > 5V")

sch_file = st.file_uploader("Upload KiCad .kicad_sch", type=["kicad_sch"])
failure_csv = st.file_uploader("Upload Failure Rates & Modes CSV", type=["csv"])

if sch_file and failure_csv:
    sch_text = sch_file.read().decode("utf-8")
    components_df = parse_kicad_sch_components(sch_text)
    failure_rates_df = pd.read_csv(failure_csv)

    merged_fmeda_df = merge_with_failure_modes(components_df, failure_rates_df)

    # Classification step
    labels, reasons = [], []
    for _, row in merged_fmeda_df.iterrows():
        label, reason = llm_classify_row(row, safety_goal)
        labels.append(label)
        reasons.append(reason)

    merged_fmeda_df["Label"] = labels
    merged_fmeda_df["Reason"] = reasons

    st.subheader("FMEDA with Classification")
    st.dataframe(merged_fmeda_df)

    # Export
    csv_bytes = merged_fmeda_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download FMEDA CSV", csv_bytes, "fmeda_with_classification.csv", "text/csv")
