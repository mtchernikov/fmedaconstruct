import streamlit as st
import pandas as pd
import openai
from io import StringIO

st.set_page_config(page_title="FMEDA Assistant", layout="wide")
st.title("🛠️ Safety Co-Pilot: FMEDA Generator")

# --- Inputs ---
st.header("📂 Input Files")

# 1. RAG1: Failure Rate & Mode Table
rag1_file = st.file_uploader("Upload Component Failure Rates, Modes & Distributions (RAG1 CSV)", type=["csv"])

# 2. RAG2: Optional FMEDA Examples
rag2_file = st.file_uploader("Upload Example FMEDA Tables (Optional, RAG2 CSV)", type=["csv"])

# 3. Netlist file
netlist_file = st.file_uploader("Upload KiCad Netlist File (.net)", type=["net"])

# 4. Safety Requirement
safety_goal = st.text_input("Enter Safety Goal / Requirement")

submit = st.button("🔍 Generate FMEDA Table")

# --- Helper Function: Parse Netlist ---
def parse_netlist(file):
    lines = file.read().decode("utf-8").splitlines()
    components = []
    capture = False
    comp = {}
    for line in lines:
        line = line.strip()
        if line == "(components":
            capture = True
        elif line == ")" and capture:
            break
        elif capture:
            if line.startswith("(comp"):
                comp = {}
            elif line.startswith("(ref"):
                tokens = line.split()
                if len(tokens) >= 2:
                    comp["Component"] = tokens[1].replace(")", "")
            elif line.startswith("(value"):
                tokens = line.split()
                if len(tokens) >= 2:
                    comp["Value"] = tokens[1].replace(")", "")
            elif line.startswith("(footprint"):
                tokens = line.split()
                if len(tokens) >= 2:
                    comp["Footprint"] = tokens[1].replace(")", "")
                else:
                    comp["Footprint"] = "UNKNOWN"
                components.append(comp)
    return pd.DataFrame(components)

# --- Helper Function: LLM call ---
def ask_llm_about_fault(component_name, failure_mode, safety_goal):
    prompt = f"""
You are a functional safety expert. A fault has occurred in component '{component_name}' with failure mode: '{failure_mode}'.
Safety goal: "{safety_goal}".

Would this fault violate the safety goal? Answer 'Yes' or 'No' and briefly explain.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            api_key=st.secrets["OPENAI_API_KEY"],
            messages=[
                {"role": "system", "content": "You are an expert in FMEDA and ISO 26262 safety analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"LLM error: {e}"

# --- Process ---
if submit and rag1_file and netlist_file and safety_goal:
    with st.spinner("Processing files and generating FMEDA table..."):
        # Load RAG1
        rag1_df = pd.read_csv(rag1_file)

        # Load RAG2 if provided (not used directly here)
        if rag2_file:
            rag2_df = pd.read_csv(rag2_file)

        # Parse netlist
        netlist_df = parse_netlist(netlist_file)

        if netlist_df.empty:
            st.error("❌ No components found in netlist. Please check the file format.")
            st.stop()

        # Try to find usable column to merge
        merge_key = None
        for candidate in ["Normalized_Component", "Component", "Value", "PartNumber"]:
            if candidate in rag1_df.columns:
                merge_key = candidate
                break

        if not merge_key:
            st.error("❌ Could not find a valid column in RAG1 to merge with netlist 'Value'")
            st.stop()

        # Merge
        merged = netlist_df.merge(rag1_df, how="left", left_on="Value", right_on=merge_key)

        # Call LLM for each component
        merged["Violates Safety Goal"] = merged.apply(
            lambda row: ask_llm_about_fault(row["Component"], row.get("Failure Mode", "unknown"), safety_goal)
            if pd.notna(row.get("Failure Mode")) else "N/A",
            axis=1
        )

        # Rename for clarity
        result_df = merged[[
            "Component", "Value",
            *(col for col in ["Category", "Subcategory", "Base FIT", "Failure Mode", "Probability (%)"] if col in merged.columns),
            "Violates Safety Goal"
        ]].rename(columns={
            "Base FIT": "Base Failure Rate (FIT)",
            "Probability (%)": "Failure Mode Distribution (%)"
        })

        st.success("FMEDA table generated.")
        st.dataframe(result_df, use_container_width=True)
else:
    st.info("Please upload required files and enter safety goal to begin.")
