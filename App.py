"""
Streamlit front-end for the Ice Cream S&OP Agent.

Architecture:
  1. User fills in a form (demand, inventory, manpower, incoming materials).
  2. On submit, we call the DETERMINISTIC engine (sop_engine.py) to get the
     actual MPS and MRP numbers. This math is never touched by the AI.
  3. We then send that structured result to Claude (via the Anthropic API)
     and ask it to explain the plan in plain language, as "the Agent" would
     to a small business owner -- highlighting binding constraints and what
     changed vs. a straightforward "just meet demand" plan.

Run locally with:
    streamlit run app.py

Requires an ANTHROPIC_API_KEY, set as an environment variable or entered
in the sidebar at runtime (never hard-code your key in this file).
"""

import os
import json
import streamlit as st

from sop_engine import (
    PeriodInputs, run_period_plan, SKUS, ALL_MATERIALS,
    DEFAULT_SKU_PRIORITY,
)

st.set_page_config(page_title="Ice Cream S&OP Agent", layout="wide")

st.title("🍦 Ice Cream S&OP Agent")
st.caption(
    "Deterministic MPS/MRP engine + AI narration layer. "
    "All production/material numbers are computed by fixed formulas -- "
    "the AI only explains the result, it never changes the math."
)

# ---------------------------------------------------------------------------
# Sidebar: API key + constraint inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Agent Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Not stored anywhere -- used only for this session's narration call.",
    )
    st.divider()
    st.header("Capacity & Manpower")
    shifts_per_day = st.slider("Shifts per day", 1, 3, 2)
    workers_required = st.number_input("Workers required per shift (baseline)", value=14, min_value=1)
    workers_available = st.number_input(
        "Workers actually available this period", value=14, min_value=0,
        help="Set below the required number to simulate a staff shortage."
    )

# ---------------------------------------------------------------------------
# Main form: demand + inventory + incoming materials
# ---------------------------------------------------------------------------
st.subheader("1. Demand for this period")
demand_cols = st.columns(len(SKUS))
demand_units = {}
for col, sku in zip(demand_cols, SKUS):
    with col:
        demand_units[sku] = st.number_input(f"{sku} demand (units)", value=50000, min_value=0, step=1000, key=f"d_{sku}")

st.subheader("2. Opening finished-goods inventory")
inv_cols = st.columns(len(SKUS))
opening_fg = {}
for col, sku in zip(inv_cols, SKUS):
    with col:
        opening_fg[sku] = st.number_input(f"{sku} stock on hand", value=2000, min_value=0, step=100, key=f"i_{sku}")

st.subheader("3. Opening raw material & packaging inventory")
mat_cols = st.columns(3)
opening_materials = {}
for i, material in enumerate(ALL_MATERIALS):
    with mat_cols[i % 3]:
        opening_materials[material] = st.number_input(
            f"{material}", value=5000, min_value=0, step=100, key=f"m_{material}"
        )

st.subheader("4. Incoming materials already in transit (optional)")
incoming_cols = st.columns(3)
material_incoming = {}
for i, material in enumerate(ALL_MATERIALS):
    with incoming_cols[i % 3]:
        material_incoming[material] = st.number_input(
            f"{material} incoming", value=0, min_value=0, step=100, key=f"inc_{material}"
        )

period_label = st.text_input("Period label", value="Month 1")

run_clicked = st.button("Generate S&OP Plan", type="primary")

# ---------------------------------------------------------------------------
# Run the deterministic engine, then ask Claude to narrate it
# ---------------------------------------------------------------------------
if run_clicked:
    inputs = PeriodInputs(
        period_label=period_label,
        demand_units=demand_units,
        opening_fg_inventory=opening_fg,
        opening_material_inventory=opening_materials,
        material_incoming=material_incoming,
        shifts_per_day=shifts_per_day,
        workers_available=workers_available,
        workers_required_per_shift=workers_required,
        sku_priority=dict(DEFAULT_SKU_PRIORITY),
    )

    with st.spinner("Running capacity check and BOM explosion..."):
        result = run_period_plan(inputs)

    mps, mrp = result["mps"], result["mrp"]

    st.subheader("📋 Master Production Schedule (MPS)")
    st.json(mps, expanded=False)
    st.table({"SKU": list(mps["mps_units"].keys()), "Planned Units": list(mps["mps_units"].values())})

    if mps["capacity_binding"] or mps["packing_binding"]:
        st.warning(
            f"Capacity constraint hit this period (bottleneck stage: {mps['bottleneck_stage']}). "
            "Production has been rationed by SKU priority -- see unmet demand below."
        )
        st.table({"SKU": list(mps["unmet_demand_units"].keys()),
                   "Unmet Demand": list(mps["unmet_demand_units"].values())})
    else:
        st.success("All demand met within available capacity and manpower this period.")

    st.subheader("📦 Material Requirement Plan (MRP)")
    st.table({"Material": list(mrp["net_requirement"].keys()),
               "Net Requirement": list(mrp["net_requirement"].values())})

    if mrp["reorder_recommendations"]:
        st.warning("Reorder recommendations (materials falling short):")
        st.table({
            "Material": list(mrp["reorder_recommendations"].keys()),
            "Order Qty": [v["recommended_order_qty"] for v in mrp["reorder_recommendations"].values()],
            "Lead Time (days)": [v["lead_time_days"] for v in mrp["reorder_recommendations"].values()],
        })

    # ---- AI narration layer ----
    st.subheader("🤖 Agent's Explanation")
    if not api_key:
        st.info("Enter your Anthropic API key in the sidebar to get the AI narration of this plan.")
    else:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            prompt = f"""You are an S&OP planning assistant for a small ice cream company.
Below is a computed Master Production Schedule and Material Requirement Plan
in JSON. Explain it in plain, practical language for a small business owner:
- Summarize whether demand was fully met, and if not, which SKUs were cut and why.
- Point out which single constraint (capacity, manpower, or a specific material) was the real bottleneck.
- List the 2-3 most urgent reorder actions with their deadlines.
- Keep it under 200 words. Do not recompute any numbers -- only explain the ones given.

MPS:
{json.dumps(mps, indent=2)}

MRP:
{json.dumps(mrp, indent=2)}
"""
            with st.spinner("Agent is analyzing the plan..."):
                response = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
            st.write(response.content[0].text)
        except Exception as e:
            st.error(f"Could not reach the Claude API: {e}")

st.divider()
st.caption(
    "Note: all SKU/material assumptions (BOM, lead times, manpower baseline, priority order) "
    "are documented assumptions for this project -- see sop_engine.py for the full reference data."
)
