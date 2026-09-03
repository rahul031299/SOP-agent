"""
Streamlit front-end for the Ice Cream S&OP Agent powered by Google Gemini API.

Architecture:
  1. User inputs operational parameters (demand, inventory, manpower, incoming shipments).
  2. On submit, the DETERMINISTIC engine (sop_engine.py) computes exact MPS and MRP numbers.
     Every formula is fixed and traceable — AI does not modify arithmetic.
  3. Structured results are passed to Google Gemini API (via google-generativeai SDK).
  4. Gemini generates an Executive Briefing interpreting bottlenecks, priority cuts, and urgent reorders.

Run locally:
    streamlit run app.py
"""

import os
import json
import pandas as pd
import streamlit as st
import google.generativeai as genai

from sop_engine import (
    PeriodInputs, run_period_plan, SKUS, ALL_MATERIALS, RAW_MATERIALS, PACKAGING_MATERIALS,
    DEFAULT_SKU_PRIORITY, UNIT_WEIGHT_KG, PACKING_RATE_UPH, PROCESS_RATE_TPH,
    BOM_PER_KG_MIX, PACKAGING_BOM, LEAD_TIME_DAYS, MIN_ORDER_LOT,
    get_mps_summary_table, get_mrp_summary_table
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling (Must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ice Cream S&OP Intelligence Agent",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_headers=True)

st.markdown('<div class="main-header">🍦 Ice Cream S&OP Intelligence Agent</div>', unsafe_allow_headers=True)
st.markdown(
    '<div class="sub-header">'
    'Deterministic Master Production Schedule (MPS) & Material Requirements Planning (MRP) Engine + '
    '<strong>Google Gemini AI Narration Layer</strong>'
    '</div>',
    unsafe_allow_headers=True
)

# ---------------------------------------------------------------------------
# Sidebar: API Key Configuration & Operational Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Gemini API Setup")
    
    # Safe retrieval of secrets or env var for pre-configured key
    default_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]
        elif "GOOGLE_API_KEY" in st.secrets:
            default_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        # st.secrets raises an exception if secrets.toml does not exist
        pass

    if not default_key:
        default_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Get your API key free at https://aistudio.google.com/"
    )
    
    if api_key:
        st.success("API Key detected")
    else:
        st.info("Enter Gemini API Key to enable AI Agent Briefing")

    st.divider()
    st.header("⚡ Plant & Capacity Setup")
    shifts_per_day = st.slider("Shifts per day (8 hrs/shift)", 1, 3, 2)
    workers_required = st.number_input("Workers required per shift", value=14, min_value=1)
    workers_available = st.number_input(
        "Workers available this period", 
        value=14, 
        min_value=0,
        help="Simulate manpower shortages by reducing this number below baseline."
    )

    st.divider()
    st.header("🎯 Target Model")
    model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-pro"],
        index=0,
        help="Recommended: gemini-1.5-flash for fast, reliable reasoning."
    )

# ---------------------------------------------------------------------------
# Main Form: Inputs Section
# ---------------------------------------------------------------------------
with st.expander("📝 1. Enter Period Parameters (Demand, Inventory & In-Transit)", expanded=True):
    period_label = st.text_input("Planning Period Label", value="Month 4 (Peak Season Opening)")
    
    col_d, col_i = st.columns(2)
    
    with col_d:
        st.subheader("Gross Demand Forecast (Units)")
        demand_units = {}
        d_cols = st.columns(len(SKUS))
        default_demands = {"Cup": 60000, "Stick": 90000, "Cone": 30000, "Pack200": 20000, "Pack500": 8000}
        for i, sku in enumerate(SKUS):
            with d_cols[i % len(SKUS)]:
                demand_units[sku] = st.number_input(
                    f"{sku}", value=default_demands.get(sku, 50000), min_value=0, step=1000, key=f"d_{sku}"
                )

    with col_i:
        st.subheader("Opening Finished Goods Stock (Units)")
        opening_fg = {}
        i_cols = st.columns(len(SKUS))
        default_fg = {"Cup": 5000, "Stick": 8000, "Cone": 3000, "Pack200": 2000, "Pack500": 1000}
        for i, sku in enumerate(SKUS):
            with i_cols[i % len(SKUS)]:
                opening_fg[sku] = st.number_input(
                    f"{sku}", value=default_fg.get(sku, 2000), min_value=0, step=500, key=f"i_{sku}"
                )

    st.subheader("Raw Materials & Packaging Inventory")
    mat_tab1, mat_tab2 = st.tabs(["Raw Materials Stock & Incoming", "Packaging Stock & Incoming"])
    
    opening_materials = {}
    material_incoming = {}

    default_raw_mat = {"Milk_Cream_Base": 8000, "Sugar": 6000, "Stabilizer": 300, "Flavoring": 1500, "Water_Other": 1000}
    default_pack_mat = 15000

    with mat_tab1:
        rm_cols = st.columns(len(RAW_MATERIALS))
        for i, mat in enumerate(RAW_MATERIALS):
            with rm_cols[i]:
                st.markdown(f"**{mat}**")
                opening_materials[mat] = st.number_input("On Hand (kg)", value=default_raw_mat.get(mat, 5000), min_value=0, key=f"m_{mat}")
                material_incoming[mat] = st.number_input("In-Transit (kg)", value=0, min_value=0, key=f"inc_{mat}")

    with mat_tab2:
        pk_cols = st.columns(3)
        for i, mat in enumerate(PACKAGING_MATERIALS):
            with pk_cols[i % 3]:
                st.markdown(f"**{mat}**")
                opening_materials[mat] = st.number_input("On Hand (units)", value=default_pack_mat, min_value=0, key=f"m_{mat}")
                material_incoming[mat] = st.number_input("In-Transit (units)", value=0, min_value=0, key=f"inc_{mat}")

run_clicked = st.button("🚀 Compute S&OP Plan & Generate AI Briefing", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# S&OP Calculation & Results Display
# ---------------------------------------------------------------------------
if run_clicked or "last_result" in st.session_state:
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
        with st.spinner("Executing deterministic MPS capacity check and MRP explosion..."):
            result = run_period_plan(inputs)
            st.session_state["last_result"] = result
            st.session_state["last_inputs"] = inputs
    else:
        result = st.session_state["last_result"]
        inputs = st.session_state["last_inputs"]

    mps, mrp = result["mps"], result["mrp"]

    # ---------------------------------------------------------------------------
    # KPI Metrics Header Bar
    # ---------------------------------------------------------------------------
    st.divider()
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    total_demand_tons = mps["total_demand_tons"]
    total_mps_units = sum(mps["mps_units"].values())
    total_mps_tons = sum(mps["mps_units"][sku] * UNIT_WEIGHT_KG[sku] / 1000 for sku in SKUS)
    cap_pct = (total_mps_tons / mps["max_upstream_tons"] * 100) if mps["max_upstream_tons"] > 0 else 0
    unmet_skus = sum(1 for v in mps["unmet_demand_units"].values() if v > 0)
    reorders_needed = len(mrp["reorder_recommendations"])

    with m_col1:
        st.metric("Gross Demand", f"{total_demand_tons:.2f} Tons", f"{sum(inputs.demand_units.values()):,} units")
    with m_col2:
        st.metric("Planned Production", f"{total_mps_tons:.2f} Tons", f"{int(total_mps_units):,} units")
    with m_col3:
        st.metric("Capacity Utilization", f"{cap_pct:.1f}%", f"Max: {mps['max_upstream_tons']} Tons")
    with m_col4:
        st.metric("Shortfall SKUs", f"{unmet_skus}", delta="Capacity Constrained" if unmet_skus > 0 else "Fully Satisfied", delta_color="inverse" if unmet_skus > 0 else "normal")
    with m_col5:
        st.metric("Reorder Action Items", f"{reorders_needed}", delta="Shortage Risk" if reorders_needed > 0 else "Stocked", delta_color="inverse" if reorders_needed > 0 else "normal")

    # ---------------------------------------------------------------------------
    # Main Tabs: MPS/MRP Results, Gemini Agent Briefing, Reference Data, Export
    # ---------------------------------------------------------------------------
    tab_plan, tab_agent, tab_ref, tab_export = st.tabs([
        "📊 Production & Material Plan (MPS / MRP)",
        "🤖 Gemini AI Agent Briefing",
        "⚙️ BOM & Capacity Reference",
        "📥 Export Reports"
    ])

    # ---------------------------------------------------------------------------
    # TAB 1: MPS & MRP TABLES AND VISUALIZATIONS
    # ---------------------------------------------------------------------------
    with tab_plan:
        st.subheader("📋 Master Production Schedule (MPS)")
        
        mps_rows = get_mps_summary_table(mps, inputs.demand_units, inputs.opening_fg_inventory)
        df_mps = pd.DataFrame(mps_rows)
        
        st.dataframe(
            df_mps.style.format({
                "Gross Demand": "{:,.0f}",
                "Opening Inventory": "{:,.0f}",
                "Net Demand": "{:,.0f}",
                "Planned Production": "{:,.0f}",
                "Unmet Demand": "{:,.0f}",
                "Closing Inventory": "{:,.0f}"
            }),
            use_container_width=True
        )

        if mps["capacity_binding"]:
            st.warning(
                f"⚠️ **Capacity Constraint Active**: Available upstream line limit is **{mps['max_upstream_tons']} Tons** "
                f"({mps['available_hours']} hours at bottleneck stage: *{mps['bottleneck_stage']}*). "
                "Production has been rationed according to SKU priority."
            )

        # Bar chart comparing Demand vs Planned Production
        st.markdown("#### 📈 Demand vs. Planned Production by SKU")
        df_chart = df_mps.set_index("SKU")[["Gross Demand", "Planned Production"]]
        st.bar_chart(df_chart)

        st.subheader("📦 Material Requirement Plan (MRP)")
        mrp_rows = get_mrp_summary_table(mrp, inputs.opening_material_inventory, inputs.material_incoming)
        df_mrp = pd.DataFrame(mrp_rows)

        st.dataframe(
            df_mrp.style.format({
                "Gross Requirement": "{:,.1f}",
                "On Hand Inventory": "{:,.1f}",
                "Incoming (In-Transit)": "{:,.1f}",
                "Net Requirement": "{:,.1f}",
                "Recommended Order Qty": "{:,.1f}"
            }),
            use_container_width=True
        )

    # ---------------------------------------------------------------------------
    # TAB 2: GEMINI AI AGENT EXECUTIVE BRIEFING
    # ---------------------------------------------------------------------------
    with tab_agent:
        st.subheader("🤖 S&OP AI Agent Executive Briefing")
        
        if not api_key:
            st.warning("⚠️ Please enter a valid Google Gemini API key in the sidebar to generate the AI briefing.")
        else:
            try:
                # Configure Google Generative AI
                genai.configure(api_key=api_key)

                system_prompt = f"""
You are an expert Sales & Operations Planning (S&OP) Executive Assistant for a manufacturing plant.
Analyze the following computed Master Production Schedule (MPS) and Material Requirement Plan (MRP) JSON datasets.

Your goal is to provide a crisp, highly actionable S&OP Executive Briefing for plant operations leadership.

CRITICAL INSTRUCTIONS:
- DO NOT alter or recalculate any numbers. All calculations in the JSON are ground-truth arithmetic.
- Structure your response using markdown with the following headers:

### 🎯 Executive S&OP Summary
- State period performance ({inputs.period_label}).
- Highlight total demand vs total planned production tonnage.

### ⚙️ Constraint & Bottleneck Analysis
- Identify the exact bottleneck ({mps['bottleneck_stage']}) and capacity utilization ({cap_pct:.1f}%).
- Explain any SKU production rationing decisions based on priority logic.

### 🚨 Critical Material Reorders & Deadlines
- List all materials with net shortages, required order quantities, lead times, and urgency.

### 💡 Operational Action Items
- Provide 3 specific, practical recommendations for the plant manager.

---
MPS Data:
{json.dumps(mps, indent=2)}

MRP Data:
{json.dumps(mrp, indent=2)}
"""

                with st.spinner(f"Querying Google Gemini ({model_choice}) for executive briefing..."):
                    # Call Gemini Model
                    model = genai.GenerativeModel(model_choice)
                    response = model.generate_content(system_prompt)
                    
                    st.markdown(response.text)
                    st.success("✅ AI Executive Briefing generated successfully via Gemini API.")

            except Exception as e:
                st.error(f"❌ Could not generate AI briefing: {e}")
                st.info("Tip: Ensure your API key is valid and check model availability in your region.")

    # ---------------------------------------------------------------------------
    # TAB 3: REFERENCE DATA
    # ---------------------------------------------------------------------------
    with tab_ref:
        st.subheader("⚙️ Industry Process & BOM Reference Specs")
        
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            st.markdown("#### Finished Goods SKU Specs")
            df_sku = pd.DataFrame([
                {"SKU": sku, "Unit Weight (kg)": UNIT_WEIGHT_KG[sku], "Packing Rate (units/hr)": PACKING_RATE_UPH[sku], "Priority": DEFAULT_SKU_PRIORITY[sku]}
                for sku in SKUS
            ])
            st.table(df_sku)

            st.markdown("#### Upstream Line Bottleneck Capacities")
            df_proc = pd.DataFrame([
                {"Process Stage": stage, "Capacity Rate (Tons/Hr)": tph}
                for stage, tph in PROCESS_RATE_TPH.items()
            ])
            st.table(df_proc)

        with ref_col2:
            st.markdown("#### Raw Material BOM (Per Kg Ice Cream Mix)")
            df_bom = pd.DataFrame([
                {"Raw Material": mat, "Ratio (kg/kg Mix)": ratio}
                for mat, ratio in BOM_PER_KG_MIX.items()
            ])
            st.table(df_bom)

            st.markdown("#### Material Lead Times & Min Order Lots")
            df_lead = pd.DataFrame([
                {"Material": mat, "Lead Time (Days)": LEAD_TIME_DAYS[mat], "Min Order Lot": MIN_ORDER_LOT[mat]}
                for mat in ALL_MATERIALS
            ])
            st.dataframe(df_lead, use_container_width=True)

    # ---------------------------------------------------------------------------
    # TAB 4: EXPORT REPORTS
    # ---------------------------------------------------------------------------
    with tab_export:
        st.subheader("📥 Export S&OP Reports for Phase 3 Deliverables")
        st.write("Download period MPS, MRP, and raw JSON plans for report documentation and presentations.")
        
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            csv_mps = df_mps.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download MPS Report (CSV)",
                data=csv_mps,
                file_name=f"MPS_Plan_{inputs.period_label.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        with exp_col2:
            csv_mrp = df_mrp.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download MRP Report (CSV)",
                data=csv_mrp,
                file_name=f"MRP_Plan_{inputs.period_label.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        with exp_col3:
            json_data = json.dumps(result, indent=2).encode('utf-8')
            st.download_button(
                label="📦 Download Full S&OP (JSON)",
                data=json_data,
                file_name=f"SOP_Full_Plan_{inputs.period_label.replace(' ', '_')}.json",
                mime="application/json"
            )

st.divider()
st.caption("Powered by Google Gemini Generative AI • Deterministic S&OP Engine • Designed for Academic & Industrial Operations Planning")
