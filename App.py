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
import sys
import json
import pandas as pd
import streamlit as st

# Ensure current directory is at the top of sys.path so Streamlit Cloud finds sop_engine.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Safe import for Google Generative AI
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    genai = None
    HAS_GEMINI = False

# Safe import for sop_engine
try:
    from sop_engine import (
        PeriodInputs, run_period_plan, SKUS, ALL_MATERIALS, RAW_MATERIALS, PACKAGING_MATERIALS,
        DEFAULT_SKU_PRIORITY, UNIT_WEIGHT_KG, PACKING_RATE_UPH, PROCESS_RATE_TPH,
        BOM_PER_KG_MIX, PACKAGING_BOM, LEAD_TIME_DAYS, MIN_ORDER_LOT,
        get_mps_summary_table, get_mrp_summary_table
    )
    HAS_ENGINE = True
except ImportError as e:
    ENGINE_ERROR = str(e)
    HAS_ENGINE = False

# ---------------------------------------------------------------------------
# Page Configuration & Advanced Aesthetic Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ice Cream S&OP Intelligence Agent",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Design System
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #FFFFFF;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #E0E7FF;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
    }

    .status-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #818CF8;
        margin-bottom: 1rem;
    }

    .kpi-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
    }

    .kpi-sub {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    .sub-green { color: #10B981; }
    .sub-amber { color: #F59E0B; }
    .sub-rose { color: #EF4444; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4338CA !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    .ai-box {
        background-color: #F8FAFC;
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Banner Header
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <div class="status-badge">⚡ Enterprise S&OP Planning Engine v3.0</div>
        <div class="hero-title">🍦 Ice Cream S&OP Intelligence Agent</div>
        <div class="hero-subtitle">
            Deterministic Master Production Schedule (MPS) & Material Requirements Planning (MRP) Engine 
            integrated with <strong>Google Gemini Generative AI Executive Briefing</strong>.
        </div>
    </div>
""", unsafe_allow_html=True)

if not HAS_ENGINE:
    st.error(
        f"⚠️ **Engine File Missing on Server**: `sop_engine.py` could not be loaded.\n\n"
        f"**Details**: `{ENGINE_ERROR}`\n\n"
        "**Solution**: Please ensure `sop_engine.py` is located in the same GitHub directory as `App.py` / `app.py` and push it to GitHub."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: API Key Configuration & Operational Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 Gemini API Control")
    
    # Safe retrieval of secrets or env var for pre-configured key
    default_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]
        elif "GOOGLE_API_KEY" in st.secrets:
            default_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    if not default_key:
        default_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Get your free API key at https://aistudio.google.com/"
    )
    
    if api_key:
        st.success("✅ API Key Connected")
    else:
        st.info("💡 Enter key to unlock AI Agent Briefings")

    st.divider()
    st.markdown("### ⚙️ Production Constraints")
    shifts_per_day = st.slider("Shifts per day (8 hrs/shift)", 1, 3, 2)
    workers_required = st.number_input("Workers required per shift", value=14, min_value=1)
    workers_available = st.number_input(
        "Workers available this period", 
        value=14, 
        min_value=0,
        help="Simulate staff shortages by reducing this number."
    )

    st.divider()
    st.markdown("### 🤖 Intelligence Model")
    model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-pro"],
        index=0,
        help="Recommended: gemini-1.5-flash for high performance."
    )

# ---------------------------------------------------------------------------
# Main Form: Inputs Section
# ---------------------------------------------------------------------------
with st.expander("📝 1. Operational Input Parameters (Demand, Inventory & In-Transit)", expanded=True):
    period_label = st.text_input("Planning Period Label", value="Month 4 (Peak Season Opening)")
    
    col_d, col_i = st.columns(2)
    
    with col_d:
        st.markdown("#### 📈 Gross Demand Forecast (Units)")
        demand_units = {}
        d_cols = st.columns(len(SKUS))
        default_demands = {"Cup": 60000, "Stick": 90000, "Cone": 30000, "Pack200": 20000, "Pack500": 8000}
        for i, sku in enumerate(SKUS):
            with d_cols[i % len(SKUS)]:
                demand_units[sku] = st.number_input(
                    f"{sku}", value=default_demands.get(sku, 50000), min_value=0, step=1000, key=f"d_{sku}"
                )

    with col_i:
        st.markdown("#### 🏢 Opening Finished Goods Stock (Units)")
        opening_fg = {}
        i_cols = st.columns(len(SKUS))
        default_fg = {"Cup": 5000, "Stick": 8000, "Cone": 3000, "Pack200": 2000, "Pack500": 1000}
        for i, sku in enumerate(SKUS):
            with i_cols[i % len(SKUS)]:
                opening_fg[sku] = st.number_input(
                    f"{sku}", value=default_fg.get(sku, 2000), min_value=0, step=500, key=f"i_{sku}"
                )

    st.divider()
    st.markdown("#### 📦 Raw Materials & Packaging Inventory")
    mat_tab1, mat_tab2 = st.tabs(["🌾 Raw Materials Stock & In-Transit", "🏷️ Packaging Stock & In-Transit"])
    
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

run_clicked = st.button("⚡ Run Deterministic S&OP Engine & AI Agent", type="primary", use_container_width=True)

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
        with st.spinner("Calculating deterministic MPS capacity check, MRP explosion, and Day 1-24 dispatch grid..."):
            result = run_period_plan(inputs)
            st.session_state["last_result"] = result
            st.session_state["last_inputs"] = inputs
    else:
        result = st.session_state["last_result"]
        inputs = st.session_state["last_inputs"]

    mps, mrp, daily = result["mps"], result["mrp"], result.get("daily", {})

    # ---------------------------------------------------------------------------
    # Premium KPI Cards Header
    # ---------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    total_demand_tons = mps["total_demand_tons"]
    total_mps_units = sum(mps["mps_units"].values())
    total_mps_tons = sum(mps["mps_units"][sku] * UNIT_WEIGHT_KG[sku] / 1000 for sku in SKUS)
    cap_pct = (total_mps_tons / mps["max_upstream_tons"] * 100) if mps["max_upstream_tons"] > 0 else 0
    unmet_skus = sum(1 for v in mps["unmet_demand_units"].values() if v > 0)
    reorders_needed = len(mrp["reorder_recommendations"])

    with m_col1:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #4338CA;">
                <div class="kpi-label">Gross Demand</div>
                <div class="kpi-value">{total_demand_tons:.1f}T</div>
                <div class="kpi-sub sub-green">{sum(inputs.demand_units.values()):,} units</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10B981;">
                <div class="kpi-label">Planned Production</div>
                <div class="kpi-value">{total_mps_tons:.1f}T</div>
                <div class="kpi-sub sub-green">{int(total_mps_units):,} units</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #F59E0B;">
                <div class="kpi-label">Line Capacity</div>
                <div class="kpi-value">{cap_pct:.1f}%</div>
                <div class="kpi-sub sub-amber">Max: {mps['max_upstream_tons']} Tons</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {'#EF4444' if unmet_skus > 0 else '#10B981'};">
                <div class="kpi-label">Shortfall SKUs</div>
                <div class="kpi-value">{unmet_skus}</div>
                <div class="kpi-sub {'sub-rose' if unmet_skus > 0 else 'sub-green'}">
                    {'⚠️ Rationed' if unmet_skus > 0 else '✅ Satisfied'}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {'#EF4444' if reorders_needed > 0 else '#10B981'};">
                <div class="kpi-label">Reorder Actions</div>
                <div class="kpi-value">{reorders_needed}</div>
                <div class="kpi-sub {'sub-rose' if reorders_needed > 0 else 'sub-green'}">
                    {'🚨 Shortage Risk' if reorders_needed > 0 else '✅ Fully Stocked'}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Main Tabs: MPS/MRP Results, Day-Wise Schedule, Gemini Agent Briefing, Reference, Export
    # ---------------------------------------------------------------------------
    tab_plan, tab_daily, tab_agent, tab_ref, tab_export = st.tabs([
        "📊 Monthly S&OP Summary",
        "📅 Day-Wise Dispatch (Days 1–24)",
        "🤖 Gemini AI Agent Briefing",
        "⚙️ BOM & Line Reference",
        "📥 Export S&OP Reports"
    ])

    # ---------------------------------------------------------------------------
    # TAB 1: MONTHLY SUMMARY TABLES AND VISUALIZATIONS
    # ---------------------------------------------------------------------------
    with tab_plan:
        st.subheader("📋 Master Production Schedule (MPS) Summary")
        
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
                f"⚠️ **Capacity Bottleneck Active**: Shared mixing/pasteurization line limit is **{mps['max_upstream_tons']} Tons** "
                f"({mps['available_hours']} hours at bottleneck stage: *{mps['bottleneck_stage']}*). "
                "Production has been rationed according to ABC SKU contribution margin priority."
            )

        st.markdown("#### 📈 Gross Demand vs. Planned Production by SKU")
        df_chart = df_mps.set_index("SKU")[["Gross Demand", "Planned Production"]]
        st.bar_chart(df_chart)

        st.divider()
        st.subheader("📦 Material Requirement Plan (MRP) Summary")
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
    # TAB 2: DAY-WISE DISPATCH SCHEDULE (DAYS 1 TO 24)
    # ---------------------------------------------------------------------------
    with tab_daily:
        st.subheader("📅 Day-by-Day Master Production Schedule (Days 1 to 24)")
        
        # Display Daily Operating Constraints
        daily_hours = mps["available_hours"] / 24.0
        daily_max_tons = mps["max_upstream_tons"] / 24.0
        daily_prod_tons = total_mps_tons / 24.0
        
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            st.metric("Operating Hours / Day", f"{daily_hours:.1f} Hours", f"{inputs.shifts_per_day} shifts/day")
        with d_c2:
            st.metric("Daily Line Capacity Limit", f"{daily_max_tons:.2f} Tons/Day", f"Bottleneck: {mps['bottleneck_stage']}")
        with d_c3:
            st.metric("Daily Planned Output", f"{daily_prod_tons:.2f} Tons/Day", f"{int(total_mps_units/24):,} units/day")
        with d_c4:
            st.metric("Daily Capacity Loading", f"{(daily_prod_tons/daily_max_tons*100):.1f}%", f"{inputs.workers_available}/{inputs.workers_required_per_shift} Workers")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Operational dispatch matrix showing daily production runs and closing stock trajectory for each SKU across 24 working days.")
        
        df_daily_mps = pd.DataFrame(daily.get("daily_mps_grid", []))
        st.dataframe(df_daily_mps, use_container_width=True)

        st.divider()
        st.subheader("📊 Daily Production Dispatch Breakdown (Units / Day)")
        
        # Prepare daily production bar chart
        df_prod_only = df_daily_mps[df_daily_mps["Metric"] == "Planned Prod (Units)"].set_index("SKU")
        day_cols = [f"Day {d}" for d in range(1, 25)]
        df_prod_chart = df_prod_only[day_cols].T
        st.bar_chart(df_prod_chart)

        st.divider()
        st.subheader("📦 Day-by-Day Raw Material & Packaging Depletion Schedule")
        st.caption("Tracks inventory levels from Day 1 to Day 24, flagging exact stockout days if material reorders are delayed.")
        
        df_daily_mat = pd.DataFrame(daily.get("daily_material_grid", []))
        st.dataframe(df_daily_mat, use_container_width=True)

    # ---------------------------------------------------------------------------
    # TAB 3: GEMINI AI AGENT EXECUTIVE BRIEFING
    # ---------------------------------------------------------------------------
    with tab_agent:
        st.subheader("🤖 S&OP AI Agent Executive Briefing")
        
        if not HAS_GEMINI:
            st.error(
                "⚠️ **Package Missing**: `google-generativeai` is not installed on this server/environment.\n\n"
                "To fix this on Streamlit Cloud or locally, make sure your **`requirements.txt`** contains:\n"
                "```text\nstreamlit>=1.35.0\ngoogle-generativeai>=0.8.0\npandas>=2.0.0\n```\n"
                "Then reboot your app on Streamlit Cloud."
            )
        elif not api_key:
            st.warning("⚠️ Please enter a valid Google Gemini API key in the sidebar to generate the AI briefing.")
        else:
            try:
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
                    model = genai.GenerativeModel(model_choice)
                    response = model.generate_content(system_prompt)
                    
                    st.markdown(f"""
                        <div class="ai-box">
                            {response.text}
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ AI Executive Briefing generated successfully via Gemini API.")

            except Exception as e:
                st.error(f"❌ Could not generate AI briefing: {e}")
                st.info("Tip: Ensure your API key is valid and check model availability in your region.")

    # ---------------------------------------------------------------------------
    # TAB 4: REFERENCE DATA & MASTER EXCEL DOWNLOAD
    # ---------------------------------------------------------------------------
    with tab_ref:
        st.subheader("⚙️ Industry Process & BOM Reference Specs")
        
        master_excel_path = os.path.join(CURRENT_DIR, "SOP_Master_Constraints_and_Processes.xlsx")
        if os.path.exists(master_excel_path):
            with open(master_excel_path, "rb") as f:
                excel_bytes = f.read()
            st.download_button(
                label="📊 Download Master S&OP Constraints & Processes Workbook (.xlsx)",
                data=excel_bytes,
                file_name="SOP_Master_Constraints_and_Processes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        st.divider()
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
    # TAB 5: EXPORT REPORTS
    # ---------------------------------------------------------------------------
    with tab_export:
        st.subheader("📥 Export S&OP Reports for Phase 3 Deliverables")
        st.write("Download period monthly & day-wise MPS, MRP, raw JSON, and Master Constraints Excel workbook for report documentation.")
        
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        
        with exp_col1:
            csv_mps = df_mps.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Monthly MPS (CSV)",
                data=csv_mps,
                file_name=f"MPS_Monthly_{inputs.period_label.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        with exp_col2:
            csv_mrp = df_mrp.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Monthly MRP (CSV)",
                data=csv_mrp,
                file_name=f"MRP_Monthly_{inputs.period_label.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        with exp_col3:
            csv_daily_mps = pd.DataFrame(daily.get("daily_mps_grid", [])).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📅 Day-Wise Schedule (CSV)",
                data=csv_daily_mps,
                file_name=f"SOP_DayWise_Schedule_{inputs.period_label.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        with exp_col4:
            json_data = json.dumps(result, indent=2).encode('utf-8')
            st.download_button(
                label="📦 Full S&OP (JSON)",
                data=json_data,
                file_name=f"SOP_Full_Plan_{inputs.period_label.replace(' ', '_')}.json",
                mime="application/json"
            )

        st.divider()
        if os.path.exists(master_excel_path):
            st.download_button(
                label="📊 Download Master Constraints & Processes Workbook (.xlsx)",
                data=excel_bytes,
                file_name="SOP_Master_Constraints_and_Processes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.divider()
st.caption("Powered by Google Gemini Generative AI • Deterministic S&OP Engine • Designed for Academic & Industrial Operations Planning")
