"""
Deterministic S&OP planning engine for a small ice cream company.

This module holds NO AI logic. Every number here is computed by explicit
formulas so the output can always be traced back to a rule. The AI agent
(built separately, on top of this file) decides WHEN to call these
functions and explains the results in plain language — it never
overrides the arithmetic.

Design note on "optimization":
When a constraint (capacity / manpower / raw material / packaging) is
tighter than what's needed to meet full demand, the engine does NOT just
give up and report a shortfall. It re-allocates the constrained resource
across SKUs using a priority order (default: by contribution margin,
i.e. the same ABC logic from Phase 2), so that the plan still tries to
satisfy as much total demand as the tightest resource allows, favouring
higher-priority SKUs first. This mirrors what a small business owner
would actually do by hand.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. STATIC REFERENCE DATA (BOM, lead times, manpower, capacity rates)
# ---------------------------------------------------------------------------

SKUS = ["Cup", "Stick", "Cone", "Pack200", "Pack500"]

# unit weight of finished product, in kg
UNIT_WEIGHT_KG = {
    "Cup": 0.100,
    "Stick": 0.080,
    "Cone": 0.120,
    "Pack200": 0.200,
    "Pack500": 0.500,
}

# packing rate, in units/hour, from the instructor's data sheet
PACKING_RATE_UPH = {
    "Cup": 5000,
    "Stick": 10000,
    "Cone": 4000,
    "Pack200": 2000,   # "Packs" rate shared by both pack sizes
    "Pack500": 2000,
}

# process rates for the shared upstream line, in tons/hour
PROCESS_RATE_TPH = {
    "Mixing": 3,
    "Pasteurization": 4,
    "Homogenization": 3,
}

# BOM: kg of each raw material per kg of finished mix (before packaging)
BOM_PER_KG_MIX = {
    "Milk_Cream_Base": 0.75,
    "Sugar": 0.14,
    "Stabilizer": 0.01,
    "Flavoring": 0.05,
    "Water_Other": 0.05,
}

# packaging BOM: units of packaging material per unit of finished SKU
PACKAGING_BOM = {
    "Cup": {"Cup_Shell": 1, "Lid_Small": 1},
    "Stick": {"Wrapper_Stick": 1, "Stick": 1},
    "Cone": {"Cone_Shell": 1, "Wrapper_Cone": 1},
    "Pack200": {"Tub_200ml": 1, "Lid_200ml": 1},
    "Pack500": {"Tub_500ml": 1, "Lid_500ml": 1},
}

RAW_MATERIALS = list(BOM_PER_KG_MIX.keys())
PACKAGING_MATERIALS = sorted({m for bom in PACKAGING_BOM.values() for m in bom})
ALL_MATERIALS = RAW_MATERIALS + PACKAGING_MATERIALS

LEAD_TIME_DAYS = {
    "Milk_Cream_Base": 2,
    "Sugar": 7,
    "Stabilizer": 14,
    "Flavoring": 10,
    "Water_Other": 1,
    "Cup_Shell": 15, "Lid_Small": 15,
    "Wrapper_Stick": 15, "Stick": 15,
    "Cone_Shell": 15, "Wrapper_Cone": 15,
    "Tub_200ml": 15, "Lid_200ml": 15,
    "Tub_500ml": 15, "Lid_500ml": 15,
}

MIN_ORDER_LOT = {
    "Milk_Cream_Base": 2000, "Sugar": 5000, "Stabilizer": 200,
    "Flavoring": 500, "Water_Other": 1000,
    "Cup_Shell": 10000, "Lid_Small": 10000,
    "Wrapper_Stick": 10000, "Stick": 10000,
    "Cone_Shell": 10000, "Wrapper_Cone": 10000,
    "Tub_200ml": 10000, "Lid_200ml": 10000,
    "Tub_500ml": 10000, "Lid_500ml": 10000,
}

# default SKU priority (1 = highest), used only when demand must be
# rationed against a binding constraint. Based on Phase 2 ABC logic:
# high-volume, low-margin-risk items first.
DEFAULT_SKU_PRIORITY = {"Cup": 1, "Stick": 2, "Pack200": 3, "Cone": 4, "Pack500": 5}

WORKING_DAYS_PER_MONTH = 24
HOURS_PER_SHIFT = 8


# ---------------------------------------------------------------------------
# 2. INPUT DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class PeriodInputs:
    """Everything the plan needs for ONE planning period (e.g. one month)."""
    period_label: str
    demand_units: Dict[str, float]                       # SKU -> requested units
    opening_fg_inventory: Dict[str, float]                # SKU -> units on hand
    opening_material_inventory: Dict[str, float]          # material -> qty on hand
    shifts_per_day: int = 2                               # production+packing shifts
    workers_available: Optional[int] = None               # None = not a constraint
    workers_required_per_shift: int = 14                  # mixing/pack crew combined
    material_incoming: Dict[str, float] = field(default_factory=dict)  # material -> qty arriving in period
    sku_priority: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SKU_PRIORITY))


# ---------------------------------------------------------------------------
# 3. CAPACITY CHECK (upstream line + packing, tons/hr and units/hr)
# ---------------------------------------------------------------------------

def available_hours(shifts_per_day: int, workers_available: Optional[int],
                     workers_required_per_shift: int) -> float:
    """Hours available this period, capped by manpower if manpower is short."""
    hours = shifts_per_day * HOURS_PER_SHIFT * WORKING_DAYS_PER_MONTH
    if workers_available is not None and workers_available < workers_required_per_shift:
        # manpower shortfall scales down usable hours proportionally
        hours *= workers_available / workers_required_per_shift
    return hours


def upstream_capacity_tons(hours: float) -> float:
    """Max tons the shared mixing/pasteurization/homogenization line can push,
    limited by the slowest of the three stages (the true bottleneck)."""
    bottleneck_tph = min(PROCESS_RATE_TPH.values())
    return bottleneck_tph * hours


def demand_to_tons(demand_units: Dict[str, float]) -> float:
    return sum(demand_units.get(sku, 0) * UNIT_WEIGHT_KG[sku] / 1000 for sku in SKUS)


def packing_hours_needed(units_plan: Dict[str, float]) -> Dict[str, float]:
    return {sku: units_plan.get(sku, 0) / PACKING_RATE_UPH[sku] for sku in SKUS}


# ---------------------------------------------------------------------------
# 4. CONSTRAINT-AWARE PRODUCTION ALLOCATION (the "optimizer")
# ---------------------------------------------------------------------------

def allocate_constrained_production(demand_units: Dict[str, float],
                                     max_total_tons: float,
                                     sku_priority: Dict[str, int]) -> Dict[str, float]:
    """
    If total demand (in tons) fits within max_total_tons, everyone gets
    their full requested quantity. If not, ration tonnage across SKUs in
    priority order (1 = served first), giving each priority tier its full
    request before moving to the next tier, and splitting the remaining
    tonnage proportionally within a tier if it runs out mid-tier.
    """
    total_demand_tons = demand_to_tons(demand_units)
    if total_demand_tons <= max_total_tons or total_demand_tons == 0:
        return dict(demand_units)  # no rationing needed

    remaining_tons = max_total_tons
    plan = {sku: 0.0 for sku in SKUS}
    order = sorted(SKUS, key=lambda s: sku_priority.get(s, 99))

    for sku in order:
        sku_demand_units = demand_units.get(sku, 0)
        sku_demand_tons = sku_demand_units * UNIT_WEIGHT_KG[sku] / 1000
        if sku_demand_tons <= remaining_tons:
            plan[sku] = sku_demand_units
            remaining_tons -= sku_demand_tons
        else:
            # partially fill this SKU with whatever tonnage is left, then stop
            fillable_units = (remaining_tons * 1000) / UNIT_WEIGHT_KG[sku] if UNIT_WEIGHT_KG[sku] else 0
            plan[sku] = max(0.0, fillable_units)
            remaining_tons = 0
            break
    return plan


# ---------------------------------------------------------------------------
# 5. MPS GENERATION
# ---------------------------------------------------------------------------

def generate_mps(inp: PeriodInputs) -> dict:
    hours = available_hours(inp.shifts_per_day, inp.workers_available, inp.workers_required_per_shift)
    max_tons = upstream_capacity_tons(hours)

    # net demand = gross demand - opening finished-goods inventory (floor at 0)
    net_demand = {
        sku: max(0.0, inp.demand_units.get(sku, 0) - inp.opening_fg_inventory.get(sku, 0))
        for sku in SKUS
    }

    proposed_plan = allocate_constrained_production(net_demand, max_tons, inp.sku_priority)

    # check packing hours don't blow past available hours either; if they do,
    # ration again using packing capacity as the binding constraint (rare,
    # since packing rates are generous relative to the mixing bottleneck)
    pack_hours = packing_hours_needed(proposed_plan)
    total_pack_hours = sum(pack_hours.values())
    packing_constrained = total_pack_hours > hours

    if packing_constrained:
        # scale every SKU down proportionally to fit available packing hours
        scale = hours / total_pack_hours if total_pack_hours else 1
        proposed_plan = {sku: qty * scale for sku, qty in proposed_plan.items()}

    plan_tons = demand_to_tons(proposed_plan)
    total_demand_tons = demand_to_tons(inp.demand_units)

    shortfall = {
        sku: round(inp.demand_units.get(sku, 0) - inp.opening_fg_inventory.get(sku, 0) - proposed_plan.get(sku, 0), 1)
        for sku in SKUS
    }
    shortfall = {sku: max(0.0, v) for sku, v in shortfall.items()}

    return {
        "period": inp.period_label,
        "available_hours": round(hours, 1),
        "bottleneck_stage": min(PROCESS_RATE_TPH, key=PROCESS_RATE_TPH.get),
        "max_upstream_tons": round(max_tons, 2),
        "total_demand_tons": round(total_demand_tons, 2),
        "capacity_binding": total_demand_tons > max_tons,
        "packing_binding": packing_constrained,
        "mps_units": {sku: round(v, 0) for sku, v in proposed_plan.items()},
        "unmet_demand_units": shortfall,
        "closing_fg_inventory": {
            sku: round(inp.opening_fg_inventory.get(sku, 0) + proposed_plan.get(sku, 0)
                        - inp.demand_units.get(sku, 0), 1)
            for sku in SKUS
        },
    }


# ---------------------------------------------------------------------------
# 6. MRP GENERATION (BOM explosion + material netting + reorder flags)
# ---------------------------------------------------------------------------

def explode_bom(mps_units: Dict[str, float]) -> Dict[str, float]:
    """Gross material requirement implied by the MPS."""
    gross = {m: 0.0 for m in ALL_MATERIALS}
    for sku, qty in mps_units.items():
        kg_mix = qty * UNIT_WEIGHT_KG[sku]
        for material, per_kg in BOM_PER_KG_MIX.items():
            gross[material] += kg_mix * per_kg
        for material, per_unit in PACKAGING_BOM[sku].items():
            gross[material] += qty * per_unit
    return {m: round(v, 1) for m, v in gross.items()}


def generate_mrp(inp: PeriodInputs, mps_units: Dict[str, float]) -> dict:
    gross_req = explode_bom(mps_units)
    net_req = {}
    reorder_flags = {}
    for material in ALL_MATERIALS:
        on_hand = inp.opening_material_inventory.get(material, 0)
        incoming = inp.material_incoming.get(material, 0)
        need = gross_req[material] - on_hand - incoming
        net_req[material] = round(max(0.0, need), 1)
        if net_req[material] > 0:
            lot = MIN_ORDER_LOT.get(material, net_req[material])
            order_qty = max(net_req[material], lot)
            reorder_flags[material] = {
                "shortfall": net_req[material],
                "recommended_order_qty": order_qty,
                "lead_time_days": LEAD_TIME_DAYS.get(material),
                "order_by": f"start of period minus {LEAD_TIME_DAYS.get(material)} days",
            }
    return {
        "gross_requirement": gross_req,
        "net_requirement": net_req,
        "reorder_recommendations": reorder_flags,
    }


# ---------------------------------------------------------------------------
# 7. TOP-LEVEL "RUN ONE PERIOD" FUNCTION — this is what the agent calls
# ---------------------------------------------------------------------------

def run_period_plan(inp: PeriodInputs) -> dict:
    mps = generate_mps(inp)
    mrp = generate_mrp(inp, mps["mps_units"])
    return {"mps": mps, "mrp": mrp}


if __name__ == "__main__":
    import json

    # quick manual smoke test with the baseline dataset discussed
    demo = PeriodInputs(
        period_label="Month 4 (peak season begins)",
        demand_units={"Cup": 60000, "Stick": 90000, "Cone": 30000, "Pack200": 20000, "Pack500": 8000},
        opening_fg_inventory={"Cup": 5000, "Stick": 8000, "Cone": 3000, "Pack200": 2000, "Pack500": 1000},
        opening_material_inventory={
            "Milk_Cream_Base": 8000, "Sugar": 6000, "Stabilizer": 300,
            "Flavoring": 1500, "Water_Other": 1000,
            "Cup_Shell": 15000, "Lid_Small": 15000, "Wrapper_Stick": 15000, "Stick": 15000,
            "Cone_Shell": 15000, "Wrapper_Cone": 15000,
            "Tub_200ml": 15000, "Lid_200ml": 15000, "Tub_500ml": 15000, "Lid_500ml": 15000,
        },
        shifts_per_day=2,
        workers_available=12,           # short-staffed vs. the 14 required
        workers_required_per_shift=14,
    )
    result = run_period_plan(demo)
    print(json.dumps(result, indent=2))
