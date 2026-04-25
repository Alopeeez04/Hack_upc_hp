"""
Phase 2 — Simulation Engine: HP Metal Jet S100 Digital Twin
============================================================
Single-file version. Requires phase1.py in the same folder.

Run directly with:
    python phase2.py

Dependencies:
    pip install scikit-learn numpy matplotlib pandas
"""

from __future__ import annotations

import csv
import math
import random
import sqlite3
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from phase1 import LogicEngine, EnvironmentalDrivers, OperationalStatus


# ─────────────────────────────────────────────
#  SimulationConfig
# ─────────────────────────────────────────────

@dataclass
class SimulationConfig:
    scenario_id:     str
    total_hours:     float
    hours_per_step:  float
    profile_fn:      Callable[[float, float], EnvironmentalDrivers]
    seed:            int  = 42
    description:     str  = ""
    enable_ai_agent: bool = True


# ─────────────────────────────────────────────
#  Historian
# ─────────────────────────────────────────────

class Historian:
    DB_FILE  = "historian.db"
    CSV_FILE = "historian.csv"

    def __init__(self):
        self._records = []
        self._conn = sqlite3.connect(self.DB_FILE)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                scenario_id TEXT, timestamp TEXT, elapsed_hours REAL,
                temperature_stress REAL, contamination REAL,
                operational_load REAL, maintenance_level REAL, shock_magnitude REAL,
                blade_health REAL, blade_status TEXT, blade_thickness REAL,
                nozzle_health REAL, nozzle_status TEXT, nozzle_clog REAL,
                heater_health REAL, heater_status TEXT, heater_resistance REAL,
                cascade_events TEXT, maintenance_action TEXT,
                PRIMARY KEY (scenario_id, timestamp)
            )
        """)
        self._conn.commit()

    def record(self, scenario_id, sim_time, elapsed_hours, drivers, report, maintenance_action=""):
        blade  = report.get("recoater_blade")
        nozzle = report.get("nozzle_plate")
        heater = report.get("heating_element")
        row = dict(
            scenario_id=scenario_id, timestamp=sim_time.isoformat(), elapsed_hours=elapsed_hours,
            temperature_stress=drivers.temperature_stress, contamination=drivers.contamination,
            operational_load=drivers.operational_load, maintenance_level=drivers.maintenance_level,
            shock_magnitude=drivers.shock_magnitude,
            blade_health=blade.health_index, blade_status=blade.operational_status.value,
            blade_thickness=blade.metrics.get("blade_thickness_norm", 0),
            nozzle_health=nozzle.health_index, nozzle_status=nozzle.operational_status.value,
            nozzle_clog=nozzle.metrics.get("clog_fraction", 0),
            heater_health=heater.health_index, heater_status=heater.operational_status.value,
            heater_resistance=heater.metrics.get("normalised_resistance", 1),
            cascade_events=" | ".join(report.cascade_events),
            maintenance_action=maintenance_action,
        )
        self._records.append(row)
        self._conn.execute("""
            INSERT OR REPLACE INTO telemetry VALUES (
                :scenario_id,:timestamp,:elapsed_hours,
                :temperature_stress,:contamination,:operational_load,
                :maintenance_level,:shock_magnitude,
                :blade_health,:blade_status,:blade_thickness,
                :nozzle_health,:nozzle_status,:nozzle_clog,
                :heater_health,:heater_status,:heater_resistance,
                :cascade_events,:maintenance_action
            )
        """, row)
        self._conn.commit()

    def export_csv(self):
        if not self._records:
            return
        with open(self.CSV_FILE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._records[0].keys()))
            w.writeheader()
            w.writerows(self._records)
        print(f"  📄  CSV saved → {self.CSV_FILE}")

    def query(self, scenario_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM telemetry WHERE scenario_id=? ORDER BY elapsed_hours",
            self._conn, params=(scenario_id,)
        )

    def close(self):
        self._conn.close()


# ─────────────────────────────────────────────
#  AI Maintenance Agent
# ─────────────────────────────────────────────

class MaintenanceAgent:
    COOLDOWN_HOURS  = 48.0
    DEGRADED_THRESH = 0.60

    def __init__(self):
        self._last_h        = -999.0
        self._interventions = 0
        self._pending_boost = 0.0
        self._pending_label = ""

    def apply_boost(self, drivers: EnvironmentalDrivers) -> str:
        """Apply pending boost to drivers BEFORE engine.step()."""
        label = self._pending_label
        if self._pending_boost > 0:
            drivers.maintenance_level = min(1.0, drivers.maintenance_level + self._pending_boost)
            self._pending_boost = 0.0
            self._pending_label = ""
        return label

    def observe(self, report, elapsed: float):
        """Decide next intervention AFTER engine.step()."""
        if elapsed - self._last_h < self.COOLDOWN_HOURS:
            return
        statuses = [c.operational_status for c in report.components]
        min_h    = min(c.health_index for c in report.components)

        if OperationalStatus.CRITICAL in statuses or OperationalStatus.FAILED in statuses:
            self._last_h = elapsed
            self._interventions += 1
            self._pending_boost = 0.4
            self._pending_label = f"EMERGENCY @ {elapsed:.0f}h (health={min_h:.2f})"
        elif OperationalStatus.DEGRADED in statuses and min_h < self.DEGRADED_THRESH:
            self._last_h = elapsed
            self._interventions += 1
            self._pending_boost = 0.2
            self._pending_label = f"SCHEDULED @ {elapsed:.0f}h (health={min_h:.2f})"

    @property
    def total_interventions(self):
        return self._interventions


# ─────────────────────────────────────────────
#  Simulation Engine
# ─────────────────────────────────────────────

class SimulationEngine:
    def __init__(self, historian: Historian):
        self._historian = historian

    def run(self, config: SimulationConfig) -> pd.DataFrame:
        print(f"\n{'─'*60}")
        print(f"  🚀  {config.scenario_id}  —  {config.description}")
        print(f"      Duration: {config.total_hours:.0f}h  |  Step: {config.hours_per_step}h")
        print(f"{'─'*60}")

        random.seed(config.seed)
        engine      = LogicEngine(seed=config.seed)
        agent       = MaintenanceAgent() if config.enable_ai_agent else None
        sim_start   = datetime(2025, 1, 1)
        steps       = int(config.total_hours / config.hours_per_step)
        failure_log = {}

        for step in range(1, steps + 1):
            elapsed  = step * config.hours_per_step
            sim_time = sim_start + timedelta(hours=elapsed)

            drivers = config.profile_fn(elapsed, config.total_hours)

            # Apply pending maintenance from previous step's agent decision
            maint_action = ""
            if agent:
                maint_action = agent.apply_boost(drivers)
                if maint_action:
                    print(f"  🔧  {maint_action}")

            # Run Phase 1
            report = engine.step(drivers, elapsed)

            # Agent observes result and schedules next intervention if needed
            if agent:
                agent.observe(report, elapsed)

            # Track first failures
            for comp in report.components:
                if comp.operational_status == OperationalStatus.FAILED and comp.component_id not in failure_log:
                    failure_log[comp.component_id] = {
                        "hours": elapsed, "time": sim_time.isoformat(), "reason": comp.failure_reason
                    }

            self._historian.record(config.scenario_id, sim_time, elapsed, drivers, report, maint_action)

        # Summary
        print(f"\n  FAILURE ANALYSIS")
        if not failure_log:
            print("  ✅  No components failed.")
        else:
            for cid, info in failure_log.items():
                print(f"  ❌  {cid}  →  {info['hours']:.0f}h  —  {info['reason']}")
        if agent:
            print(f"  🤖  AI Agent interventions: {agent.total_interventions}")
        df = self._historian.query(config.scenario_id)
        print(f"  💾  {len(df)} records saved to historian")
        return df


# ─────────────────────────────────────────────
#  Environmental Profiles
# ─────────────────────────────────────────────

def profile_normal(elapsed, total):
    shock = round(random.uniform(0.3, 0.6), 2) if random.random() < 0.005 else 0.0
    return EnvironmentalDrivers(
        temperature_stress = 40 + min(elapsed * 0.03, 25),
        contamination      = min(0.05 + 0.001 * elapsed, 0.6),
        operational_load   = elapsed,
        maintenance_level  = max(0.2, 0.8 - elapsed * 0.0005),
        shock_magnitude    = shock,
    )

def profile_dirty_factory(elapsed, total):
    shock = round(random.uniform(0.4, 0.8), 2) if random.random() < 0.015 else 0.0
    return EnvironmentalDrivers(
        temperature_stress = 55 + 10 * math.sin(elapsed / 50),
        contamination      = min(0.3 + 0.002 * elapsed, 0.95),
        operational_load   = elapsed,
        maintenance_level  = 0.1,
        shock_magnitude    = shock,
    )

def profile_chaos(elapsed, total):
    shock = round(random.uniform(0.5, 1.0), 2) if random.random() < 0.03 else 0.0
    return EnvironmentalDrivers(
        temperature_stress = random.uniform(30, 85),
        contamination      = random.uniform(0.1, 0.9),
        operational_load   = elapsed,
        maintenance_level  = random.choice([0.0, 0.0, 0.0, 0.5, 1.0]),
        shock_magnitude    = shock,
    )


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def plot_all_scenarios(historian: Historian, ids: list):
    n = len(ids)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n), facecolor="#0d1117")
    if n == 1:
        axes = [axes]

    comps = [
        ("blade_health",  "Recoater Blade",  "#3498db"),
        ("nozzle_health", "Nozzle Plate",    "#9b59b6"),
        ("heater_health", "Heating Element", "#e67e22"),
    ]

    for r, sid in enumerate(ids):
        df = historian.query(sid)
        for c, (hcol, label, color) in enumerate(comps):
            ax = axes[r][c]
            ax.set_facecolor("#161b22")
            ax.plot(df["elapsed_hours"], df[hcol], color=color, lw=2)
            ax.axhspan(0.65, 1.0,  alpha=0.08, color="#2ecc71")
            ax.axhspan(0.35, 0.65, alpha=0.08, color="#f39c12")
            ax.axhspan(0.0,  0.35, alpha=0.08, color="#e74c3c")
            ax.axhline(0.65, color="#2ecc71", lw=0.5, ls="--", alpha=0.5)
            ax.axhline(0.35, color="#f39c12", lw=0.5, ls="--", alpha=0.5)

            mdf = df[df["maintenance_action"] != ""]
            for _, mr in mdf.iterrows():
                ax.axvline(mr["elapsed_hours"], color="#00d4ff", lw=1, alpha=0.6, ls=":")

            sdf = df[df["shock_magnitude"] > 0]
            ax.scatter(sdf["elapsed_hours"], sdf[hcol], color="#ff4757", s=25, zorder=5, marker="v")

            ax.set_title(f"{sid} — {label}", color="white", fontsize=10)
            ax.set_xlabel("Hours", color="#8b949e", fontsize=8)
            ax.set_ylabel("Health Index", color="#8b949e", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(colors="#8b949e", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")

            patches = [
                mpatches.Patch(color="#2ecc71", alpha=0.5, label="FUNCTIONAL"),
                mpatches.Patch(color="#f39c12", alpha=0.5, label="DEGRADED"),
                mpatches.Patch(color="#e74c3c", alpha=0.5, label="CRITICAL/FAILED"),
            ]
            if not mdf.empty:
                patches.append(mpatches.Patch(color="#00d4ff", alpha=0.6, label="AI Maintenance"))
            ax.legend(handles=patches, fontsize=6, loc="upper right",
                     facecolor="#0d1117", edgecolor="#30363d", labelcolor="white")

    plt.suptitle("HP Metal Jet S100 — Digital Twin Health Monitor",
                color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("health_timeseries.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print("  📈  Saved → health_timeseries.png")


def plot_comparison(historian: Historian, ids: list):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    cols   = ["blade_health", "nozzle_health", "heater_health"]
    labels = ["Recoater Blade", "Nozzle Plate", "Heating Element"]
    colors = ["#3498db", "#9b59b6", "#e67e22"]
    width  = 0.25

    for i, (col, label, color) in enumerate(zip(cols, labels, colors)):
        vals = [historian.query(sid)[col].iloc[-1] for sid in ids]
        xs   = [j + (i - 1) * width for j in range(len(ids))]
        bars = ax.bar(xs, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    ax.set_xticks(list(range(len(ids))))
    ax.set_xticklabels(ids, color="white")
    ax.set_ylabel("Final Health Index", color="#8b949e")
    ax.set_ylim(0, 1.15)
    ax.set_title("Final Component Health by Scenario", color="white", fontsize=12)
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig("scenario_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print("  📊  Saved → scenario_comparison.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HP Metal Jet S100 — Phase 2 Simulation Engine")
    print("=" * 60)

    if os.path.exists("historian.db"):
        os.remove("historian.db")

    historian = Historian()
    sim       = SimulationEngine(historian)

    scenarios = [
        SimulationConfig("NORMAL",        1200, 4, profile_normal,        42, "Baseline, AI agent ON",              True),
        SimulationConfig("DIRTY_FACTORY", 1200, 4, profile_dirty_factory, 42, "High contamination, AI agent ON",    True),
        SimulationConfig("CHAOS",         1200, 4, profile_chaos,         99, "Chaos engineering, no agent",        False),
    ]

    ids = []
    for cfg in scenarios:
        sim.run(cfg)
        ids.append(cfg.scenario_id)

    historian.export_csv()

    print("\n" + "=" * 60)
    print("  Generating visualizations...")
    print("=" * 60)
    plot_all_scenarios(historian, ids)
    plot_comparison(historian, ids)
    historian.close()

    print("\n✅  Phase 2 complete.")
    print("\n  Output files:")
    print("    historian.db            ← SQLite (queryable by Phase 3)")
    print("    historian.csv           ← Full telemetry export")
    print("    health_timeseries.png   ← Health curves per scenario")
    print("    scenario_comparison.png ← Final health comparison\n")
