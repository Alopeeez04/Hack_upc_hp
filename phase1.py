"""
Phase 1 — Logic Engine: HP Metal Jet S100 Digital Twin
=======================================================
Single-file version. Run directly with:
    python phase1.py

Dependencies:
    pip install scikit-learn numpy
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# ─────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────

class OperationalStatus(str, Enum):
    FUNCTIONAL = "FUNCTIONAL"
    DEGRADED   = "DEGRADED"
    CRITICAL   = "CRITICAL"
    FAILED     = "FAILED"


# ─────────────────────────────────────────────
#  Data Contract Structures
# ─────────────────────────────────────────────

@dataclass
class EnvironmentalDrivers:
    """Input vector passed at every time step."""
    temperature_stress: float   # Ambient temperature in °C  (e.g. 20–80)
    contamination:      float   # 0.0 (clean) → 1.0 (heavily contaminated)
    operational_load:   float   # Cumulative print-hours since last maintenance
    maintenance_level:  float   # 0.0 (neglected) → 1.0 (perfect upkeep)
    shock_magnitude:    float = 0.0   # 0.0 = no shock, 1.0 = catastrophic


@dataclass
class ComponentState:
    """Output produced by the Logic Engine for a single component."""
    component_id:       str
    subsystem:          str
    health_index:       float
    operational_status: OperationalStatus
    metrics:            dict
    failure_reason:     Optional[str] = None


@dataclass
class StateReport:
    """Full output of the Logic Engine for one time step."""
    components: list[ComponentState]
    cascade_events: list[str] = field(default_factory=list)

    def get(self, component_id: str) -> Optional[ComponentState]:
        return next((c for c in self.components if c.component_id == component_id), None)


# ─────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────

def _status_from_health(h: float) -> OperationalStatus:
    if h > 0.65:  return OperationalStatus.FUNCTIONAL
    if h > 0.35:  return OperationalStatus.DEGRADED
    if h > 0.0:   return OperationalStatus.CRITICAL
    return OperationalStatus.FAILED


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _weibull_hazard(t: float, shape: float, scale: float) -> float:
    """Weibull hazard rate h(t) = (k/λ)(t/λ)^(k-1)."""
    if t <= 0:
        return 0.0
    return (shape / scale) * ((t / scale) ** (shape - 1))


def _exponential_decay(health: float, rate: float) -> float:
    """One-step exponential decay: H(t+1) = H(t) * e^(-rate)."""
    return health * math.exp(-rate)


# ─────────────────────────────────────────────
#  Subsystem A — Recoater Blade
#  Model: Weibull Abrasive Wear
# ─────────────────────────────────────────────

class RecoaterBladeModel:
    WEIBULL_SHAPE = 2.5
    WEIBULL_SCALE = 500.0

    def __init__(self):
        self._health:    float = 1.0
        self._thickness: float = 1.0

    def compute(
        self,
        drivers: EnvironmentalDrivers,
        elapsed_hours: float,
        external_contamination_boost: float = 0.0,
    ) -> ComponentState:

        hazard = _weibull_hazard(elapsed_hours, self.WEIBULL_SHAPE, self.WEIBULL_SCALE)
        contamination_factor = 1.0 + 3.0 * drivers.contamination + 2.0 * external_contamination_boost
        maintenance_protection = 1.0 - 0.6 * drivers.maintenance_level
        wear_delta = hazard * contamination_factor * maintenance_protection

        shock_damage = 0.15 * drivers.shock_magnitude * random.uniform(0.8, 1.2) if drivers.shock_magnitude > 0 else 0.0

        self._health    = _clamp(self._health - wear_delta - shock_damage)
        self._thickness = _clamp(self._thickness - wear_delta * 0.5)

        status = _status_from_health(self._health)
        reason = "Blade worn below minimum thickness — powder layer uniformity lost." if status == OperationalStatus.FAILED else None

        return ComponentState(
            component_id       = "recoater_blade",
            subsystem          = "Recoating System",
            health_index       = round(self._health, 4),
            operational_status = status,
            metrics            = {
                "blade_thickness_norm": round(self._thickness, 4),
                "wear_delta_step":      round(wear_delta, 6),
                "shock_damage_step":    round(shock_damage, 4),
            },
            failure_reason = reason,
        )

    @property
    def health(self) -> float:
        return self._health


# ─────────────────────────────────────────────
#  Subsystem B — Nozzle Plate
#  Model: Exponential Decay + Probabilistic Clogging
# ─────────────────────────────────────────────

class NozzlePlateModel:
    BASE_DECAY_RATE  = 0.0008
    CLOG_SENSITIVITY = 0.6

    def __init__(self, seed: int = 42):
        self._health:        float = 1.0
        self._clog_fraction: float = 0.0
        self._rng = random.Random(seed)

    def compute(self, drivers: EnvironmentalDrivers, blade_health: float) -> ComponentState:

        # Thermal fatigue (exponential decay)
        temp_factor = max(0, (drivers.temperature_stress - 50) / 30) ** 1.5
        decay_rate  = self.BASE_DECAY_RATE * (1 + 2.5 * temp_factor)
        decay_rate *= (1.0 - 0.5 * drivers.maintenance_level)
        self._health = _exponential_decay(self._health, decay_rate)

        # Clogging — cascade from blade degradation
        cascade_contamination   = (1.0 - blade_health) * 0.4
        effective_contamination = _clamp(drivers.contamination + cascade_contamination)
        clog_prob = self.CLOG_SENSITIVITY * effective_contamination * (1 - drivers.maintenance_level)
        if self._rng.random() < clog_prob * 0.1:
            new_clog = self._rng.uniform(0.02, 0.08)
            self._clog_fraction = _clamp(self._clog_fraction + new_clog)
            self._health = _clamp(self._health - new_clog * 1.5)

        if drivers.shock_magnitude > 0:
            self._health = _clamp(self._health - 0.1 * drivers.shock_magnitude)

        status = _status_from_health(self._health)
        if status == OperationalStatus.FAILED:
            reason = "Nozzle plate fully clogged." if self._clog_fraction > 0.5 else "Thermal fatigue fracture."
        else:
            reason = None

        return ComponentState(
            component_id       = "nozzle_plate",
            subsystem          = "Printhead Array",
            health_index       = round(self._health, 4),
            operational_status = status,
            metrics            = {
                "clog_fraction":      round(self._clog_fraction, 4),
                "thermal_decay_rate": round(decay_rate, 6),
                "temp_stress_C":      drivers.temperature_stress,
            },
            failure_reason = reason,
        )

    @property
    def health(self) -> float:
        return self._health

    @property
    def contamination_output(self) -> float:
        return (1.0 - self._health) * 0.3


# ─────────────────────────────────────────────
#  Subsystem C — Heating Elements
#  Model: ML (Gradient Boosting Regressor)
# ─────────────────────────────────────────────

class HeatingElementMLModel:
    def __init__(self, seed: int = 0):
        self._health:     float = 1.0
        self._resistance: float = 1.0
        self._model = self._train_model(seed)

    @staticmethod
    def _generate_training_data(n: int = 4000, seed: int = 0):
        rng = np.random.default_rng(seed)
        temp     = rng.uniform(20, 85,  n)
        contam   = rng.uniform(0,  1,   n)
        load     = rng.uniform(0,  800, n)
        maint    = rng.uniform(0,  1,   n)
        nozzle_h = rng.uniform(0,  1,   n)

        base_wear     = 0.0003 + 0.0005 * (load / 800) ** 1.8
        temp_factor   = np.clip((temp - 40) / 40, 0, 1) ** 2
        cascade_extra = 0.0004 * (1 - nozzle_h)
        maint_protect = 1.0 - 0.55 * maint
        noise         = rng.normal(0, 0.00005, n)

        delta = (base_wear + 0.001 * temp_factor + cascade_extra) * maint_protect + noise
        delta = np.clip(delta, 0, 0.05)
        X = np.column_stack([temp, contam, load, maint, nozzle_h])
        return X, delta

    def _train_model(self, seed: int) -> GradientBoostingRegressor:
        X, y = self._generate_training_data(seed=seed)
        model = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=seed)
        model.fit(X, y)
        return model

    def compute(self, drivers: EnvironmentalDrivers, nozzle_health: float) -> ComponentState:
        features = np.array([[
            drivers.temperature_stress,
            drivers.contamination,
            drivers.operational_load,
            drivers.maintenance_level,
            nozzle_health,
        ]])
        predicted_delta = max(0.0, float(self._model.predict(features)[0]))

        if drivers.shock_magnitude > 0:
            predicted_delta += 0.05 * drivers.shock_magnitude

        self._health = _clamp(self._health - predicted_delta)
        self._resistance = _clamp(1.0 + (1.0 - self._health) * 0.8, lo=1.0, hi=2.0)

        status = _status_from_health(self._health)
        reason = "Heating element resistance exceeded safe limit." if status == OperationalStatus.FAILED else None

        return ComponentState(
            component_id       = "heating_element",
            subsystem          = "Thermal Control",
            health_index       = round(self._health, 4),
            operational_status = status,
            metrics            = {
                "normalised_resistance": round(self._resistance, 4),
                "ml_predicted_delta":    round(predicted_delta, 6),
                "effective_temp_C":      drivers.temperature_stress,
            },
            failure_reason = reason,
        )

    @property
    def health(self) -> float:
        return self._health


# ─────────────────────────────────────────────
#  Logic Engine — Orchestrator
# ─────────────────────────────────────────────

class LogicEngine:
    """
    Main entry point. Phase 2 calls engine.step() on every tick.
    """

    def __init__(self, seed: int = 42):
        self._blade  = RecoaterBladeModel()
        self._nozzle = NozzlePlateModel(seed=seed)
        self._heater = HeatingElementMLModel(seed=seed)

    def step(self, drivers: EnvironmentalDrivers, elapsed_hours: float) -> StateReport:
        cascade_events = []

        # 1. Recoater Blade (cascade IN: nozzle contamination)
        nozzle_boost = self._nozzle.contamination_output
        blade_state  = self._blade.compute(drivers, elapsed_hours, external_contamination_boost=nozzle_boost)
        if nozzle_boost > 0.05:
            cascade_events.append(f"[CASCADE] Nozzle contamination ({nozzle_boost:.3f}) accelerating Blade wear.")

        # 2. Nozzle Plate (cascade IN: blade health)
        nozzle_state = self._nozzle.compute(drivers, blade_health=self._blade.health)
        if self._blade.health < 0.4:
            cascade_events.append(f"[CASCADE] Degraded blade (health={self._blade.health:.2f}) raising clog risk.")

        # 3. Heating Elements ML (cascade IN: nozzle health)
        heater_state = self._heater.compute(drivers, nozzle_health=self._nozzle.health)
        if self._nozzle.health < 0.4:
            cascade_events.append(f"[CASCADE] Poor nozzle (health={self._nozzle.health:.2f}) overloading Heaters.")

        return StateReport(
            components     = [blade_state, nozzle_state, heater_state],
            cascade_events = cascade_events,
        )

    def reset(self, seed: int = 42):
        self.__init__(seed=seed)


# ─────────────────────────────────────────────
#  Demo — run this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    engine = LogicEngine(seed=42)

    STEPS          = 300
    HOURS_PER_STEP = 4

    print("=" * 70)
    print("  HP Metal Jet S100 — Phase 1 Logic Engine Demo")
    print("=" * 70)
    print(f"{'Step':>5}  {'Hours':>6}  {'Blade':>10}  {'Nozzle':>10}  {'Heater':>10}  Event")
    print("-" * 70)

    failure_log = {}

    for step in range(1, STEPS + 1):
        elapsed     = step * HOURS_PER_STEP
        temperature = 40 + min(step * 0.1, 30) + (15 if step % 50 == 0 else 0)
        contamination = min(0.05 + 0.002 * step, 0.9)
        load  = elapsed
        maint = max(0.1, 0.9 - step * 0.002)
        shock = round(random.uniform(0.3, 0.7), 2) if step % 80 == 0 else 0.0

        drivers = EnvironmentalDrivers(
            temperature_stress = temperature,
            contamination      = contamination,
            operational_load   = load,
            maintenance_level  = maint,
            shock_magnitude    = shock,
        )

        report = engine.step(drivers, elapsed_hours=elapsed)

        for comp in report.components:
            if comp.operational_status == OperationalStatus.FAILED and comp.component_id not in failure_log:
                failure_log[comp.component_id] = {"step": step, "hours": elapsed, "reason": comp.failure_reason}

        if step % 10 == 0 or shock > 0:
            blade  = report.get("recoater_blade")
            nozzle = report.get("nozzle_plate")
            heater = report.get("heating_element")

            def fmt(c):
                return f"{c.health_index:.3f}/{c.operational_status.value[:4]}"

            event = "⚡ SHOCK!" if shock > 0 else ("🔗 CASCADE" if report.cascade_events else "")
            print(f"{step:>5}  {elapsed:>6.0f}h  {fmt(blade):>10}  {fmt(nozzle):>10}  {fmt(heater):>10}  {event}")

    print("\n" + "=" * 70)
    print("  FAILURE ANALYSIS")
    print("=" * 70)
    if not failure_log:
        print("  ✅  No components reached FAILED state.")
    else:
        for cid, info in failure_log.items():
            print(f"\n  ❌  {cid}")
            print(f"      Step  : {info['step']}")
            print(f"      Hours : {info['hours']} h")
            print(f"      Reason: {info['reason']}")

    print("\n" + "=" * 70)
    print("  FINAL COMPONENT METRICS")
    print("=" * 70)
    for comp in report.components:
        print(f"\n  [{comp.subsystem}] {comp.component_id}")
        print(f"    Health : {comp.health_index:.4f}  |  Status: {comp.operational_status.value}")
        for k, v in comp.metrics.items():
            print(f"    {k}: {v}")

    print("\n✅  Phase 1 complete.\n")
