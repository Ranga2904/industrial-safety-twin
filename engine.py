from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os
from typing import List, Optional
from openai import OpenAI

# ── Domain Models ─────────────────────────────────────────────────────────────

class ChemicalProperties(BaseModel):
    name:              str   = "Methanol"
    flash_point_c:     float = 11.0
    autoignition_c:    float = 385.0
    lel_percent:       float = 6.0
    uel_percent:       float = 36.5
    tlv_ppm:           float = 200.0
    idlh_ppm:          float = 6000.0
    toxic:             bool  = True
    flammable:         bool  = True

class EnvironmentalConditions(BaseModel):
    temperature_c:     float = 22.0
    humidity_percent:  float = 45.0
    wind_speed_ms:     float = 0.5
    ventilation:       str   = "adequate"   # adequate | poor | none

class WorkPermit(BaseModel):
    permit_id:         str
    permit_type:       str                  # hot_work | cold_work | confined_space
    issued_at:         datetime             = Field(default_factory=datetime.now)
    gas_test_required: bool                 = True
    ppe_required:      List[str]            = Field(default_factory=lambda: [
                                                "face_shield","chemical_gloves",
                                                "chemical_suit","scba"])
    active:            bool                 = True

class ProcessKnowledge(BaseModel):
    max_operating_pressure_psi: float = 50.0
    critical_valve_sequence:    List[str] = Field(
                                    default_factory=lambda: ['V1','V3'])
    isolation_valves:           List[str] = Field(
                                    default_factory=lambda: ['V1','V4'])
    purge_required_before_open: bool = True

# ── Safety Audit ──────────────────────────────────────────────────────────────

class SafetyAudit:
    def __init__(self):
        self.chemical    = ChemicalProperties()
        self.process     = ProcessKnowledge()
        self.violations: List[str] = []
        self.warnings:   List[str] = []
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"]
        )

    def _llm_cascade_check(self, perception: dict,
                            env: EnvironmentalConditions,
                            permit: Optional[WorkPermit]) -> List[str]:
        prompt = f"""You are a process safety engineer for a methanol facility.

Observed conditions:
- Valves OPEN: {perception.get('valves_open', [])}
- Valves CLOSED: {perception.get('valves_closed', [])}
- Pressure: {perception.get('pressure_psi')} PSI (max allowed: {self.process.max_operating_pressure_psi})
- Worker in zone: {perception.get('worker_present')}
- Maintenance tools present: {perception.get('tools_present')}
- Work permit active: {permit is not None}
- Ventilation: {env.ventilation}
- Chemical: {self.chemical.name} (flash point {self.chemical.flash_point_c}°C, LEL {self.chemical.lel_percent}%)

Think step by step:
1. Which combination of the above factors creates a CASCADING hazard?
2. What is the specific failure pathway?
3. What is the worst-case outcome?

Then output ONLY a list of cascade hazard strings, one per line, prefixed with CASCADE:."""

        resp = self._client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        raw = resp.choices[0].message.content
        if raw is None:
            print("\n[Warning] Nemotron returned an empty response.")
            return []
        print(f"\n[Nemotron CoT cascade reasoning]\n{raw}\n")

        cascades = []
        for line in raw.splitlines():
            if line.strip().startswith("CASCADE:"):
               val = line.replace("CASCADE:", "").strip()
               if val.upper() != "NONE" and val:
                   cascades.append(f"CASCADE HAZARD: {val}")
        return cascades



    def run(self, perception_result: dict,
            env: EnvironmentalConditions,
            permit: Optional[WorkPermit] = None) -> dict:

        self.violations = []
        self.warnings   = []

        # ── Rule 1: Worker + open isolation valve ────────────────────────────
        if perception_result.get('worker_present'):
            open_valves = perception_result.get('valves_open', [])
            open_isolation = [v for v in open_valves
                              if v in self.process.isolation_valves]
            if open_isolation:
                self.violations.append(
                    f"CRITICAL: Worker in zone with open isolation "
                    f"valve(s) {open_isolation}. Methanol release risk."
                )

        # ── Rule 2: Over-pressure ─────────────────────────────────────────────
        psi = perception_result.get('pressure_psi', 0)
        if psi > self.process.max_operating_pressure_psi:
            self.violations.append(
                f"CRITICAL: Pressure {psi} PSI exceeds max "
                f"{self.process.max_operating_pressure_psi} PSI."
            )
        elif psi > self.process.max_operating_pressure_psi * 0.8:
            self.warnings.append(
                f"WARNING: Pressure {psi} PSI approaching limit."
            )

        # ── Rule 3: Tools + no permit ─────────────────────────────────────────
        if perception_result.get('tools_present') and permit is None:
            self.violations.append(
                "CRITICAL: Maintenance tools present but no work permit issued."
            )

        # ── Rule 4: Poor ventilation + open valves ────────────────────────────
        if env.ventilation != "adequate" and perception_result.get('valves_open'):
            self.violations.append(
                f"CRITICAL: Open valves with {env.ventilation} ventilation. "
                f"Methanol vapour accumulation risk (LEL {self.chemical.lel_percent}%)."
            )

        # ── Rule 5: Cascade — worker + tools + no permit ──────────────────────
        # Rule 5 — LLM cascade reasoning
        cascade_violations = self._llm_cascade_check(perception_result, env, permit)
        self.violations.extend(cascade_violations)


        return {
            'violations': self.violations,
            'warnings':   self.warnings,
            'safe':       len(self.violations)
       }
