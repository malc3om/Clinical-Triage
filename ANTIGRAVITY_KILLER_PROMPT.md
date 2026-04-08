# ANTIGRAVITY PROMPT — ClinicalTriageEnv
## OpenEnv Hackathon x SST | Team Unfazed | Deadline: 8 Apr 11:59 PM

---

## SYSTEM CONTEXT

You are an expert AI Environment Architect and Python Developer working with Team "Unfazed"
(Sanskar Singh Rawat — Team Lead, Neel Kumawat, Vaibhav Neema) to win the
Meta × Scaler OpenEnv Hackathon Round 1.

You combine two disciplines: the structured rigour of a senior backend engineer and the
creative instinct of an out-of-the-box systems designer. You write production-quality,
test-passing code from the first attempt. You think in systems, not files.

---

## WHAT WE ARE BUILDING

We are building **`ClinicalTriageEnv`** — a complete, fully-spec-compliant OpenEnv
environment that simulates real Emergency Department (ED) triage and patient disposition
decisions, the kind that happen thousands of times a day in every hospital worldwide.

An AI agent steps into the role of an ED triage clinician. It receives a patient's presenting
complaint, vital signs, medical history, and arriving lab/imaging results — and must decide
what diagnostics to order, what ESI (Emergency Severity Index) level to assign, and how to
dispose of the patient (admit, discharge, transfer, or activate a specialist pathway).

This environment does NOT exist in the OpenEnv ecosystem. It fills a real gap:
the RL/agent community currently has no standardized benchmark for clinical decision-making
under uncertainty, partial information, and resource constraints. Judges from Meta and
Hugging Face will immediately recognize its value.

**Why it scores 26–30 on real-world utility (the top 30% category):**
- Emergency triage is one of the highest-stakes sequential decision tasks humans perform.
- Wrong triage leads to real harm: delayed MI treatment = myocardial death. Over-triage = ED
  gridlock, reduced bed availability, downstream patient harm.
- Research teams training clinical AI agents currently have no standardized environment.
  This closes that gap.
- The environment runs entirely on synthetic patient data — no HIPAA concerns, fully open.

---

## TECHNICAL ENVIRONMENT SPEC (follow exactly)

### Core Interface
Implement the full OpenEnv interface:
- `step(action: TriageAction) -> StepResult` returns observation, reward, done, info
- `reset(task_id: str = None) -> TriageObservation` returns initial observation
- `state() -> EnvState` returns current environment state
- `openenv.yaml` with all required metadata fields
- Passes `openenv validate` without errors

### Pydantic Models (typed, strict)

```python
from pydantic import BaseModel
from typing import Optional, List, Literal
from enum import Enum

class VitalSigns(BaseModel):
    heart_rate: int           # bpm
    systolic_bp: int          # mmHg
    diastolic_bp: int         # mmHg
    respiratory_rate: int     # breaths/min
    spo2: float               # percentage 0-100
    temperature: float        # Celsius
    gcs: int                  # Glasgow Coma Scale 3-15

class LabResult(BaseModel):
    name: str                 # e.g. "troponin_I", "d_dimer", "cbc_wbc"
    value: float
    unit: str
    reference_range: str
    critical: bool            # True if critically abnormal

class PatientState(BaseModel):
    patient_id: str
    age: int
    sex: Literal["M", "F"]
    chief_complaint: str
    onset_minutes: int        # How long ago symptoms started
    vitals: VitalSigns
    medical_history: List[str]
    current_medications: List[str]
    available_labs: List[LabResult]       # Labs already resulted
    pending_labs: List[str]               # Labs ordered but not yet back
    imaging_available: List[str]          # e.g. ["EKG", "CXR"]
    pending_imaging: List[str]
    time_in_department_minutes: int
    resource_tokens_remaining: int        # Budget for ordering tests

class TriageObservation(BaseModel):
    task_id: str
    task_difficulty: Literal["easy", "medium", "hard"]
    step_number: int
    max_steps: int
    patients: List[PatientState]          # 1 patient for easy/medium, 5 for hard
    available_beds: int
    last_action_result: Optional[str]
    last_action_error: Optional[str]

class TriageAction(BaseModel):
    action_type: Literal[
        "order_diagnostic",     # Order a test
        "assign_esi_level",     # Assign ESI 1-5
        "activate_pathway",     # e.g. "cath_lab", "stroke_code", "trauma"
        "disposition",          # admit/discharge/transfer
        "request_consult",      # Specialist consult
        "wait"                  # Deliberately wait for pending results
    ]
    patient_id: str
    parameter: str            # test name, ESI level, pathway type, etc.
    rationale: Optional[str]  # Agent's reasoning (for logging/explainability)

class TriageReward(BaseModel):
    total: float              # -1.0 to +1.0
    components: dict          # Breakdown by sub-reward source
    explanation: str

class StepResult(BaseModel):
    observation: TriageObservation
    reward: TriageReward
    done: bool
    info: dict

class EnvState(BaseModel):
    task_id: str
    step_number: int
    episode_history: List[dict]
    current_observation: TriageObservation
    cumulative_reward: float
```

---

## THE 3 TASKS — EXACT SPECS

### Task 1: `task_stemi_code` (Easy, ESI 1)
**Scenario:** 58-year-old male, sudden crushing chest pain radiating to left arm, onset 22 minutes
ago. Diaphoretic. HR 102, BP 88/60, SpO2 94%, GCS 15.
EKG already in hand showing ST-elevation in leads II, III, aVF.
Troponin ordered but not yet resulted.

**What the agent MUST do (in any order, within 8 steps):**
1. Assign ESI level 1
2. Activate "cath_lab" pathway
3. Order "aspirin_325mg" (or equivalent medication action)
4. Order IV access (or "resuscitation" action)

**Grader logic (deterministic, scores 0.0–1.0):**
```python
def grade_stemi(episode_history) -> float:
    score = 0.0
    actions = [step["action"] for step in episode_history]
    
    # ESI assigned correctly (0.25)
    esi_actions = [a for a in actions if a["action_type"] == "assign_esi_level"]
    if any(a["parameter"] == "1" for a in esi_actions):
        score += 0.25
    
    # Cath lab activated (0.30)
    pathway_actions = [a for a in actions if a["action_type"] == "activate_pathway"]
    if any("cath_lab" in a["parameter"].lower() for a in pathway_actions):
        score += 0.30
    
    # Correct disposition: admit ICU (0.25)
    disp_actions = [a for a in actions if a["action_type"] == "disposition"]
    if any("admit" in a["parameter"].lower() for a in disp_actions):
        score += 0.25
    
    # Time penalty: deduct 0.05 for every 2 steps over 4
    steps_taken = len(episode_history)
    if steps_taken > 4:
        score -= 0.05 * ((steps_taken - 4) // 2)
    
    # Penalize if agent ordered non-urgent tests before activating cath lab (-0.15)
    cath_step = next((i for i, s in enumerate(episode_history)
                      if s["action"].get("parameter","").lower() == "cath_lab"), None)
    if cath_step is not None and cath_step > 3:
        score -= 0.15
    
    return max(0.0, min(1.0, score))
```

**Episode terminates when:** correct disposition is given OR 15 steps reached.

---

### Task 2: `task_chest_pain_workup` (Medium, ESI 2-3)
**Scenario:** 44-year-old female, sharp pleuritic chest pain, worsened by inspiration,
onset 4 hours ago. Recent 6-hour flight 3 days ago. HR 98, BP 122/78, SpO2 96%, RR 20.
No EKG yet. No labs yet. No imaging yet. PMH: oral contraceptives.

**Differential the agent must navigate:** PE vs STEMI vs MSK vs anxiety.

**What the agent MUST do (order matters for partial credit):**
1. Order EKG (rule out STEMI first — correct clinical priority)
2. Order D-dimer OR Wells Score assessment
3. If D-dimer positive (it will be, per scenario): order CT-PA
4. Assign correct ESI (2 or 3 both accepted)
5. Correct disposition based on findings (CT-PA will show bilateral PE)

**Grader logic:**
```python
def grade_chest_workup(episode_history) -> float:
    score = 0.0
    actions_in_order = [s["action"] for s in episode_history]
    action_names = [a["parameter"].lower() for a in actions_in_order]
    
    # EKG ordered before CT-PA (correct sequence, +0.20)
    ekg_idx = next((i for i, n in enumerate(action_names) if "ekg" in n or "ecg" in n), None)
    ctpa_idx = next((i for i, n in enumerate(action_names) if "ct" in n and "pa" in n), None)
    if ekg_idx is not None:
        score += 0.20
        if ctpa_idx is not None and ekg_idx < ctpa_idx:
            score += 0.05  # Bonus for correct sequence
    
    # D-dimer ordered (+0.15)
    if any("d_dimer" in n or "d-dimer" in n for n in action_names):
        score += 0.15
    
    # CT-PA ordered (+0.20)
    if ctpa_idx is not None:
        score += 0.20
    
    # Correct disposition: admit (PE confirmed) (+0.25)
    disp = [a for a in actions_in_order if a["action_type"] == "disposition"]
    if any("admit" in a["parameter"].lower() for a in disp):
        score += 0.25
    
    # Penalize if agent immediately ordered expensive imaging without EKG (-0.15)
    if ctpa_idx is not None and (ekg_idx is None or ctpa_idx < ekg_idx):
        score -= 0.15
    
    # Penalize discharge without working up PE risk (-0.30)
    if any("discharge" in a["parameter"].lower() for a in disp):
        score -= 0.30
    
    # Resource efficiency penalty: deduct if > 6 diagnostics ordered total
    diagnostic_count = sum(1 for a in actions_in_order if a["action_type"] == "order_diagnostic")
    if diagnostic_count > 6:
        score -= 0.05 * (diagnostic_count - 6)
    
    return max(0.0, min(1.0, score))
```

---

### Task 3: `task_mci_surge` (Hard, Multi-Patient)
**Scenario:** Mass Casualty Incident. 5 patients arrive simultaneously.
Only 3 beds available. Agent must correctly triage all 5 within 20 steps.

**Patient pool (deterministic, seeded):**
- P1: 72yo M, unresponsive, HR 40, GCS 6 → ESI 1, immediate
- P2: 28yo F, broken arm, pain 6/10, vitals stable → ESI 3, delayed
- P3: 15yo M, anaphylaxis, BP 70/40, stridor → ESI 1, immediate
- P4: 60yo M, stable atrial fibrillation, known history → ESI 2, urgent
- P5: 35yo F, anxiety/hyperventilation, vitals normal → ESI 4, non-urgent

**What the agent must do:**
- Correctly assign ESI levels to all 5 patients
- Correctly identify P1 and P3 as highest priority for the 2 immediate beds
- Correctly triage P5 as lowest priority (or "treat and street")
- Manage 3-bed constraint without gridlock

**Grader logic:**
```python
def grade_mci(episode_history) -> float:
    score = 0.0
    
    esi_assignments = {}
    disposition_map = {}
    
    for step in episode_history:
        a = step["action"]
        pid = a.get("patient_id", "")
        if a["action_type"] == "assign_esi_level":
            esi_assignments[pid] = int(a["parameter"])
        if a["action_type"] == "disposition":
            disposition_map[pid] = a["parameter"].lower()
    
    # Correct ESI assignments (0.10 each, max 0.50)
    correct_esi = {"P1": 1, "P2": 3, "P3": 1, "P4": 2, "P5": 4}
    for pid, expected in correct_esi.items():
        assigned = esi_assignments.get(pid)
        if assigned == expected:
            score += 0.10
        elif assigned and abs(assigned - expected) == 1:
            score += 0.05  # Partial credit for near-miss
    
    # Correct bed allocation: P1 and P3 in immediate beds (+0.20)
    immediate_admits = [pid for pid, disp in disposition_map.items()
                        if "admit" in disp and pid in ["P1", "P3"]]
    score += 0.10 * len(immediate_admits)
    
    # Correctly not admitting P5 (+0.10)
    if disposition_map.get("P5") in ["discharge", "treat_and_street", "waiting_room"]:
        score += 0.10
    
    # Penalize if P5 gets a bed over P1 or P3 (-0.20 each)
    if "P5" in [pid for pid, disp in disposition_map.items() if "admit" in disp]:
        if "P1" not in [pid for pid, disp in disposition_map.items() if "admit" in disp]:
            score -= 0.20
        if "P3" not in [pid for pid, disp in disposition_map.items() if "admit" in disp]:
            score -= 0.20
    
    # Bonus for correct ordering: handles ESI 1 patients in first 4 steps (+0.10)
    first_4_pids = [s["action"].get("patient_id") for s in episode_history[:4]]
    if "P1" in first_4_pids and "P3" in first_4_pids:
        score += 0.10
    
    return max(0.0, min(1.0, score))
```

---

## REWARD FUNCTION SPEC (Dense, Multi-Component)

The reward signal must be dense across the full trajectory — NOT binary end-of-episode.

```python
def compute_reward(action, prev_state, new_state, task_id) -> TriageReward:
    components = {}
    
    # 1. Clinical correctness signal (+0.1 to +0.3 per step)
    #    Reward ordering a test that is clinically indicated given current vitals/history
    #    Penalize ordering irrelevant tests given the presentation
    
    # 2. Efficiency signal (-0.05 per unnecessary test ordered)
    #    Tests have a "resource_token" cost; running out reduces final score
    
    # 3. Time pressure signal (-0.02 per step for ESI-1 patients without disposition)
    #    ESI-1 patients bleed reward every step they're not being actively treated
    
    # 4. Sequence correctness (+0.05 bonus for following evidence-based protocols)
    #    E.g. EKG before troponin, D-dimer before CT-PA
    
    # 5. Safety guardrails (-0.5 flat penalty, terminal)
    #    Discharging an ESI-1 patient
    #    Waiting more than 3 steps without any action on an ESI-1 patient
    #    Infinite loop detection: same action 3 times in a row
    
    total = sum(components.values())
    total = max(-1.0, min(1.0, total))
    
    return TriageReward(
        total=total,
        components=components,
        explanation=f"Step reward breakdown: {components}"
    )
```

---

## PROJECT STRUCTURE (generate exactly this)

```
clinical-triage-env/
├── env/
│   ├── __init__.py
│   ├── clinical_triage_env.py     # Main environment class
│   ├── models.py                  # All Pydantic models above
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── task_stemi.py          # Task 1 scenario + grader
│   │   ├── task_chest_workup.py   # Task 2 scenario + grader
│   │   └── task_mci_surge.py      # Task 3 scenario + grader
│   ├── reward.py                  # Dense reward function
│   └── patient_generator.py      # Synthetic patient data (seeded, deterministic)
├── app.py                         # FastAPI server (HF Space entry point)
├── inference.py                   # MUST be in root directory
├── openenv.yaml                   # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
├── README.md
└── validate_submission.py         # Pre-submission checklist runner
```

---

## `openenv.yaml` (generate this exactly)

```yaml
name: clinical-triage-env
version: "1.0.0"
description: >
  A real-world Emergency Department triage environment where an AI agent
  must assess patients, order diagnostics, assign acuity levels, and make
  disposition decisions under time pressure and resource constraints.
author: "Team Unfazed"
tags:
  - openenv
  - healthcare
  - clinical-decision-making
  - triage
  - reinforcement-learning
tasks:
  - id: task_stemi_code
    difficulty: easy
    description: "Clear STEMI presentation. Agent must activate cath lab pathway within time window."
    max_steps: 15
  - id: task_chest_pain_workup
    difficulty: medium
    description: "Ambiguous chest pain. Agent must navigate differential diagnosis with ordered test sequencing."
    max_steps: 20
  - id: task_mci_surge
    difficulty: hard
    description: "Mass casualty: 5 simultaneous patients, 3 beds. Agent must correctly triage under scarcity."
    max_steps: 25
action_space:
  type: categorical
  description: "Pydantic TriageAction model with action_type, patient_id, parameter fields"
observation_space:
  type: structured
  description: "Pydantic TriageObservation with patient states, vitals, labs, resource tokens"
reward_range: [-1.0, 1.0]
baseline_scores:
  task_stemi_code: 0.72
  task_chest_pain_workup: 0.48
  task_mci_surge: 0.31
```

---

## `inference.py` (generate this in root directory)

```python
"""
Baseline inference script for ClinicalTriageEnv.
Uses OpenAI client with configurable model and endpoint.
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Must complete in < 20 minutes on vcpu=2, memory=8gb.
"""
import os
import json
from openai import OpenAI
from env.clinical_triage_env import ClinicalTriageEnv
from env.models import TriageAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

FALLBACK_ACTION = json.dumps({
    "action_type": "assign_esi_level",
    "patient_id": "P1",
    "parameter": "2",
    "rationale": "Fallback: defaulting to ESI 2 pending further assessment"
})

SYSTEM_PROMPT = """You are an experienced Emergency Department triage clinician.
You will receive a patient observation and must respond with a valid JSON action.
Think step by step about: (1) how urgent is this patient? (2) what information do you need?
(3) what is the most evidence-based next action?
Always respond with ONLY a valid JSON matching the TriageAction schema.
No markdown, no explanation — just the JSON object."""

def parse_model_action(response_text: str) -> dict:
    try:
        cleaned = response_text.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except Exception:
        return json.loads(FALLBACK_ACTION)

def run_task(env, task_id: str, client: OpenAI, max_steps: int = 20) -> float:
    observation = env.reset(task_id=task_id)
    history = []
    cumulative_reward = 0.0
    
    for step in range(max_steps):
        obs_str = observation.model_dump_json(indent=2)
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current observation:\n{obs_str}\n\nHistory:\n{json.dumps(history[-3:], indent=2)}\n\nWhat is your next action?"}
                ]
            )
            response_text = response.choices[0].message.content
        except Exception as exc:
            print(f"  Model request failed ({exc}). Using fallback action.")
            response_text = FALLBACK_ACTION
        
        action_dict = parse_model_action(response_text)
        print(f"  Step {step+1}: {action_dict.get('action_type')} → {action_dict.get('parameter')}")
        
        try:
            action = TriageAction(**action_dict)
            result = env.step(action)
        except Exception as exc:
            print(f"  Action parse failed: {exc}")
            break
        
        cumulative_reward += result.reward.total
        history.append({"step": step+1, "action": action_dict, "reward": result.reward.total})
        
        print(f"  Reward: {result.reward.total:+.2f} | Cumulative: {cumulative_reward:+.2f} | Done: {result.done}")
        
        if result.done:
            print(f"  Episode complete at step {step+1}.")
            break
        
        observation = result.observation
    
    return cumulative_reward

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    env = ClinicalTriageEnv()
    
    tasks = [
        ("task_stemi_code", 15),
        ("task_chest_pain_workup", 20),
        ("task_mci_surge", 25),
    ]
    
    scores = {}
    print("\n=== ClinicalTriageEnv Baseline Inference ===\n")
    
    for task_id, max_steps in tasks:
        print(f"\n--- Task: {task_id} ---")
        score = run_task(env, task_id, client, max_steps)
        grader_score = env.get_task_grader_score(task_id)
        scores[task_id] = grader_score
        print(f"  Grader score: {grader_score:.3f}")
    
    print("\n=== FINAL SCORES ===")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.3f}")
    print(f"\n  Average: {sum(scores.values()) / len(scores):.3f}")
    
    env.close()

if __name__ == "__main__":
    main()
```

---

## `Dockerfile` (generate this exactly, optimized for vcpu=2 / 8GB RAM)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## `app.py` — FastAPI server (HF Space compatible)

The FastAPI server must expose these endpoints:
- `GET /` → returns 200 with `{"status": "ok", "env": "clinical-triage-env"}`
- `POST /reset` → accepts `{"task_id": str}` → returns `TriageObservation` JSON
- `POST /step` → accepts `TriageAction` JSON → returns `StepResult` JSON
- `GET /state` → returns `EnvState` JSON
- `GET /tasks` → returns list of all task IDs with descriptions
- `POST /grade` → accepts `{"task_id": str, "episode_history": [...]}` → returns grader score

The server must maintain session state per request using a single global env instance
(sufficient for hackathon scope) and handle exceptions with clear error messages.

---

## `README.md` requirements (must include all of these sections)

1. **Environment Description & Motivation** — explain the real-world clinical problem,
   why this matters for the RL/agent community, and what makes it novel.
2. **Observation Space** — table with all fields, types, descriptions.
3. **Action Space** — table with action_type enum values, parameter examples.
4. **Reward Signal** — explain each component of the dense reward function.
5. **Task Descriptions** — for each task: scenario text, what the agent must do,
   expected difficulty, known baseline scores.
6. **Setup Instructions** — Docker and local Python instructions.
7. **Baseline Scores** — reproducible table from running inference.py.
8. **Quickstart** — 10-line code example showing reset + step loop.

---

## CRITICAL CONSTRAINTS (these cause disqualification if violated)

1. All 3 graders must return DIFFERENT scores for different agent policies.
   They must never return the same constant value. Test with a random agent first.

2. The environment must respond to `reset()` within 2 seconds (no network calls on init).
   All patient data is synthetic and generated from seeded random — no external API needed.

3. `inference.py` must be named exactly `inference.py` and placed in the project root.

4. Use `OpenAI` client for ALL LLM calls. Read credentials from env vars:
   `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`. Never hardcode keys.

5. Inference runtime must be under 20 minutes on 2 vCPU, 8GB RAM.
   Keep max_tokens ≤ 512 per call, use gpt-4o-mini as default model.

6. HF Space must return HTTP 200 on `GET /` and correctly handle `POST /reset`.

7. `openenv validate` must pass. Include `openenv.yaml` in project root.

8. Docker must build with `docker build -t clinical-triage-env .` and run with
   `docker run -p 7860:7860 clinical-triage-env` without errors.

---

## WHAT TO BUILD FIRST (execution order)

Generate these files in this order to allow incremental testing:

1. `env/models.py` — all Pydantic models (unblocks everything else)
2. `env/patient_generator.py` — seeded synthetic patient data for all 3 tasks
3. `env/tasks/task_stemi.py`, `task_chest_workup.py`, `task_mci_surge.py`
4. `env/reward.py` — dense reward function
5. `env/clinical_triage_env.py` — main env class wiring everything together
6. `app.py` — FastAPI server
7. `inference.py` — baseline script
8. `openenv.yaml`, `Dockerfile`, `requirements.txt`
9. `README.md`
10. `validate_submission.py` — pre-submission checklist

---

## BEGIN

Start by outputting `env/models.py` and `env/patient_generator.py` in full.
Then wait for my confirmation before proceeding to the task files.
I will review the schemas before you wire up the environment.

Team Unfazed. Deadline: 8 April 11:59 PM. Let's build something the judges remember.
