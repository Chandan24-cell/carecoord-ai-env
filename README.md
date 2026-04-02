<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a73e8,100:00bfa5&height=200&section=header&text=CareCoordEnv&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Healthcare+Operations+Environment+for+AI+Agents&descAlignY=58&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Based-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenEnv-Compatible-4CAF50?style=for-the-badge&logo=checkmarx&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-BSD--style-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tasks-3%20Difficulty%20Levels-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Hackathon-Meta%20PyTorch%20OpenEnv-9b59b6?style=flat-square&logo=meta"/>
  <img src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
</p>

<p align="center">
  <strong>A realistic healthcare operations environment for training and evaluating AI agents on care-coordination tasks.</strong>
</p>

<p align="center">
  <a href="#-setup--installation">🚀 Quick Start</a> ·
  <a href="#-action-space">📖 Docs</a> ·
  <a href="#-running-tests">🧪 Run Tests</a> ·
  <a href="#-hugging-face-spaces-deployment">🌐 Deploy</a> ·
  <a href="#-acknowledgements">🤝 Contribute</a>
</p>

---

## 🎥 Demo

> *GIF demo coming soon — run `python inference.py` from `care_coord_env/` to see the agent in action.*

<!-- Replace the placeholder below with your actual demo GIF -->
```text
┌─────────────────────────────────────────────────────────────┐
│                    LIVE AGENT DEMO                          │
│                                                             │
│  [START] task=easy_referral_routing  model=gpt-4o          │
│  [STEP 1] review_case(referral_details)   reward=+0.10     │
│  [STEP 2] route_referral(prov_derm_north) reward=+0.25     │
│  [STEP 3] schedule_visit(slot_north_3d)   reward=+0.25     │
│  [STEP 4] finalize_case(scheduled)        reward=+0.15     │
│  [END]  success=True  steps=7  score=1.00 ✅               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        CareCoordEnv System                          │
│                                                                     │
│   ┌──────────────┐        HTTP / WebSocket        ┌─────────────┐   │
│   │              │ ──── /reset ─────────────────► │             │   │
│   │   AI Agent   │ ──── /step  ─────────────────► │   FastAPI   │   │
│   │  (LLM-based) │ ◄─── Observation + Reward ──── │   Server    │   │
│   │              │ ──── /state ─────────────────► │  (app.py)   │   │
│   └──────────────┘                                └──────┬──────┘   │
│                                                          │           │
│          ┌───────────────────────────────────────────────┤           │
│          │                                               │           │
│          ▼                                               ▼           │
│   ┌─────────────┐   score delta   ┌─────────────────────────────┐   │
│   │ graders.py  │ ◄────────────── │ care_coord_env_environment  │   │
│   │(Deterministic│                │       (Main Engine)         │   │
│   │  Scoring)   │                 │  State ▸ Action ▸ Reward    │   │
│   └─────────────┘                 └──────────────┬──────────────┘   │
│                                                  │                   │
│          ┌───────────────────┬───────────────────┤                   │
│          ▼                   ▼                   ▼                   │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐         │
│   │  models.py  │   │task_library  │   │    Pydantic      │         │
│   │(Typed Schema│   │ .py (Tasks)  │   │ Observation /    │         │
│   │  Pydantic)  │   │ Easy·Med·Hard│   │ Action / State   │         │
│   └─────────────┘   └──────────────┘   └──────────────────┘         │
│                                                                     │
│                   🐳 Dockerised · ☁️ HF Spaces · ✅ OpenEnv          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Why CareCoordEnv?

Care coordination is a real operational bottleneck in healthcare. A strong agent must combine **policy understanding**, **workflow reasoning**, **deadline awareness**, **safe escalation**, and **document/scheduling discipline**.

This environment focuses on **operational execution** rather than clinical diagnosis, which makes it safer, reproducible, and useful for agent research.

---

## 📦 Key Features

| Feature | Detail |
|:---|:---|
| 🏥 **Real-world domain** | Healthcare referral, prior authorization, and post-discharge follow-up |
| ✅ **Full OpenEnv compliance** | Typed `Observation`, `Action`, and `State` models with FastAPI runtime |
| 📶 **3 difficulty tiers** | Easy → Medium → Hard with distinct workflows |
| 🎯 **Deterministic graders** | Scores always stay in `[0.0, 1.0]` and remain fully auditable |
| 💰 **Dense reward shaping** | `reward = score_after - score_before - penalty` on every step |
| 🤖 **Baseline inference script** | OpenAI-client-based, env-driven, and log-friendly |
| 🐳 **Containerised** | Dockerfile included, ready for local runs and Spaces deployment |
| 📚 **Complete documentation** | Action space, observation space, setup, and usage examples |

---

## 🧩 Task Catalog

| Difficulty | Task ID | Agent Objective |
|:---:|:---|:---|
| 🟢 **Easy** | `easy_referral_routing` | Route a dermatology referral to the correct in-network provider and book the earliest acceptable slot. |
| 🟡 **Medium** | `medium_prior_auth_recovery` | Recover a denied MRI prior authorization: identify the missing document, request it, resubmit, and close as authorised. |
| 🔴 **Hard** | `hard_post_discharge_followup` | Escalate worsening heart-failure symptoms, route to the cardiology clinic, book follow-up within 7 days, arrange accessible covered transport, and finalise correctly. |

Each task has a clear objective, a bounded action set, and a deterministic grader that computes a score between `0.0` and `1.0`.

---

## 🎮 Action Space

The environment uses a single typed action model, `CareCoordAction`. Depending on the task, only a subset of action types is allowed at any step, and the current observation makes that explicit.

### Action Types

| Action | Description |
|:---|:---|
| `review_case` | Examine a case section such as `"referral_details"` or `"insurance_rules"` |
| `route_referral` | Select a provider by `provider_id` for a given `specialty` |
| `request_document` | Request a missing document by `document_id` |
| `submit_authorization` | Resubmit prior authorization after gathering requirements |
| `schedule_visit` | Book a slot by `slot_id` for a given `provider_id` |
| `arrange_transport` | Choose a transport option by `transport_id` |
| `escalate_case` | Escalate symptoms or unresolved issues with a short `summary` |
| `finalize_case` | Complete the task with a `resolution_code` and `summary` |

### Example Action

```python
from care_coord_env import CareCoordAction

action = CareCoordAction(
    action_type="route_referral",
    provider_id="prov_derm_north",
    specialty="Dermatology",
)
```

> See [`models.py`](./care_coord_env/models.py) for the full typed schema.

---

## 👁️ Observation Space

`CareCoordObservation` provides the agent with the state it needs for the next decision.

<details>
<summary><b>Click to expand field reference</b></summary>

| Field | Description |
|:---|:---|
| `task_id`, `task_title`, `task_type`, `difficulty` | Task metadata |
| `objective`, `objective_checklist` | Human-readable goal and checklist |
| `patient_name`, `patient_age`, `patient_brief` | Patient context relevant to operations |
| `allowed_action_types` | Which actions are currently permitted |
| `available_sections`, `visible_sections`, `visible_section_content` | Reviewable sections and the content already revealed |
| `provider_options` | Available providers with typed attributes |
| `available_slots` | Appointment slots currently visible |
| `documents` | Documents and their current states |
| `authorization_status` | Current prior-auth state |
| `transport_options` | Accessible transport choices |
| `blockers` | Obstacles that still need resolution |
| `progress`, `current_score`, `score_breakdown` | Structured progress and real-time grading signal |
| `last_action_feedback`, `reward_rationale` | Immediate feedback for the last step |

</details>

---

## 🧠 State Space (Internal)

`CareCoordState` tracks the underlying workflow state across an episode, including:

- reviewed sections
- selected provider and scheduled slot
- requested and received documents
- authorization status
- transport arrangement
- escalation completion
- final resolution code
- cumulative reward and current score

> Exposed via the `/state` endpoint for debugging and analysis.

---

## 🎁 Reward Design

Rewards are dense and shaped using deterministic grader deltas:

```text
1. Compute task score BEFORE the action
2. Apply the action
3. Compute task score AFTER the action
4. reward = score_after - score_before - penalty
```

- ✅ Useful actions raise the score and produce positive reward
- ❌ Invalid or redundant actions incur a small penalty
- 🏁 Final episode score is the grader's score in `[0.0, 1.0]`

This gives the agent meaningful feedback at each step instead of only at episode end.

---

## 📊 Graders & Reference Scores

Each task has a deterministic, rule-based grader in [`graders.py`](./care_coord_env/graders.py):

- ✔️ No randomness, so results are reproducible and auditable
- ✔️ Returns a score in `[0.0, 1.0]` with a detailed breakdown
- ✔️ Rewards partial progress, not just all-or-nothing completion

| Task | Max Score | Happy-Path Score |
|:---|:---:|:---:|
| `easy_referral_routing` | **1.00** | **1.00** |
| `medium_prior_auth_recovery` | **1.00** | **1.00** |
| `hard_post_discharge_followup` | **1.00** | **1.00** |

> A strong model with the right prompts should be able to reach near-perfect scores on all three tasks.

---

## 🚀 Setup & Installation

### Prerequisites

- Python **3.10+**
- **Docker** for containerised execution
- OpenEnv CLI: `pip install openenv-core`

### Local Development

```bash
# 1. Enter the environment project
cd care_coord_env

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install in editable mode
pip install -e .
```

---

## 🔐 Environment Variables

```bash
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your_openai_key"     # optional if HF_TOKEN is used as the API key
export API_BASE_URL="https://api.openai.com/v1"
export HF_TOKEN="your_huggingface_token"    # currently required by inference.py
```

> ⚠️ Never commit real keys. Keep local secrets out of git.

---

## 🖥️ Running the Server

From inside `care_coord_env/`:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Server available at: `http://localhost:8000`

### Client Example

```python
from care_coord_env import CareCoordAction, CareCoordEnv

with CareCoordEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy_referral_routing")
    print(result.observation.task_title)

    action = CareCoordAction(action_type="review_case", section="referral_details")
    result = env.step(action)
    print(f"Reward: {result.reward}")
    print(f"Feedback: {result.observation.last_action_feedback}")
```

---

## 🧪 Running Tests

From inside `care_coord_env/`:

```bash
python -m unittest discover -s tests
```

These tests cover environment flows, state transitions, and inference smoke behavior.

---

## 🤖 Baseline Inference Script

From inside `care_coord_env/`:

```bash
python inference.py
```

**Expected log format:**

```text
[START] task=easy_referral_routing env=care_coord_env model=gpt-4o
[STEP] step=1 action={"action_type":"review_case","section":"referral_details"} reward=0.1000 done=false error=null
...
[END] success=true steps=7 score=1.0000 rewards=[...]
```

> If required environment variables are missing, the script exits with a clear error message.

---

## 🐳 Docker

From inside `care_coord_env/`:

```bash
# Build
docker build -t care_coord_env-env:latest .

# Run
docker run -p 8000:8000 care_coord_env-env:latest
```

---

## ✅ OpenEnv Validation

From inside `care_coord_env/`:

```bash
openenv validate .
```

Checks include `openenv.yaml`, typed models, and the environment API surface.

---

## 🌐 Hugging Face Spaces Deployment

From inside `care_coord_env/`:

```bash
openenv push
```

After deployment, the environment exposes endpoints such as:

| Endpoint | Purpose |
|:---:|:---|
| `/reset` | Start a new episode |
| `/step` | Submit an action |
| `/state` | Inspect internal state |
| `/schema` | View typed models |
| `/health` | Liveness check |
| `/ws` | WebSocket interface |
| `/web` | Browser UI |

> The Space should be publicly accessible for external evaluation.

---

## 📁 Project Structure

```text
.
├── README.md
└── care_coord_env/
    ├── .dockerignore
    ├── .env.example
    ├── __init__.py
    ├── client.py
    ├── docs/
    │   └── project_blueprint.md
    ├── Dockerfile
    ├── graders.py
    ├── inference.py
    ├── models.py
    ├── openenv.yaml
    ├── pyproject.toml
    ├── server/
    │   ├── app.py
    │   ├── care_coord_env_environment.py
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── task_library.py
    ├── tests/
    │   ├── test_environment_flows.py
    │   └── test_inference_smoke.py
    └── uv.lock
```

---

## 🛠️ Troubleshooting

<details>
<summary><b>Click to expand common issues & fixes</b></summary>

| Problem | Fix |
|:---|:---|
| `ModuleNotFoundError: No module named 'care_coord_env'` | Activate the venv and run `pip install -e .` from `care_coord_env/` |
| `openenv: command not found` | Run `pip install openenv-core` inside the active venv |
| Docker build fails | Ensure Docker Desktop is running and network access is available |
| Port 8000 already in use | Stop the existing process or run with a different port |
| `inference.py` exits before running | Set `MODEL_NAME` and `HF_TOKEN`, plus any API credentials you need |
| `openenv validate` fails | Confirm `openenv.yaml` exists and typed models import cleanly |

</details>

---

## ✅ Submission Checklist

- [x] `README.md` is complete
- [ ] `python -m unittest discover -s tests` passes
- [ ] `docker build -t care_coord_env-env:latest .` succeeds
- [ ] `openenv validate .` passes
- [ ] `openenv push` deploys successfully
- [ ] `inference.py` runs to completion with valid credentials
- [ ] No hardcoded secrets remain in the repository

---

## 📜 License

Source headers in this project reference a **BSD-style license**. Add or verify the repository `LICENSE` file before publishing externally.

---

## 🙏 Acknowledgements

<p align="center">
  Built for the <strong>Meta PyTorch OpenEnv Hackathon × SST | India AI Hackathon 2026</strong>
</p>

<p align="center">
  Special thanks to the <strong>OpenEnv team</strong>, <strong>Hugging Face</strong>, and <strong>PyTorch</strong> for the framework and support.
</p>

<p align="center">
  <em>Questions or feedback? Open an issue or reach out via the Hugging Face Space community tab.</em>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00bfa5,100:1a73e8&height=120&section=footer" width="100%"/>
</p>
