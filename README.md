---
title: CareCoordEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - healthcare
  - care-coordination
  - rl-environment
---

<div align="center">

# 🏥 CareCoordEnv

**A realistic healthcare operations environment for training and evaluating AI agents on care-coordination tasks.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green?style=for-the-badge)](https://github.com/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*Built for the Meta PyTorch OpenEnv Hackathon × SST | India AI Hackathon 2026*

<br />

<img src="https://via.placeholder.com/800x400.png?text=🎥+Insert+Your+Demo+GIF+Here" alt="CareCoordEnv Demo" width="800" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
<br />
<br />

</div>

CareCoordEnv simulates the daily back-office work of a care‑coordination assistant – routing specialist referrals, recovering denied prior authorizations, and arranging urgent post‑discharge follow‑up. It is built following the standard `step()` / `reset()` / `state()` API with fully typed Pydantic models.

> **💡 Why this matters:** Care coordination is a real operational bottleneck in healthcare. A strong agent must combine policy understanding, workflow reasoning, deadline awareness, safe escalation, and document/scheduling discipline. This environment focuses on **operational execution**, not clinical diagnosis, making it safe, reproducible, and valuable for agent research.

---

## 🧠 Architecture Overview

<div align="center">
  <img src="https://via.placeholder.com/800x400.png?text=🧠+Insert+Your+Architecture+Diagram+Here" alt="Architecture Diagram" width="800" style="border-radius: 8px;"/>
</div>

---

## 📦 Key Features

- ✅ **Real‑world domain:** Healthcare referral, prior authorization, and discharge follow‑up.
- ✅ **Full OpenEnv compliance:** Typed `Observation`, `Action`, `State` models; `openenv validate` passes.
- ✅ **3 Tasks with increasing difficulty:** Easy → Medium → Hard.
- ✅ **Deterministic programmatic graders:** Scores always in `[0.0, 1.0]`, zero randomness.
- ✅ **Dense reward shaping:** Reward = progress difference (`score_after - score_before - penalty`).
- ✅ **Baseline inference script:** Uses OpenAI client, reads env vars, produces reproducible logs.
- ✅ **Containerised:** Dockerfile included, ready for deployment to Hugging Face Spaces.
- ✅ **Comprehensive documentation:** Action/observation spaces, setup, and usage examples.

---

## 🧩 Task Catalog

| Difficulty | Task ID | What the agent must do |
| :--- | :--- | :--- |
| 🟢 **Easy** | `easy_referral_routing` | Route a dermatology referral to the correct in‑network provider and book the earliest acceptable slot. |
| 🟡 **Medium** | `medium_prior_auth_recovery` | Recover a denied MRI prior authorization: identify missing document, request it, resubmit, and close as authorised. |
| 🔴 **Hard** | `hard_post_discharge_followup` | Escalate worsening heart‑failure symptoms, route to cardiology clinic, book follow‑up within 7 days, arrange accessible covered transport, and finalise correctly. |

*Each task has a clear objective, a set of allowed actions, and a deterministic grader that computes a score between `0.0` and `1.0` based on the completion of key steps.*

---

## 🎮 Action Space

The environment uses a single typed action model `CareCoordAction`. Depending on the task, only a subset of action types is allowed (visible in the observation).

### 🛠️ Action Types
* `review_case` – Examine a case section (e.g., `"referral_details"`, `"insurance_rules"`)
* `route_referral` – Select a provider (by `provider_id`) for a given `specialty`
* `request_document` – Request a missing document by `document_id`
* `submit_authorization` – Resubmit prior authorization after gathering requirements
* `schedule_visit` – Book a slot (`slot_id`) for a given `provider_id`
* `arrange_transport` – Choose a transport option by `transport_id`
* `escalate_case` – Escalate symptoms or unresolved issues (with a `summary`)
* `finalize_case` – Complete the task with a `resolution_code` and `summary`

**Example Action (Python):**
```python
from care_coord_env import CareCoordAction

action = CareCoordAction(
    action_type="route_referral",
    provider_id="prov_derm_north",
    specialty="Dermatology"
)
(See models.py for the full typed schema.)👁️ Observation SpaceCareCoordObservation provides the agent with all necessary context for the next decision. Key fields include:FieldDescriptiontask_id, task_title, difficultyTask metadataobjectiveHuman‑readable goalchecklistStep‑by‑step completion hintspatient_summaryRelevant clinical and demographic infoallowed_action_typesWhich actions are currently permittedvisible_sections, revealed_textCase sections that can be reviewedprovider_optionsList of available providers with attributesscheduling_slotsAvailable appointment slotsdocument_queueDocuments that can be requestedauthorization_statusCurrent prior‑auth statetransport_optionsAccessible transport choicesblockersObstacles that must be resolvedcurrent_grader_score, score_breakdownReal‑time progress signallast_action_feedback, reward_rationaleImmediate feedback for the last step⚙️ State Space (Internal)CareCoordState tracks the underlying workflow state. This state is exposed via the state() endpoint for debugging and analysis:Sections reviewedSelected provider and scheduled slotRequested and received documentsAuthorization statusTransport arrangementEscalation completionFinal resolution codeCumulative reward and current grader score🎁 Reward DesignRewards are dense and shaped using deterministic grader deltas:Compute task score before the action.Apply the action.Compute task score after the action.Reward = score_after - score_before - penaltyUseful actions increase the score → positive reward.Invalid / low‑value actions (e.g., reviewing irrelevant sections, repeating actions) incur a small penalty.The final episode score is the grader’s final score (always between 0.0 and 1.0).📊 Graders & Reference ScoresEach task has a deterministic, programmatic grader implemented in graders.py. Graders are rule‑based (no randomness), return a score in [0.0, 1.0] with a detailed breakdown, and reflect partial progress.Because graders are deterministic, the environment is reproducible and easy to audit.Maximum Possible Scores:easy_referral_routing → 1.00medium_prior_auth_recovery → 1.00hard_post_discharge_followup → 1.00Baseline scores from the provided inference.py script depend on the LLM used. A capable model (e.g., GPT‑4o) should achieve near‑perfect scores on all tasks when given the correct prompts.🚀 Setup & InstallationPrerequisitesPython: 3.10 or higherDocker: (for containerised execution)OpenEnv CLI: pip install openenv-coreLocal DevelopmentClone the repository and navigate to the project root:Bashcd care_coord_env
Create a virtual environment and activate it:Bashpython3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
Install the package in editable mode:Bashpip install -e .
🔐 Environment Variables for InferenceThe baseline script inference.py uses the OpenAI client. Provide the following variables (e.g., in a .env file or export them):Bashexport OPENAI_API_KEY="your_openai_key"
export MODEL_NAME="gpt-4o"               # or any model supported by your endpoint
export API_BASE_URL="[https://api.openai.com/v1](https://api.openai.com/v1)"
export HF_TOKEN="your_huggingface_token" # optional, for pushing to Spaces
⚠️ Security Warning: Never commit real keys to version control. Use .env and add it to .gitignore.🖥️ Running the Environment ServerStart the FastAPI server locally:Bashuvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
The server will be available at http://localhost:8000.Client ExamplePythonfrom care_coord_env import CareCoordAction, CareCoordEnv

with CareCoordEnv(base_url="http://localhost:8000").sync() as env:
    # Start a new episode with the easy task
    result = env.reset(task_id="easy_referral_routing")
    print(result.observation.task_title)

    # Take a step
    action = CareCoordAction(action_type="review_case", section="referral_details")
    result = env.step(action)
    print(f"Reward: {result.reward}")
    print(f"Feedback: {result.observation.last_action_feedback}")
🧪 Testing & InferenceRunning TestsRun the full test suite (covers environment flows, state transitions, and grader correctness):Bashpython -m unittest discover -s tests
🤖 Baseline Inference ScriptThe file inference.py runs all three tasks against a model using the OpenAI client, producing structured logs required by the hackathon evaluator.Bashpython inference.py
Expected output format:Plaintext[START] task=easy_referral_routing model=gpt-4o
[STEP] step=1 action=review_case(section=referral_details) reward=+0.15 done=False error=None
...
[END] success=True steps=7 score=1.00 rewards=[...]
🐳 Docker Build & RunThe repository includes a Dockerfile at the root for containerised execution (required for Hugging Face Spaces).Build the image:Bashdocker build -t care_coord_env-env:latest .
Run the container:Bashdocker run -p 8000:8000 care_coord_env-env:latest
🌐 OpenEnv & Hugging Face SpacesOpenEnv ValidationEnsure your environment complies with the OpenEnv specification:Bashopenenv validate .
Hugging Face DeploymentThis environment is configured as a Docker‑based OpenEnv Space.Bashopenenv push
After deployment, the Space will publicly expose endpoints required for evaluation (/reset, /step, /state, /schema, /health, /ws, /web).📁 Project StructurePlaintextcare_coord_env/
├── .dockerignore
├── .env.example
├── __init__.py
├── client.py                 # Sync/async client for the environment
├── docs/
│   └── project_blueprint.md
├── graders.py                # Deterministic scoring logic
├── inference.py              # Baseline inference script (root)
├── models.py                 # Typed Pydantic models
├── openenv.yaml              # OpenEnv metadata
├── pyproject.toml
├── README.md                 # This file
├── task_library.py           # Task definitions and data
├── tests/
│   ├── test_environment_flows.py
│   └── test_inference_smoke.py
└── server/
    ├── app.py                # FastAPI app
    ├── care_coord_env_environment.py  # Main environment engine
    ├── Dockerfile            # Dockerfile for server
    └── requirements.txt
🛠️ Troubleshooting Common IssuesProblemLikely FixModuleNotFoundError: No module named 'care_coord_env'Activate venv and run pip install -e .openenv: command not foundpip install openenv-core while venv is activeDocker build failsEnsure Docker Desktop is running and you have internet accessPort 8000 already in useStop the existing process or use --port 8001inference.py returns 0.0 scoresCheck your OpenAI API key quota; set OPENAI_API_KEY correctlyopenenv validate failsVerify openenv.yaml exists and models are correctly typed✅ Submission ChecklistBefore final submission, confirm the following:[x] README.md (this file) is complete[x] inference.py is in the project root[x] Dockerfile is in the project root[x] openenv.yaml exists and passes validation[x] 3 tasks are defined with deterministic graders[x] All tests pass (python -m unittest discover -s tests)[x] docker build succeeds[x] openenv validate . passes[x] Hugging Face Space deploys and responds to /reset[x] Baseline script runs to completion with a valid API key[x] No hardcoded secrets in the repository📜 License & AcknowledgementsThis project is open‑source and available under the MIT License.🙏 Acknowledgements: Built for the Meta PyTorch OpenEnv Hackathon × SST | India AI Hackathon 2026. Special thanks to the OpenEnv team, Hugging Face, and PyTorch for the framework and support.Questions or feedback? Open an issue or reach out via the Hugging Face Space community tab.