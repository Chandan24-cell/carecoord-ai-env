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
---

# CareCoordEnv

CareCoordEnv is a healthcare operations OpenEnv benchmark where an AI agent acts
as a care-coordination assistant. The environment simulates realistic back-office
tasks that human teams perform every day:

- routing specialist referrals
- recovering denied prior authorizations
- coordinating post-discharge follow-up

The project is designed for the Meta PyTorch OpenEnv Hackathon Round 1 problem
statement and follows the standard `step()` / `reset()` / `state()` API with
typed Pydantic action, observation, and state models.

## Why This Environment Matters

Care coordination is a real operational bottleneck in healthcare. A strong agent
must combine:

- policy understanding
- workflow reasoning
- deadline awareness
- safe escalation behavior
- document and scheduling discipline

This environment keeps the domain realistic while avoiding unsafe diagnosis or
treatment generation. The agent is evaluated on operational execution, not
clinical judgment.

## Task Catalog

### 1. Easy: Referral Routing

Task id: `easy_referral_routing`

Route a dermatology referral to the correct in-network provider and book the
earliest acceptable slot.

What makes it easy:

- only one correct specialty
- clear network rule
- small provider directory
- straightforward scheduling choice

### 2. Medium: Prior Authorization Recovery

Task id: `medium_prior_auth_recovery`

Recover a denied MRI prior authorization by identifying the missing document,
requesting it, resubmitting, and closing the case as authorized.

What makes it medium:

- requires document reasoning
- authorization can fail if submitted too early
- more workflow state than the referral task

### 3. Hard: Post-Discharge Follow-Up

Task id: `hard_post_discharge_followup`

Escalate worsening heart-failure symptoms, route to the right in-network
cardiology clinic, book follow-up within 7 days, arrange accessible covered
transport, and finalize with the correct escalated resolution.

What makes it hard:

- multi-constraint workflow
- symptom escalation requirement
- transport coordination
- strict timing window

## Action Space

The environment uses a single typed action model, `CareCoordAction`, with
task-specific action subsets surfaced in each observation.

Main action types:

- `review_case`
- `route_referral`
- `request_document`
- `submit_authorization`
- `schedule_visit`
- `arrange_transport`
- `escalate_case`
- `finalize_case`

Main action fields:

- `action_type`
- `section`
- `specialty`
- `provider_id`
- `slot_id`
- `document_id`
- `transport_id`
- `resolution_code`
- `summary`

See [models.py](/Users/chandankumarsah/Documents/New%20project/care_coord_env/models.py) for the full typed schema.

## Observation Space

`CareCoordObservation` returns the full case context needed for the next action.

Important fields:

- task metadata: `task_id`, `task_title`, `task_type`, `difficulty`
- objective text and checklist
- patient summary
- allowed action types
- visible sections and revealed text
- provider options
- scheduling slots
- document queue
- authorization status
- transport options
- blockers
- progress snapshot
- current grader score and score breakdown
- last action feedback and reward rationale

## State Space

`CareCoordState` tracks the underlying workflow state:

- reviewed sections
- selected provider
- scheduled slot
- requested and received documents
- authorization status
- arranged transport
- escalation completion
- final resolution code
- cumulative reward
- current grader score

## Reward Design

The reward function is shaped using deterministic grader deltas.

At each step:

1. The environment computes the task score before the action.
2. It applies the action.
3. It recomputes the score.
4. Reward = `score_after - score_before - penalty`

This gives dense, meaningful rewards:

- reviewing the right section increases score
- retrieving the missing document increases score
- selecting the correct provider increases score
- arranging covered accessible transport increases score
- invalid or low-value actions are penalized

Because grader scores live in `[0.0, 1.0]`, the task score is deterministic and
easy to audit.

## Graders

Each task has a deterministic programmatic grader in
[graders.py](/Users/chandankumarsah/Documents/New%20project/care_coord_env/graders.py).

Scoring properties:

- score range is always `0.0` to `1.0`
- each task has named weighted components
- no randomness is used
- partial progress is visible through `score_breakdown`

## Reference Scores

Deterministic perfect-path scores for the environment logic:

- `easy_referral_routing`: `1.00`
- `medium_prior_auth_recovery`: `1.00`
- `hard_post_discharge_followup`: `1.00`

The model-driven baseline script is [inference.py](/Users/chandankumarsah/Documents/New%20project/care_coord_env/inference.py). It uses the OpenAI client and reads:

- `OPENAI_API_KEY`
- `MODEL_NAME`
- `API_BASE_URL`
- `HF_TOKEN`

## Local Setup

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Secrets should stay in environment variables, not committed files.

Example:

```bash
cp .env.example .env
```

## Running Locally

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Connect with the client:

```python
from care_coord_env import CareCoordAction, CareCoordEnv

with CareCoordEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy_referral_routing")
    result = env.step(
        CareCoordAction(action_type="review_case", section="referral_details")
    )
    print(result.observation.last_action_feedback)
```

## Running Tests

```bash
python -m unittest discover -s tests
```

## Running the Baseline Inference Script

The inference script must be located at the project root for hackathon
submission. It is already provided as:

- [inference.py](/Users/chandankumarsah/Documents/New%20project/care_coord_env/inference.py)

Run it with:

```bash
python inference.py
```

Use an API key with available quota. If the provider returns `429 insufficient_quota`, the script will still complete but all task scores will remain `0.0`.

## Docker Build

The repo includes a root-level [Dockerfile](/Users/chandankumarsah/Documents/New%20project/care_coord_env/Dockerfile) for Hugging Face Spaces and local container testing.

Build and run locally with:

```bash
docker build -t care_coord_env-env:latest .
docker run -p 8000:8000 care_coord_env-env:latest
```

Expected behavior:

- starts from Docker by default using `care_coord_env-env:latest`
- runs all 3 tasks
- prints structured `[START]`, `[STEP]`, and `[END]` logs
- prints a JSON summary at the end

## OpenEnv Validation

From the project root:

```bash
openenv validate .
```

## Hugging Face Spaces

This environment is configured as a Docker-based OpenEnv Space.

Push with:

```bash
openenv push
```

After deployment the Space should expose:

- `/reset`
- `/step`
- `/state`
- `/schema`
- `/health`
- `/ws`
- `/web`

## Project Structure

```text
care_coord_env/
├── .dockerignore
├── .env.example
├── __init__.py
├── client.py
├── docs/
│   └── project_blueprint.md
├── graders.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── task_library.py
├── tests/
│   └── test_environment_flows.py
└── server/
    ├── app.py
    ├── care_coord_env_environment.py
    ├── Dockerfile
    └── requirements.txt
```
