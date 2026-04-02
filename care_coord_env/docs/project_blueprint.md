# CareCoordEnv Project Blueprint

## Goal

Build a real-world OpenEnv environment where an AI agent acts like a healthcare
care-coordination assistant. The agent will work through structured cases that
involve referrals, prior authorization, and post-discharge follow-up.

## Why This Project Fits The Hackathon

- Real-world utility: care coordination is a daily operational task in
  hospitals, clinics, and insurance-linked care teams.
- Safe scope: the environment evaluates workflow execution, not medical
  diagnosis or treatment advice.
- Clear grading: tasks can be scored deterministically from the final case
  state, timeline, and required artifacts.
- Good reward shaping: the environment can reward intermediate progress such as
  identifying missing documents, choosing the right specialty, and booking a
  compliant follow-up.

## Environment Concept

The environment simulates a queue of patient coordination cases. For each case,
the agent receives:

- patient summary
- insurance information
- constraints and deadlines
- available providers, slots, and documentation
- a case objective

The agent must inspect the case, take allowed actions, and complete the
coordination workflow without unsafe or wasteful behavior.

## First Three Tasks

### 1. Easy: Referral Routing

Route a straightforward dermatology referral to the correct in-network clinic
and book the earliest valid slot.

### 2. Medium: Prior Authorization Recovery

Review an MRI authorization case, discover the missing clinical note, request
the missing item, and resubmit successfully.

### 3. Hard: Post-Discharge Follow-Up

Coordinate a cardiology follow-up within seven days after discharge while
respecting insurance coverage, clinic availability, transport limitations, and
urgent symptoms that require escalation.

## Planned Agent Action Categories

- `review_case`: inspect different parts of the case
- `route_referral`: choose specialty or destination clinic
- `request_document`: ask for missing paperwork
- `submit_authorization`: file or resubmit an auth request
- `schedule_visit`: book a provider slot
- `escalate_case`: flag urgent or blocked cases
- `finalize_case`: submit the final resolution

## Planned Observation Areas

- case summary
- visible documents
- action history
- pending blockers
- scheduling options
- current score breakdown

## Reward Design Direction

- positive reward for valid progress toward completion
- small reward for uncovering the correct blocker early
- bonus for efficient completion in fewer steps
- penalties for invalid actions, repeated loops, unsafe routing, or submitting
  incomplete cases

## What We Build Next

1. Replace the echo models with real typed models for actions, observations, and
   state.
2. Implement the internal case simulator and task catalog.
3. Add deterministic graders and shaped rewards.
4. Add `inference.py`, Docker, validation flow, and deployment setup.
