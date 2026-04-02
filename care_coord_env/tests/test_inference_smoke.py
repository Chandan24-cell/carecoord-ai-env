"""Smoke tests for the submission baseline runner."""

import asyncio
import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import inference
from care_coord_env.server.care_coord_env_environment import CareCoordEnvironment


class LocalAsyncEnv:
    """Tiny async adapter around the in-process environment for inference tests."""

    def __init__(self) -> None:
        self._env = CareCoordEnvironment()

    async def reset(self, **kwargs):
        observation = self._env.reset(**kwargs)
        return SimpleNamespace(
            observation=observation,
            reward=getattr(observation, "reward", None),
            done=getattr(observation, "done", False),
        )

    async def step(self, action):
        observation = self._env.step(action)
        return SimpleNamespace(
            observation=observation,
            reward=getattr(observation, "reward", None),
            done=getattr(observation, "done", False),
        )

    async def state(self):
        return self._env.state


def scripted_action(_client, _model_name, observation, _history):
    """Return the next perfect-path action for each deterministic task."""
    visible_sections = set(observation["visible_sections"])
    progress = observation["progress"]
    task_id = observation["task_id"]

    if task_id == "easy_referral_routing":
        for section in [
            "referral_details",
            "insurance_rules",
            "provider_directory",
            "scheduling_board",
        ]:
            if section not in visible_sections:
                return {"action_type": "review_case", "section": section}
        if not progress["selected_provider_id"]:
            return {
                "action_type": "route_referral",
                "specialty": "Dermatology",
                "provider_id": "prov_derm_north",
            }
        if not progress["scheduled_slot_id"]:
            return {
                "action_type": "schedule_visit",
                "provider_id": "prov_derm_north",
                "slot_id": "slot_north_3d",
            }
        return {
            "action_type": "finalize_case",
            "resolution_code": "scheduled",
            "summary": "Scheduled the correct dermatology follow-up.",
        }

    if task_id == "medium_prior_auth_recovery":
        for section in [
            "referral_details",
            "insurance_rules",
            "authorization_requirements",
            "document_queue",
        ]:
            if section not in visible_sections:
                return {"action_type": "review_case", "section": section}
        if "doc_latest_clinical_note" not in progress["requested_document_ids"]:
            return {
                "action_type": "request_document",
                "document_id": "doc_latest_clinical_note",
            }
        if progress["authorization_status"] != "approved":
            return {"action_type": "submit_authorization"}
        return {
            "action_type": "finalize_case",
            "resolution_code": "authorized",
            "summary": "Recovered the MRI prior authorization.",
        }

    for section in [
        "discharge_plan",
        "insurance_rules",
        "symptom_alerts",
        "provider_directory",
        "scheduling_board",
        "transport_options",
    ]:
        if section not in visible_sections:
            return {"action_type": "review_case", "section": section}
    if not progress["escalation_completed"]:
        return {
            "action_type": "escalate_case",
            "summary": "Escalated symptom alerts to the nurse triage line.",
        }
    if not progress["selected_provider_id"]:
        return {
            "action_type": "route_referral",
            "specialty": "Cardiology",
            "provider_id": "prov_cardio_hf",
        }
    if not progress["scheduled_slot_id"]:
        return {
            "action_type": "schedule_visit",
            "provider_id": "prov_cardio_hf",
            "slot_id": "slot_cardio_2d",
        }
    if not progress["arranged_transport_id"]:
        return {
            "action_type": "arrange_transport",
            "transport_id": "transport_wheelchair_van",
        }
    return {
        "action_type": "finalize_case",
        "resolution_code": "scheduled_and_escalated",
        "summary": "Escalated symptoms, scheduled urgent follow-up, and arranged transport.",
    }


class InferenceSmokeTests(unittest.TestCase):
    """Verify the submission runner works against the local environment API."""

    def test_run_task_emits_structured_logs(self) -> None:
        env = LocalAsyncEnv()
        stdout = io.StringIO()

        with patch.object(inference, "get_model_action", side_effect=scripted_action):
            with redirect_stdout(stdout):
                result = asyncio.run(
                    inference.run_task(
                        env=env,
                        client=object(),
                        model_name="stub-model",
                        task_id="hard_post_discharge_followup",
                    )
                )

        logs = stdout.getvalue().strip().splitlines()
        self.assertTrue(logs[0].startswith("[START]"))
        self.assertTrue(any(line.startswith("[STEP]") for line in logs))
        self.assertTrue(logs[-1].startswith("[END]"))
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["score"], 1.0)

    def test_inference_script_runs_directly_without_import_error(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("MODEL_NAME", None)
        env.pop("OPENAI_API_KEY", None)
        env.pop("HF_TOKEN", None)
        env.pop("API_BASE_URL", None)

        result = subprocess.run(
            [sys.executable, "inference.py"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        combined_output = result.stdout + result.stderr
        self.assertIn("Missing required environment variable: MODEL_NAME", combined_output)
        self.assertNotIn("ModuleNotFoundError: No module named 'care_coord_env'", combined_output)


if __name__ == "__main__":
    unittest.main()
