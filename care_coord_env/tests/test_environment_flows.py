"""End-to-end regression tests for CareCoordEnv."""

import sys
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from care_coord_env.models import ActionType, CaseSection, ResolutionCode
from care_coord_env.server.care_coord_env_environment import CareCoordEnvironment
from care_coord_env import CareCoordAction


class CareCoordEnvironmentTests(unittest.TestCase):
    """Validate the happy path and a representative invalid path."""

    def setUp(self) -> None:
        self.env = CareCoordEnvironment()

    def test_easy_referral_happy_path(self) -> None:
        self.env.reset(task_id="easy_referral_routing")
        actions = [
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.REFERRAL),
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.INSURANCE),
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.PROVIDERS),
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.AVAILABILITY),
            CareCoordAction(
                action_type=ActionType.ROUTE_REFERRAL,
                specialty="Dermatology",
                provider_id="prov_derm_north",
            ),
            CareCoordAction(
                action_type=ActionType.SCHEDULE_VISIT,
                provider_id="prov_derm_north",
                slot_id="slot_north_3d",
            ),
            CareCoordAction(
                action_type=ActionType.FINALIZE_CASE,
                resolution_code=ResolutionCode.SCHEDULED,
                summary="Scheduled the correct dermatology referral.",
            ),
        ]
        self._play(actions)
        self.assertAlmostEqual(self.env.state.current_score, 1.0)

    def test_medium_prior_auth_happy_path(self) -> None:
        self.env.reset(task_id="medium_prior_auth_recovery")
        actions = [
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.REFERRAL),
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.INSURANCE),
            CareCoordAction(
                action_type=ActionType.REVIEW_CASE, section=CaseSection.AUTH_REQUIREMENTS
            ),
            CareCoordAction(
                action_type=ActionType.REVIEW_CASE, section=CaseSection.DOCUMENT_QUEUE
            ),
            CareCoordAction(
                action_type=ActionType.REQUEST_DOCUMENT,
                document_id="doc_latest_clinical_note",
            ),
            CareCoordAction(action_type=ActionType.SUBMIT_AUTHORIZATION),
            CareCoordAction(
                action_type=ActionType.FINALIZE_CASE,
                resolution_code=ResolutionCode.AUTHORIZED,
                summary="Authorization approved after adding the missing note.",
            ),
        ]
        self._play(actions)
        self.assertAlmostEqual(self.env.state.current_score, 1.0)

    def test_hard_post_discharge_happy_path(self) -> None:
        self.env.reset(task_id="hard_post_discharge_followup")
        actions = [
            CareCoordAction(
                action_type=ActionType.REVIEW_CASE, section=CaseSection.DISCHARGE_PLAN
            ),
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.INSURANCE),
            CareCoordAction(
                action_type=ActionType.REVIEW_CASE, section=CaseSection.SYMPTOM_ALERTS
            ),
            CareCoordAction(action_type=ActionType.REVIEW_CASE, section=CaseSection.PROVIDERS),
            CareCoordAction(
                action_type=ActionType.REVIEW_CASE, section=CaseSection.AVAILABILITY
            ),
            CareCoordAction(
                action_type=ActionType.REVIEW_CASE, section=CaseSection.TRANSPORT_OPTIONS
            ),
            CareCoordAction(
                action_type=ActionType.ESCALATE_CASE,
                summary="Escalated worsening HF symptoms to the nurse line.",
            ),
            CareCoordAction(
                action_type=ActionType.ROUTE_REFERRAL,
                specialty="Cardiology",
                provider_id="prov_cardio_hf",
            ),
            CareCoordAction(
                action_type=ActionType.SCHEDULE_VISIT,
                provider_id="prov_cardio_hf",
                slot_id="slot_cardio_2d",
            ),
            CareCoordAction(
                action_type=ActionType.ARRANGE_TRANSPORT,
                transport_id="transport_wheelchair_van",
            ),
            CareCoordAction(
                action_type=ActionType.FINALIZE_CASE,
                resolution_code=ResolutionCode.SCHEDULED_AND_ESCALATED,
                summary="Escalated symptoms, scheduled urgent follow-up, and arranged transport.",
            ),
        ]
        self._play(actions)
        self.assertAlmostEqual(self.env.state.current_score, 1.0)

    def test_invalid_finalize_is_penalized(self) -> None:
        self.env.reset(task_id="easy_referral_routing")
        result = self.env.step(
            CareCoordAction(
                action_type=ActionType.FINALIZE_CASE,
                resolution_code=ResolutionCode.SCHEDULED,
                summary="Trying to close too early.",
            )
        )
        self.assertLess(result.reward or 0.0, 0.0)
        self.assertFalse(result.done)
        self.assertIsNone(self.env.state.final_resolution_code)

    def _play(self, actions: list[CareCoordAction]) -> None:
        for action in actions:
            result = self.env.step(action)
        self.assertTrue(result.done)


if __name__ == "__main__":
    unittest.main()
