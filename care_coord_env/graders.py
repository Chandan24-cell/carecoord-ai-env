# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic task graders for CareCoordEnv."""

from dataclasses import dataclass

try:
    from .models import (
        AuthorizationStatus,
        CaseSection,
        CareCoordState,
        ResolutionCode,
        TaskType,
    )
    from .task_library import CaseSpec
except ImportError:
    from models import (
        AuthorizationStatus,
        CaseSection,
        CareCoordState,
        ResolutionCode,
        TaskType,
    )
    from task_library import CaseSpec


@dataclass(frozen=True)
class GradeResult:
    """Structured grading result for a task state."""

    score: float
    breakdown: dict[str, float]
    success: bool


def grade_case(case: CaseSpec, state: CareCoordState) -> GradeResult:
    """Return a deterministic score in the range [0.0, 1.0] for the current state."""
    if case.task_type == TaskType.REFERRAL_ROUTING:
        breakdown = _grade_referral(case, state)
    elif case.task_type == TaskType.PRIOR_AUTHORIZATION:
        breakdown = _grade_prior_auth(case, state)
    else:
        breakdown = _grade_post_discharge(case, state)

    score = round(sum(breakdown.values()), 4)
    score = min(max(score, 0.0), 1.0)
    success = score >= 0.999 and state.final_resolution_code == case.resolution_code_required
    return GradeResult(score=score, breakdown=breakdown, success=success)


def _grade_referral(case: CaseSpec, state: CareCoordState) -> dict[str, float]:
    sections = set(state.revealed_sections)
    breakdown = {
        "review_referral_details": 0.10 if CaseSection.REFERRAL in sections else 0.0,
        "review_insurance_rules": 0.10 if CaseSection.INSURANCE in sections else 0.0,
        "review_provider_directory": 0.10 if CaseSection.PROVIDERS in sections else 0.0,
        "review_scheduling_board": 0.05 if CaseSection.AVAILABILITY in sections else 0.0,
        "select_correct_provider": 0.25
        if state.selected_provider_id == case.correct_provider_id
        else 0.0,
        "book_acceptable_slot": 0.20
        if state.scheduled_slot_id in case.acceptable_slot_ids
        else 0.0,
        "book_earliest_slot": 0.05
        if state.scheduled_slot_id == case.optimal_slot_id
        else 0.0,
        "finalize_case": 0.15
        if state.final_resolution_code == ResolutionCode.SCHEDULED
        else 0.0,
    }
    return breakdown


def _grade_prior_auth(case: CaseSpec, state: CareCoordState) -> dict[str, float]:
    sections = set(state.revealed_sections)
    requested_or_received = set(state.requested_document_ids) | set(state.received_document_ids)
    breakdown = {
        "review_referral_details": 0.10 if CaseSection.REFERRAL in sections else 0.0,
        "review_insurance_rules": 0.10 if CaseSection.INSURANCE in sections else 0.0,
        "review_auth_requirements": 0.10
        if CaseSection.AUTH_REQUIREMENTS in sections
        else 0.0,
        "review_document_queue": 0.10
        if CaseSection.DOCUMENT_QUEUE in sections
        else 0.0,
        "request_missing_required_doc": 0.20
        if "doc_latest_clinical_note" in requested_or_received
        else 0.0,
        "collect_required_documents": 0.15
        if set(case.required_document_ids).issubset(set(state.received_document_ids))
        else 0.0,
        "obtain_authorization_approval": 0.15
        if state.authorization_status == AuthorizationStatus.APPROVED
        else 0.0,
        "finalize_case": 0.10
        if state.final_resolution_code == ResolutionCode.AUTHORIZED
        else 0.0,
    }
    return breakdown


def _grade_post_discharge(case: CaseSpec, state: CareCoordState) -> dict[str, float]:
    sections = set(state.revealed_sections)
    breakdown = {
        "review_discharge_plan": 0.05
        if CaseSection.DISCHARGE_PLAN in sections
        else 0.0,
        "review_insurance_rules": 0.05 if CaseSection.INSURANCE in sections else 0.0,
        "review_symptom_alerts": 0.10
        if CaseSection.SYMPTOM_ALERTS in sections
        else 0.0,
        "review_provider_directory": 0.05 if CaseSection.PROVIDERS in sections else 0.0,
        "review_scheduling_board": 0.05
        if CaseSection.AVAILABILITY in sections
        else 0.0,
        "review_transport_options": 0.05
        if CaseSection.TRANSPORT_OPTIONS in sections
        else 0.0,
        "complete_required_escalation": 0.15 if state.escalation_completed else 0.0,
        "select_correct_provider": 0.15
        if state.selected_provider_id == case.correct_provider_id
        else 0.0,
        "book_acceptable_slot": 0.15
        if state.scheduled_slot_id in case.acceptable_slot_ids
        else 0.0,
        "arrange_accessible_transport": 0.10
        if state.arranged_transport_id == case.correct_transport_id
        else 0.0,
        "finalize_case": 0.10
        if state.final_resolution_code == ResolutionCode.SCHEDULED_AND_ESCALATED
        else 0.0,
    }
    return breakdown
