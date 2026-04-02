# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-task environment implementation for CareCoordEnv."""

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..graders import grade_case
    from ..models import (
        ActionType,
        AppointmentSlot,
        AuthorizationStatus,
        CaseSection,
        CaseStatus,
        CareCoordAction,
        CareCoordObservation,
        CareCoordState,
        DocumentOption,
        DocumentStatus,
        ProgressSnapshot,
        ProviderOption,
        ResolutionCode,
        TransportOption,
    )
    from ..task_library import CaseSpec, DocumentRecord, get_case
except ImportError:
    from graders import grade_case
    from models import (
        ActionType,
        AppointmentSlot,
        AuthorizationStatus,
        CaseSection,
        CaseStatus,
        CareCoordAction,
        CareCoordObservation,
        CareCoordState,
        DocumentOption,
        DocumentStatus,
        ProgressSnapshot,
        ProviderOption,
        ResolutionCode,
        TransportOption,
    )
    from task_library import CaseSpec, DocumentRecord, get_case


class CareCoordEnvironment(Environment):
    """
    Healthcare care-coordination environment with three deterministic tasks.

    The environment currently supports:
    - referral routing
    - prior authorization recovery
    - post-discharge follow-up
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment with the default easy case."""
        self._case = get_case()
        self._state = self._fresh_state(self._case)
        self._sync_score()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> CareCoordObservation:
        """Reset the environment into one of the deterministic benchmark tasks."""
        del seed
        requested_task_id = task_id or kwargs.get("task_id")
        self._case = get_case(requested_task_id)
        self._state = self._fresh_state(self._case, episode_id=episode_id)
        self._sync_score()
        return self._build_observation(
            reward=0.0,
            done=False,
            feedback=f"Loaded task '{self._case.task_id}'.",
            rationale="Reset initializes the episode without granting progress reward.",
        )

    def step(
        self,
        action: CareCoordAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CareCoordObservation:  # type: ignore[override]
        """Execute a care-coordination action against the active task."""
        del timeout_s, kwargs

        if self._state.case_status in {CaseStatus.COMPLETED, CaseStatus.FAILED}:
            return self._build_observation(
                reward=-0.05,
                done=True,
                feedback="Episode already closed. Reset the environment to start a new task.",
                rationale="Actions after termination receive a small penalty.",
            )

        before_score = self._state.current_score
        self._state.step_count += 1
        self._state.case_status = CaseStatus.IN_PROGRESS

        penalty = 0.0
        done = False

        if action.action_type == ActionType.REVIEW_CASE:
            penalty, feedback, rationale = self._handle_review_case(action)
        elif action.action_type == ActionType.ROUTE_REFERRAL:
            penalty, feedback, rationale = self._handle_route_referral(action)
        elif action.action_type == ActionType.REQUEST_DOCUMENT:
            penalty, feedback, rationale = self._handle_request_document(action)
        elif action.action_type == ActionType.SUBMIT_AUTHORIZATION:
            penalty, feedback, rationale = self._handle_submit_authorization(action)
        elif action.action_type == ActionType.SCHEDULE_VISIT:
            penalty, feedback, rationale = self._handle_schedule_visit(action)
        elif action.action_type == ActionType.ARRANGE_TRANSPORT:
            penalty, feedback, rationale = self._handle_arrange_transport(action)
        elif action.action_type == ActionType.ESCALATE_CASE:
            penalty, feedback, rationale = self._handle_escalation(action)
        else:
            penalty, feedback, rationale, done = self._handle_finalize_case(action)

        self._sync_score()
        reward = round(self._state.current_score - before_score - penalty, 4)

        if (
            self._state.step_count >= self._state.max_steps
            and self._state.case_status != CaseStatus.COMPLETED
        ):
            self._state.case_status = CaseStatus.FAILED
            done = True
            reward = round(reward - 0.10, 4)
            feedback = f"{feedback} Step limit reached before the task was completed."
            rationale = (
                f"{rationale} A timeout penalty was applied because the episode hit "
                "the step limit."
            )

        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)
        self._state.last_action_feedback = feedback
        self._state.last_reward_rationale = rationale
        self._state.action_history.append(self._format_action_history(action))

        return self._build_observation(
            reward=reward,
            done=done,
            feedback=feedback,
            rationale=rationale,
        )

    @property
    def state(self) -> CareCoordState:
        """Expose the current typed environment state."""
        return self._state

    def _fresh_state(
        self, case: CaseSpec, episode_id: Optional[str] = None
    ) -> CareCoordState:
        """Create a fresh state object for a selected task."""
        received_docs = [
            document.document_id
            for document in case.documents
            if document.initially_available
        ]
        return CareCoordState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=case.task_id,
            task_title=case.title,
            task_type=case.task_type,
            difficulty=case.difficulty,
            case_status=CaseStatus.READY,
            max_steps=case.max_steps,
            received_document_ids=received_docs,
        )

    def _sync_score(self) -> None:
        """Recompute the current grader score and component breakdown."""
        grade = grade_case(self._case, self._state)
        self._state.current_score = grade.score
        self._state.score_breakdown = grade.breakdown

    def _build_observation(
        self, reward: float, done: bool, feedback: str, rationale: str
    ) -> CareCoordObservation:
        """Construct the current observation from case data and state."""
        visible_sections = set(self._state.revealed_sections)

        provider_options = []
        if CaseSection.PROVIDERS in visible_sections:
            provider_options = [
                ProviderOption(
                    provider_id=provider.provider_id,
                    clinic_name=provider.clinic_name,
                    specialty=provider.specialty,
                    clinician_name=provider.clinician_name,
                    clinician_gender=provider.clinician_gender,
                    in_network=provider.in_network,
                    accepts_new_patients=provider.accepts_new_patients,
                    notes=provider.notes,
                )
                for provider in self._case.providers
            ]

        available_slots = []
        if CaseSection.AVAILABILITY in visible_sections:
            available_slots = [
                AppointmentSlot(
                    slot_id=slot.slot_id,
                    provider_id=slot.provider_id,
                    date_label=slot.date_label,
                    days_from_today=slot.days_from_today,
                    is_virtual=slot.is_virtual,
                    notes=slot.notes,
                )
                for slot in self._case.slots
            ]

        documents = []
        if CaseSection.DOCUMENT_QUEUE in visible_sections:
            documents = [self._document_option(document) for document in self._case.documents]

        transport_options = []
        if CaseSection.TRANSPORT_OPTIONS in visible_sections:
            transport_options = [
                TransportOption(
                    transport_id=option.transport_id,
                    label=option.label,
                    covered=option.covered,
                    accessible=option.accessible,
                    notes=option.notes,
                )
                for option in self._case.transport_options
            ]

        progress = ProgressSnapshot(
            reviewed_sections=self._state.revealed_sections,
            selected_provider_id=self._state.selected_provider_id,
            scheduled_slot_id=self._state.scheduled_slot_id,
            requested_document_ids=self._state.requested_document_ids,
            received_document_ids=self._state.received_document_ids,
            authorization_status=self._state.authorization_status,
            arranged_transport_id=self._state.arranged_transport_id,
            escalation_completed=self._state.escalation_completed,
            ready_to_finalize=self._ready_to_finalize(),
            invalid_action_count=self._state.invalid_action_count,
        )

        return CareCoordObservation(
            task_id=self._case.task_id,
            task_title=self._case.title,
            task_type=self._case.task_type,
            difficulty=self._case.difficulty,
            current_status=self._state.case_status,
            objective=self._case.objective,
            objective_checklist=list(self._case.objective_checklist),
            patient_name=self._case.patient_name,
            patient_age=self._case.patient_age,
            patient_brief=self._case.patient_brief,
            allowed_action_types=list(self._case.allowed_actions),
            available_sections=list(self._case.sections.keys()),
            visible_sections=self._state.revealed_sections,
            visible_section_content={
                section.value: self._case.sections[section]
                for section in self._state.revealed_sections
            },
            provider_options=provider_options,
            available_slots=available_slots,
            documents=documents,
            authorization_status=self._state.authorization_status,
            transport_options=transport_options,
            blockers=self._build_blockers(),
            progress=progress,
            current_score=self._state.current_score,
            score_breakdown=self._state.score_breakdown,
            last_action_feedback=feedback,
            reward_rationale=rationale,
            done=done,
            reward=reward,
            metadata={
                "step_count": self._state.step_count,
                "remaining_steps": max(self._state.max_steps - self._state.step_count, 0),
                "cumulative_reward": self._state.cumulative_reward,
                "task_type": self._case.task_type.value,
            },
        )

    def _document_option(self, document: DocumentRecord) -> DocumentOption:
        """Build a visible document option from internal state."""
        if document.document_id in self._state.received_document_ids:
            status = (
                DocumentStatus.AVAILABLE
                if document.initially_available
                else DocumentStatus.RECEIVED
            )
        else:
            status = DocumentStatus.MISSING
        return DocumentOption(
            document_id=document.document_id,
            label=document.label,
            status=status,
            required_for_authorization=document.required_for_authorization,
            description=document.description,
        )

    def _build_blockers(self) -> list[str]:
        """Summarize the current operational blockers for the active task."""
        blockers: list[str] = []

        if self._case.task_type == self._case.task_type.REFERRAL_ROUTING:
            if CaseSection.REFERRAL not in self._state.revealed_sections:
                blockers.append("Referral details have not been reviewed.")
            if CaseSection.INSURANCE not in self._state.revealed_sections:
                blockers.append("Insurance rules are still hidden.")
            if self._state.selected_provider_id != self._case.correct_provider_id:
                blockers.append("Correct in-network dermatology provider has not been selected.")
            if self._state.scheduled_slot_id not in self._case.acceptable_slot_ids:
                blockers.append("A compliant appointment has not been booked.")
        elif self._case.task_type == self._case.task_type.PRIOR_AUTHORIZATION:
            if CaseSection.AUTH_REQUIREMENTS not in self._state.revealed_sections:
                blockers.append("Authorization requirements have not been reviewed.")
            missing_required = [
                doc_id
                for doc_id in self._case.required_document_ids
                if doc_id not in self._state.received_document_ids
            ]
            if missing_required:
                blockers.append(
                    "Required authorization documents are still missing: "
                    + ", ".join(missing_required)
                )
            if self._state.authorization_status != AuthorizationStatus.APPROVED:
                blockers.append("Prior authorization is not yet approved.")
        else:
            if CaseSection.SYMPTOM_ALERTS not in self._state.revealed_sections:
                blockers.append("Symptom alert note has not been reviewed.")
            if not self._state.escalation_completed:
                blockers.append("Required heart-failure nurse escalation is still pending.")
            if self._state.selected_provider_id != self._case.correct_provider_id:
                blockers.append("Correct in-network cardiology clinic has not been selected.")
            if self._state.scheduled_slot_id not in self._case.acceptable_slot_ids:
                blockers.append("A compliant cardiology follow-up slot has not been booked.")
            if self._state.arranged_transport_id != self._case.correct_transport_id:
                blockers.append("Accessible covered transport has not been arranged.")

        return blockers

    def _handle_review_case(
        self, action: CareCoordAction
    ) -> tuple[float, str, str]:
        """Reveal a hidden section of the active case."""
        assert action.section is not None

        if action.section not in self._case.sections:
            return self._invalid_action(
                feedback=f"{action.section.value} is not available for this task.",
                rationale="Reviewing a non-existent section is invalid.",
                penalty=0.08,
            )

        if action.section in self._state.revealed_sections:
            return self._invalid_action(
                feedback=f"{action.section.value} was already reviewed.",
                rationale="Repeated review actions provide no new progress.",
                penalty=0.03,
            )

        self._state.revealed_sections.append(action.section)
        return (
            0.0,
            f"Reviewed {action.section.value} successfully.",
            "The action revealed a new part of the case, increasing the graded progress score.",
        )

    def _handle_route_referral(
        self, action: CareCoordAction
    ) -> tuple[float, str, str]:
        """Select a referral destination provider."""
        if ActionType.ROUTE_REFERRAL not in self._case.allowed_actions:
            return self._invalid_action(
                feedback="Referral routing is not part of this task.",
                rationale="This task does not support provider routing.",
                penalty=0.08,
            )

        provider = next(
            (provider for provider in self._case.providers if provider.provider_id == action.provider_id),
            None,
        )
        if provider is None:
            return self._invalid_action(
                feedback="Selected provider_id does not exist in the directory.",
                rationale="Choosing an unknown provider is invalid.",
                penalty=0.12,
            )

        specialty = (action.specialty or "").strip().lower()
        if specialty != provider.specialty.lower():
            return self._invalid_action(
                feedback="The chosen specialty does not match the selected provider.",
                rationale="Mismatched specialty routing is penalized.",
                penalty=0.10,
            )

        self._state.selected_provider_id = provider.provider_id

        if not provider.in_network:
            return self._invalid_action(
                feedback="That provider is out of network for this case.",
                rationale="Out-of-network routing is undesirable.",
                penalty=0.18,
            )

        if not provider.accepts_new_patients:
            return self._invalid_action(
                feedback="That provider is not accepting new patients.",
                rationale="Routing to a closed panel wastes a coordination step.",
                penalty=0.15,
            )

        if provider.provider_id != self._case.correct_provider_id:
            return (
                0.05,
                f"Provider {provider.clinic_name} selected, but it is not the best task match.",
                "The action is operationally valid, but the grader will reserve full credit for the correct provider.",
            )

        return (
            0.0,
            f"Referral routed to {provider.clinic_name}.",
            "The selected provider matches the intended care destination.",
        )

    def _handle_request_document(
        self, action: CareCoordAction
    ) -> tuple[float, str, str]:
        """Request a missing document for authorization recovery."""
        if ActionType.REQUEST_DOCUMENT not in self._case.allowed_actions:
            return self._invalid_action(
                feedback="Document requests are not part of this task.",
                rationale="This task does not support document retrieval actions.",
                penalty=0.08,
            )

        document = next(
            (document for document in self._case.documents if document.document_id == action.document_id),
            None,
        )
        if document is None:
            return self._invalid_action(
                feedback="Selected document_id does not exist.",
                rationale="Choosing an unknown document is invalid.",
                penalty=0.10,
            )

        if document.document_id in self._state.received_document_ids:
            return self._invalid_action(
                feedback=f"{document.label} is already available.",
                rationale="Re-requesting an available document wastes a step.",
                penalty=0.04,
            )

        if not document.requestable:
            return self._invalid_action(
                feedback=f"{document.label} cannot be requested through this workflow.",
                rationale="The action does not match the document policy.",
                penalty=0.10,
            )

        self._state.requested_document_ids.append(document.document_id)
        self._state.received_document_ids.append(document.document_id)
        return (
            0.0,
            f"Requested and received {document.label}.",
            "The document is now available for the next authorization attempt.",
        )

    def _handle_submit_authorization(
        self, action: CareCoordAction
    ) -> tuple[float, str, str]:
        """Submit or resubmit a prior authorization."""
        del action
        if ActionType.SUBMIT_AUTHORIZATION not in self._case.allowed_actions:
            return self._invalid_action(
                feedback="Authorization submission is not part of this task.",
                rationale="This task does not support prior authorization actions.",
                penalty=0.08,
            )

        missing_required = [
            doc_id
            for doc_id in self._case.required_document_ids
            if doc_id not in self._state.received_document_ids
        ]
        if missing_required:
            self._state.authorization_status = AuthorizationStatus.DENIED
            return (
                0.08,
                "Authorization submission failed because required documents are missing.",
                "The payer denied the request. The agent must resolve missing documentation before resubmitting.",
            )

        self._state.authorization_status = AuthorizationStatus.APPROVED
        return (
            0.0,
            "Authorization approved after complete documentation review.",
            "All required documents were available, so the submission earned approval.",
        )

    def _handle_schedule_visit(
        self, action: CareCoordAction
    ) -> tuple[float, str, str]:
        """Schedule an appointment slot for the selected provider."""
        if ActionType.SCHEDULE_VISIT not in self._case.allowed_actions:
            return self._invalid_action(
                feedback="Scheduling is not part of this task.",
                rationale="This task does not support appointment scheduling.",
                penalty=0.08,
            )

        slot = next((slot for slot in self._case.slots if slot.slot_id == action.slot_id), None)
        if slot is None:
            return self._invalid_action(
                feedback="Selected slot_id does not exist.",
                rationale="Choosing an unknown scheduling slot is invalid.",
                penalty=0.12,
            )

        if self._case.providers:
            if self._state.selected_provider_id is None:
                return self._invalid_action(
                    feedback="Select a provider before booking a slot.",
                    rationale="Scheduling before routing the case is penalized.",
                    penalty=0.10,
                )
            if action.provider_id != self._state.selected_provider_id:
                return self._invalid_action(
                    feedback="The slot provider does not match the currently selected provider.",
                    rationale="Scheduling must stay aligned with the chosen provider.",
                    penalty=0.12,
                )

        if slot.provider_id != action.provider_id:
            return self._invalid_action(
                feedback="The slot does not belong to the supplied provider_id.",
                rationale="Provider and slot identifiers must stay consistent.",
                penalty=0.12,
            )

        self._state.scheduled_slot_id = slot.slot_id

        if slot.slot_id not in self._case.acceptable_slot_ids:
            return (
                0.05,
                f"Booked {slot.date_label}, but it does not fully meet the task timing constraints.",
                "The action is recorded, but the grader will reserve full credit for compliant scheduling.",
            )

        return (
            0.0,
            f"Booked slot {slot.date_label}.",
            "The selected slot satisfies the task timing requirements.",
        )

    def _handle_arrange_transport(
        self, action: CareCoordAction
    ) -> tuple[float, str, str]:
        """Arrange transport for the follow-up visit."""
        if ActionType.ARRANGE_TRANSPORT not in self._case.allowed_actions:
            return self._invalid_action(
                feedback="Transport arrangement is not part of this task.",
                rationale="This task does not support transport actions.",
                penalty=0.08,
            )

        option = next(
            (
                transport
                for transport in self._case.transport_options
                if transport.transport_id == action.transport_id
            ),
            None,
        )
        if option is None:
            return self._invalid_action(
                feedback="Selected transport_id does not exist.",
                rationale="Choosing an unknown transport option is invalid.",
                penalty=0.10,
            )

        self._state.arranged_transport_id = option.transport_id

        if not option.covered or not option.accessible:
            return (
                0.05,
                f"Selected transport option '{option.label}', but it does not fully satisfy the case constraints.",
                "The action is operationally valid, yet the grader will reserve full credit for the covered accessible option.",
            )

        return (
            0.0,
            f"Arranged transport via {option.label}.",
            "The selected transport meets the patient mobility and coverage requirements.",
        )

    def _handle_escalation(
        self, action: CareCoordAction
    ) -> tuple[float, str, str]:
        """Escalate the case when a task requires it."""
        summary = (action.summary or "").strip()
        if not self._case.escalation_required:
            return self._invalid_action(
                feedback="This case does not require escalation.",
                rationale="Unnecessary escalation increases operational cost.",
                penalty=0.08,
            )

        if self._state.escalation_completed:
            return self._invalid_action(
                feedback="Escalation has already been completed for this case.",
                rationale="Repeated escalations do not add value.",
                penalty=0.03,
            )

        self._state.escalation_completed = True
        note = " Summary recorded." if summary else ""
        return (
            0.0,
            f"Escalation completed to the appropriate clinical support channel.{note}",
            "The required escalation step is now satisfied.",
        )

    def _handle_finalize_case(
        self, action: CareCoordAction
    ) -> tuple[float, str, str, bool]:
        """Finalize the case if all mandatory conditions are met."""
        required = self._case.resolution_code_required
        if action.resolution_code != required:
            return (
                0.10,
                f"This case should be finalized with resolution_code='{required.value}'.",
                "The proposed resolution code does not match the required outcome.",
                False,
            )

        if not self._ready_to_finalize():
            return (
                0.10,
                "Finalize failed because required task conditions are still incomplete.",
                "The environment requires the operational workflow to be completed before closure.",
                False,
            )

        self._state.final_resolution_code = action.resolution_code
        self._state.case_status = CaseStatus.COMPLETED
        return (
            0.0,
            f"Case finalized successfully with resolution '{action.resolution_code.value}'.",
            "The case meets all task requirements and is now complete.",
            True,
        )

    def _ready_to_finalize(self) -> bool:
        """Check if the active task is operationally complete."""
        if self._case.task_type == self._case.task_type.REFERRAL_ROUTING:
            return (
                self._state.selected_provider_id == self._case.correct_provider_id
                and self._state.scheduled_slot_id in self._case.acceptable_slot_ids
            )
        if self._case.task_type == self._case.task_type.PRIOR_AUTHORIZATION:
            return self._state.authorization_status == AuthorizationStatus.APPROVED
        return (
            self._state.escalation_completed
            and self._state.selected_provider_id == self._case.correct_provider_id
            and self._state.scheduled_slot_id in self._case.acceptable_slot_ids
            and self._state.arranged_transport_id == self._case.correct_transport_id
        )

    def _invalid_action(
        self, feedback: str, rationale: str, penalty: float
    ) -> tuple[float, str, str]:
        """Apply a consistent penalty for invalid or low-value actions."""
        self._state.invalid_action_count += 1
        return penalty, feedback, rationale

    def _format_action_history(self, action: CareCoordAction) -> str:
        """Create a compact history string for state inspection."""
        parts = [f"type={action.action_type.value}"]
        if action.section:
            parts.append(f"section={action.section.value}")
        if action.provider_id:
            parts.append(f"provider={action.provider_id}")
        if action.slot_id:
            parts.append(f"slot={action.slot_id}")
        if action.document_id:
            parts.append(f"document={action.document_id}")
        if action.transport_id:
            parts.append(f"transport={action.transport_id}")
        if action.resolution_code:
            parts.append(f"resolution={action.resolution_code.value}")
        return ", ".join(parts)
