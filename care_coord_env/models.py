# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the CareCoordEnv healthcare environment."""

from enum import Enum
from typing import Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    """High-level case category."""

    REFERRAL_ROUTING = "referral_routing"
    PRIOR_AUTHORIZATION = "prior_authorization"
    POST_DISCHARGE_FOLLOWUP = "post_discharge_followup"


class CaseDifficulty(str, Enum):
    """Difficulty tier for a care-coordination task."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CaseStatus(str, Enum):
    """Lifecycle state for the current case."""

    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(str, Enum):
    """Allowed agent actions for the current environment."""

    REVIEW_CASE = "review_case"
    ROUTE_REFERRAL = "route_referral"
    REQUEST_DOCUMENT = "request_document"
    SUBMIT_AUTHORIZATION = "submit_authorization"
    SCHEDULE_VISIT = "schedule_visit"
    ARRANGE_TRANSPORT = "arrange_transport"
    ESCALATE_CASE = "escalate_case"
    FINALIZE_CASE = "finalize_case"


class CaseSection(str, Enum):
    """Reviewable case sections visible to the agent."""

    REFERRAL = "referral_details"
    INSURANCE = "insurance_rules"
    PROVIDERS = "provider_directory"
    AVAILABILITY = "scheduling_board"
    AUTH_REQUIREMENTS = "authorization_requirements"
    DOCUMENT_QUEUE = "document_queue"
    DISCHARGE_PLAN = "discharge_plan"
    TRANSPORT_OPTIONS = "transport_options"
    SYMPTOM_ALERTS = "symptom_alerts"


class ResolutionCode(str, Enum):
    """High-level resolution codes used when closing a case."""

    SCHEDULED = "scheduled"
    AUTHORIZED = "authorized"
    SCHEDULED_AND_ESCALATED = "scheduled_and_escalated"
    UNABLE_TO_COMPLETE = "unable_to_complete"


class AuthorizationStatus(str, Enum):
    """Authorization workflow status."""

    NOT_STARTED = "not_started"
    APPROVED = "approved"
    DENIED = "denied"


class DocumentStatus(str, Enum):
    """Availability state for a document in the task."""

    AVAILABLE = "available"
    RECEIVED = "received"
    MISSING = "missing"


class ProviderOption(BaseModel):
    """Provider directory entry shown to the agent."""

    provider_id: str = Field(..., description="Stable provider identifier")
    clinic_name: str = Field(..., description="Clinic or practice name")
    specialty: str = Field(..., description="Provider specialty")
    clinician_name: str = Field(..., description="Displayed clinician name")
    clinician_gender: str = Field(..., description="Clinician gender")
    in_network: bool = Field(..., description="Whether the provider is in network")
    accepts_new_patients: bool = Field(
        ..., description="Whether the provider is accepting new patients"
    )
    notes: str = Field(default="", description="Operational notes for the provider")


class AppointmentSlot(BaseModel):
    """Schedulable slot shown to the agent."""

    slot_id: str = Field(..., description="Stable slot identifier")
    provider_id: str = Field(..., description="Provider attached to the slot")
    date_label: str = Field(..., description="Human-readable appointment date")
    days_from_today: int = Field(..., description="Days until the slot")
    is_virtual: bool = Field(default=False, description="Whether the slot is virtual")
    notes: str = Field(default="", description="Additional slot notes")


class DocumentOption(BaseModel):
    """Document entry used in authorization workflows."""

    document_id: str = Field(..., description="Stable document identifier")
    label: str = Field(..., description="Human-readable document name")
    status: DocumentStatus = Field(..., description="Current document availability")
    required_for_authorization: bool = Field(
        default=False, description="Whether this document is mandatory for approval"
    )
    description: str = Field(default="", description="Short document description")


class TransportOption(BaseModel):
    """Transport option shown to the agent."""

    transport_id: str = Field(..., description="Stable transport identifier")
    label: str = Field(..., description="Transport option name")
    covered: bool = Field(..., description="Whether the transport is covered")
    accessible: bool = Field(..., description="Whether it fits the patient need")
    notes: str = Field(default="", description="Operational transport notes")


class ProgressSnapshot(BaseModel):
    """High-level progress summary surfaced in each observation."""

    reviewed_sections: list[CaseSection] = Field(default_factory=list)
    selected_provider_id: Optional[str] = Field(default=None)
    scheduled_slot_id: Optional[str] = Field(default=None)
    requested_document_ids: list[str] = Field(default_factory=list)
    received_document_ids: list[str] = Field(default_factory=list)
    authorization_status: AuthorizationStatus = Field(
        default=AuthorizationStatus.NOT_STARTED
    )
    arranged_transport_id: Optional[str] = Field(default=None)
    escalation_completed: bool = Field(default=False)
    ready_to_finalize: bool = Field(default=False)
    invalid_action_count: int = Field(default=0, ge=0)


class CareCoordAction(Action):
    """Typed agent action for healthcare care coordination."""

    action_type: ActionType = Field(..., description="Operation to perform")
    section: Optional[CaseSection] = Field(
        default=None, description="Case section to review"
    )
    specialty: Optional[str] = Field(
        default=None, description="Specialty selected during referral routing"
    )
    provider_id: Optional[str] = Field(
        default=None, description="Chosen provider identifier"
    )
    slot_id: Optional[str] = Field(default=None, description="Chosen appointment slot")
    document_id: Optional[str] = Field(
        default=None, description="Requested document identifier"
    )
    transport_id: Optional[str] = Field(
        default=None, description="Chosen transport option identifier"
    )
    resolution_code: Optional[ResolutionCode] = Field(
        default=None, description="Case resolution used during finalization"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Short case note or rationale supplied by the agent",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "CareCoordAction":
        """Enforce action-specific required fields."""
        if self.action_type == ActionType.REVIEW_CASE and self.section is None:
            raise ValueError("review_case requires `section`")
        if self.action_type == ActionType.ROUTE_REFERRAL:
            if self.provider_id is None or self.specialty is None:
                raise ValueError("route_referral requires `provider_id` and `specialty`")
        if self.action_type == ActionType.REQUEST_DOCUMENT and self.document_id is None:
            raise ValueError("request_document requires `document_id`")
        if self.action_type == ActionType.SCHEDULE_VISIT:
            if self.provider_id is None or self.slot_id is None:
                raise ValueError("schedule_visit requires `provider_id` and `slot_id`")
        if self.action_type == ActionType.ARRANGE_TRANSPORT and self.transport_id is None:
            raise ValueError("arrange_transport requires `transport_id`")
        if self.action_type == ActionType.FINALIZE_CASE:
            if self.resolution_code is None or not self.summary:
                raise ValueError(
                    "finalize_case requires `resolution_code` and a non-empty `summary`"
                )
        return self


class CareCoordObservation(Observation):
    """Observation returned after each step in the care coordination workflow."""

    task_id: str = Field(..., description="Unique identifier for the active task")
    task_title: str = Field(..., description="Human-readable task title")
    task_type: TaskType = Field(..., description="Task category")
    difficulty: CaseDifficulty = Field(..., description="Difficulty tier")
    current_status: CaseStatus = Field(..., description="Current case lifecycle state")
    objective: str = Field(..., description="Primary case objective")
    objective_checklist: list[str] = Field(
        default_factory=list, description="Checklist describing the expected outcome"
    )
    patient_name: str = Field(..., description="Patient name")
    patient_age: int = Field(..., ge=0, description="Patient age")
    patient_brief: str = Field(..., description="Short patient summary")
    allowed_action_types: list[ActionType] = Field(
        default_factory=list, description="Valid actions for the current task"
    )
    available_sections: list[CaseSection] = Field(
        default_factory=list, description="Sections that can be reviewed"
    )
    visible_sections: list[CaseSection] = Field(
        default_factory=list, description="Sections already revealed to the agent"
    )
    visible_section_content: dict[str, str] = Field(
        default_factory=dict, description="Text content for revealed sections"
    )
    provider_options: list[ProviderOption] = Field(
        default_factory=list, description="Visible provider options"
    )
    available_slots: list[AppointmentSlot] = Field(
        default_factory=list, description="Visible scheduling slots"
    )
    documents: list[DocumentOption] = Field(
        default_factory=list, description="Visible documents and their current state"
    )
    authorization_status: AuthorizationStatus = Field(
        default=AuthorizationStatus.NOT_STARTED,
        description="Current prior authorization status",
    )
    transport_options: list[TransportOption] = Field(
        default_factory=list, description="Visible transport options"
    )
    blockers: list[str] = Field(
        default_factory=list, description="Outstanding blockers to complete the task"
    )
    progress: ProgressSnapshot = Field(..., description="Structured progress summary")
    current_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Current grader score"
    )
    score_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-component grader scores contributing to the total score",
    )
    last_action_feedback: str = Field(
        default="", description="Environment feedback for the most recent action"
    )
    reward_rationale: str = Field(
        default="", description="Short explanation for the latest reward value"
    )


class CareCoordState(State):
    """Internal environment state exposed via `state()`."""

    task_id: str = Field(..., description="Unique task identifier")
    task_title: str = Field(..., description="Current task title")
    task_type: TaskType = Field(..., description="Current task category")
    difficulty: CaseDifficulty = Field(..., description="Current task difficulty")
    case_status: CaseStatus = Field(default=CaseStatus.READY)
    max_steps: int = Field(default=8, ge=1, description="Episode step limit")
    revealed_sections: list[CaseSection] = Field(default_factory=list)
    selected_provider_id: Optional[str] = Field(default=None)
    scheduled_slot_id: Optional[str] = Field(default=None)
    requested_document_ids: list[str] = Field(default_factory=list)
    received_document_ids: list[str] = Field(default_factory=list)
    authorization_status: AuthorizationStatus = Field(
        default=AuthorizationStatus.NOT_STARTED
    )
    arranged_transport_id: Optional[str] = Field(default=None)
    escalation_completed: bool = Field(default=False)
    invalid_action_count: int = Field(default=0, ge=0)
    action_history: list[str] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    final_resolution_code: Optional[ResolutionCode] = Field(default=None)
    last_action_feedback: str = Field(default="")
    last_reward_rationale: str = Field(default="")
