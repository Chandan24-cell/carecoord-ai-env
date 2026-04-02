# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task catalog for CareCoordEnv."""

from dataclasses import dataclass

try:
    from .models import (
        ActionType,
        CaseDifficulty,
        CaseSection,
        ResolutionCode,
        TaskType,
    )
except ImportError:
    from models import (
        ActionType,
        CaseDifficulty,
        CaseSection,
        ResolutionCode,
        TaskType,
    )


@dataclass(frozen=True)
class ProviderRecord:
    """Internal provider record used to build task observations."""

    provider_id: str
    clinic_name: str
    specialty: str
    clinician_name: str
    clinician_gender: str
    in_network: bool
    accepts_new_patients: bool
    notes: str = ""


@dataclass(frozen=True)
class SlotRecord:
    """Internal scheduling slot record."""

    slot_id: str
    provider_id: str
    date_label: str
    days_from_today: int
    is_virtual: bool = False
    notes: str = ""


@dataclass(frozen=True)
class DocumentRecord:
    """Internal document record."""

    document_id: str
    label: str
    description: str
    initially_available: bool
    requestable: bool = False
    required_for_authorization: bool = False


@dataclass(frozen=True)
class TransportRecord:
    """Internal transport option record."""

    transport_id: str
    label: str
    covered: bool
    accessible: bool
    notes: str = ""


@dataclass(frozen=True)
class CaseSpec:
    """Single deterministic care-coordination case definition."""

    task_id: str
    task_type: TaskType
    title: str
    difficulty: CaseDifficulty
    objective: str
    objective_checklist: tuple[str, ...]
    patient_name: str
    patient_age: int
    patient_brief: str
    sections: dict[CaseSection, str]
    allowed_actions: tuple[ActionType, ...]
    providers: tuple[ProviderRecord, ...] = ()
    slots: tuple[SlotRecord, ...] = ()
    documents: tuple[DocumentRecord, ...] = ()
    transport_options: tuple[TransportRecord, ...] = ()
    correct_provider_id: str | None = None
    correct_specialty: str | None = None
    optimal_slot_id: str | None = None
    acceptable_slot_ids: tuple[str, ...] = ()
    required_document_ids: tuple[str, ...] = ()
    correct_transport_id: str | None = None
    escalation_required: bool = False
    resolution_code_required: ResolutionCode = ResolutionCode.UNABLE_TO_COMPLETE
    max_steps: int = 10


EASY_REFERRAL_CASE = CaseSpec(
    task_id="easy_referral_routing",
    task_type=TaskType.REFERRAL_ROUTING,
    title="Route a Dermatology Referral",
    difficulty=CaseDifficulty.EASY,
    objective=(
        "Route the patient to the correct in-network dermatology provider and "
        "book the earliest acceptable appointment within 14 days."
    ),
    objective_checklist=(
        "Review referral details and insurance rules",
        "Choose the best in-network dermatology provider",
        "Book the earliest acceptable slot",
        "Finalize the case as scheduled",
    ),
    patient_name="Riya Shah",
    patient_age=34,
    patient_brief=(
        "34-year-old with worsening eczema flare after urgent care visit. Needs "
        "outpatient dermatology follow-up and prefers a female clinician."
    ),
    sections={
        CaseSection.REFERRAL: (
            "Urgent care referral requests dermatology follow-up for eczema flare. "
            "Target timing: within 14 days. Female clinician preferred."
        ),
        CaseSection.INSURANCE: (
            "Insurance: Sunrise HMO Silver. Use in-network specialists only. "
            "Out-of-network referral will be denied."
        ),
        CaseSection.PROVIDERS: (
            "Provider directory loaded. Compare specialty, network status, and "
            "new-patient availability before routing."
        ),
        CaseSection.AVAILABILITY: (
            "Scheduling board loaded. Choose the earliest acceptable slot for the "
            "correct provider."
        ),
    },
    allowed_actions=(
        ActionType.REVIEW_CASE,
        ActionType.ROUTE_REFERRAL,
        ActionType.SCHEDULE_VISIT,
        ActionType.FINALIZE_CASE,
    ),
    providers=(
        ProviderRecord(
            provider_id="prov_derm_north",
            clinic_name="Northside Skin Clinic",
            specialty="Dermatology",
            clinician_name="Dr. Meera Iyer",
            clinician_gender="female",
            in_network=True,
            accepts_new_patients=True,
            notes="Adult dermatology, in network, female clinician.",
        ),
        ProviderRecord(
            provider_id="prov_allergy_central",
            clinic_name="Central Allergy Partners",
            specialty="Allergy",
            clinician_name="Dr. Aditya Nair",
            clinician_gender="male",
            in_network=True,
            accepts_new_patients=True,
            notes="In network but wrong specialty for this referral.",
        ),
        ProviderRecord(
            provider_id="prov_derm_out",
            clinic_name="Elite Derm Associates",
            specialty="Dermatology",
            clinician_name="Dr. Kavya Menon",
            clinician_gender="female",
            in_network=False,
            accepts_new_patients=True,
            notes="Correct specialty but out of network.",
        ),
    ),
    slots=(
        SlotRecord(
            slot_id="slot_north_3d",
            provider_id="prov_derm_north",
            date_label="2026-04-05 10:00",
            days_from_today=3,
        ),
        SlotRecord(
            slot_id="slot_north_10d",
            provider_id="prov_derm_north",
            date_label="2026-04-12 14:00",
            days_from_today=10,
        ),
        SlotRecord(
            slot_id="slot_allergy_4d",
            provider_id="prov_allergy_central",
            date_label="2026-04-06 09:30",
            days_from_today=4,
        ),
        SlotRecord(
            slot_id="slot_out_2d",
            provider_id="prov_derm_out",
            date_label="2026-04-04 11:15",
            days_from_today=2,
        ),
    ),
    correct_provider_id="prov_derm_north",
    correct_specialty="Dermatology",
    optimal_slot_id="slot_north_3d",
    acceptable_slot_ids=("slot_north_3d", "slot_north_10d"),
    resolution_code_required=ResolutionCode.SCHEDULED,
    max_steps=8,
)


MEDIUM_PRIOR_AUTH_CASE = CaseSpec(
    task_id="medium_prior_auth_recovery",
    task_type=TaskType.PRIOR_AUTHORIZATION,
    title="Recover a Denied MRI Prior Authorization",
    difficulty=CaseDifficulty.MEDIUM,
    objective=(
        "Recover a denied lumbar spine MRI prior authorization by identifying the "
        "missing document, requesting it, resubmitting, and closing the case as authorized."
    ),
    objective_checklist=(
        "Review the authorization requirements and document queue",
        "Request the missing clinical note",
        "Submit authorization only after all required documents are available",
        "Finalize the case as authorized",
    ),
    patient_name="Arjun Patel",
    patient_age=52,
    patient_brief=(
        "52-year-old with persistent lumbar radiculopathy after six weeks of "
        "conservative therapy. MRI was denied due to incomplete documentation."
    ),
    sections={
        CaseSection.REFERRAL: (
            "Ordering clinician requests MRI lumbar spine without contrast for "
            "persistent radicular pain and right leg numbness."
        ),
        CaseSection.INSURANCE: (
            "Payer: Meridian PPO. MRI requires prior authorization before scheduling. "
            "Claims are denied when the latest clinical note is missing."
        ),
        CaseSection.AUTH_REQUIREMENTS: (
            "Required items: imaging order and most recent clinical progress note. "
            "Conservative therapy summary is helpful but not mandatory."
        ),
        CaseSection.DOCUMENT_QUEUE: (
            "Document queue shows the imaging order already uploaded. The latest "
            "clinical progress note is missing."
        ),
    },
    allowed_actions=(
        ActionType.REVIEW_CASE,
        ActionType.REQUEST_DOCUMENT,
        ActionType.SUBMIT_AUTHORIZATION,
        ActionType.FINALIZE_CASE,
    ),
    documents=(
        DocumentRecord(
            document_id="doc_imaging_order",
            label="MRI imaging order",
            description="Signed lumbar spine MRI order from orthopedic clinic.",
            initially_available=True,
            required_for_authorization=True,
        ),
        DocumentRecord(
            document_id="doc_latest_clinical_note",
            label="Latest clinical progress note",
            description="Most recent clinic note documenting persistent symptoms.",
            initially_available=False,
            requestable=True,
            required_for_authorization=True,
        ),
        DocumentRecord(
            document_id="doc_physical_therapy_summary",
            label="Physical therapy summary",
            description="Six-week conservative therapy summary.",
            initially_available=True,
            required_for_authorization=False,
        ),
    ),
    required_document_ids=("doc_imaging_order", "doc_latest_clinical_note"),
    resolution_code_required=ResolutionCode.AUTHORIZED,
    max_steps=8,
)


HARD_DISCHARGE_CASE = CaseSpec(
    task_id="hard_post_discharge_followup",
    task_type=TaskType.POST_DISCHARGE_FOLLOWUP,
    title="Coordinate Post-Discharge Cardiology Follow-Up",
    difficulty=CaseDifficulty.HARD,
    objective=(
        "Escalate the worsening heart-failure symptoms, book an in-network "
        "cardiology follow-up within 7 days, arrange accessible covered transport, "
        "and close the case with the proper escalated resolution."
    ),
    objective_checklist=(
        "Review discharge, symptom, insurance, and transport information",
        "Escalate the symptom concern to the heart-failure nurse line",
        "Route to the correct in-network cardiology clinic",
        "Book follow-up within 7 days and arrange covered accessible transport",
        "Finalize the case as scheduled and escalated",
    ),
    patient_name="Mohan Verma",
    patient_age=68,
    patient_brief=(
        "68-year-old discharged after heart-failure exacerbation. Uses a walker, "
        "does not drive, and reports new overnight shortness of breath."
    ),
    sections={
        CaseSection.DISCHARGE_PLAN: (
            "Discharge instructions require cardiology follow-up within 7 days. "
            "Patient should avoid missed follow-up because of recent HF admission."
        ),
        CaseSection.INSURANCE: (
            "Insurance: BridgeCare Advantage. Use in-network cardiology and covered "
            "transport vendors only."
        ),
        CaseSection.PROVIDERS: (
            "Choose a cardiology clinic with heart-failure follow-up capacity and "
            "new-patient availability."
        ),
        CaseSection.AVAILABILITY: (
            "Scheduling board loaded. Appointments after day 7 do not meet the "
            "post-discharge requirement."
        ),
        CaseSection.TRANSPORT_OPTIONS: (
            "Patient needs accessible transport because family is unavailable and "
            "the patient uses a walker."
        ),
        CaseSection.SYMPTOM_ALERTS: (
            "Call note: patient reports 3-pound weight gain and waking up short of "
            "breath last night. Same-day nurse escalation is required while arranging follow-up."
        ),
    },
    allowed_actions=(
        ActionType.REVIEW_CASE,
        ActionType.ESCALATE_CASE,
        ActionType.ROUTE_REFERRAL,
        ActionType.SCHEDULE_VISIT,
        ActionType.ARRANGE_TRANSPORT,
        ActionType.FINALIZE_CASE,
    ),
    providers=(
        ProviderRecord(
            provider_id="prov_cardio_hf",
            clinic_name="City Heart Failure Clinic",
            specialty="Cardiology",
            clinician_name="Dr. Asha Raman",
            clinician_gender="female",
            in_network=True,
            accepts_new_patients=True,
            notes="Heart-failure clinic with post-discharge capacity.",
        ),
        ProviderRecord(
            provider_id="prov_cardio_out",
            clinic_name="Premier Cardiac Institute",
            specialty="Cardiology",
            clinician_name="Dr. Nitin Rao",
            clinician_gender="male",
            in_network=False,
            accepts_new_patients=True,
            notes="Correct specialty but out of network.",
        ),
        ProviderRecord(
            provider_id="prov_nephro_in",
            clinic_name="Regional Kidney Center",
            specialty="Nephrology",
            clinician_name="Dr. Farah Khan",
            clinician_gender="female",
            in_network=True,
            accepts_new_patients=True,
            notes="In network but wrong specialty.",
        ),
    ),
    slots=(
        SlotRecord(
            slot_id="slot_cardio_2d",
            provider_id="prov_cardio_hf",
            date_label="2026-04-04 09:00",
            days_from_today=2,
            notes="Earliest compliant cardiology follow-up.",
        ),
        SlotRecord(
            slot_id="slot_cardio_6d",
            provider_id="prov_cardio_hf",
            date_label="2026-04-08 13:30",
            days_from_today=6,
            notes="Still compliant but not earliest.",
        ),
        SlotRecord(
            slot_id="slot_cardio_9d",
            provider_id="prov_cardio_hf",
            date_label="2026-04-11 15:00",
            days_from_today=9,
            notes="Too late for post-discharge requirement.",
        ),
        SlotRecord(
            slot_id="slot_nephro_3d",
            provider_id="prov_nephro_in",
            date_label="2026-04-05 11:00",
            days_from_today=3,
            notes="Wrong specialty.",
        ),
    ),
    transport_options=(
        TransportRecord(
            transport_id="transport_rideshare",
            label="Standard rideshare voucher",
            covered=True,
            accessible=False,
            notes="Covered but not appropriate for walker assistance.",
        ),
        TransportRecord(
            transport_id="transport_wheelchair_van",
            label="Wheelchair-accessible van",
            covered=True,
            accessible=True,
            notes="Covered vendor with door-to-door assistance.",
        ),
        TransportRecord(
            transport_id="transport_family_friend",
            label="Family or friend ride",
            covered=False,
            accessible=True,
            notes="Family unavailable this week.",
        ),
    ),
    correct_provider_id="prov_cardio_hf",
    correct_specialty="Cardiology",
    optimal_slot_id="slot_cardio_2d",
    acceptable_slot_ids=("slot_cardio_2d", "slot_cardio_6d"),
    correct_transport_id="transport_wheelchair_van",
    escalation_required=True,
    resolution_code_required=ResolutionCode.SCHEDULED_AND_ESCALATED,
    max_steps=12,
)


TASK_SEQUENCE = (
    EASY_REFERRAL_CASE,
    MEDIUM_PRIOR_AUTH_CASE,
    HARD_DISCHARGE_CASE,
)

TASKS_BY_ID = {task.task_id: task for task in TASK_SEQUENCE}


def get_case(task_id: str | None = None) -> CaseSpec:
    """Return a case by id, or the default easy case when omitted."""
    if task_id is None:
        return TASK_SEQUENCE[0]
    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:
        valid = ", ".join(TASKS_BY_ID)
        raise ValueError(f"Unknown task_id '{task_id}'. Valid task ids: {valid}") from exc
