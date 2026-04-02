"""Baseline inference script for CareCoordEnv."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from care_coord_env import CareCoordAction, CareCoordEnv, TASK_SEQUENCE

BENCHMARK = "care_coord_env"
DEFAULT_IMAGE_NAME = "care_coord_env-env:latest"
MAX_STEPS_BY_TASK = {
    "easy_referral_routing": 8,
    "medium_prior_auth_recovery": 8,
    "hard_post_discharge_followup": 12,
}


def require_env(name: str) -> str:
    """Read an environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_client() -> OpenAI:
    """Build the OpenAI client using hackathon-required environment variables."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or HF_TOKEN before running inference.py")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    action_json = json.dumps(action, sort_keys=True, ensure_ascii=False)
    error_text = "null" if error is None else json.dumps(error, ensure_ascii=False)
    print(
        f"[STEP] step={step} action={action_json} reward={reward:.4f} done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_json = json.dumps([round(r, 4) for r in rewards], ensure_ascii=False)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={reward_json}",
        flush=True,
    )


def build_prompt(observation: dict[str, Any], history: list[str]) -> str:
    """Build the action-selection prompt for the model."""
    return f"""
You are operating a healthcare care-coordination environment.
Return exactly one JSON object with the next action.

Rules:
- Use only one action from allowed_action_types.
- Choose IDs exactly as shown in the observation.
- Prefer the action that improves task completion fastest.
- Do not add markdown fences or extra commentary.

Action schema:
{{
  "action_type": "<string>",
  "section": "<optional string>",
  "specialty": "<optional string>",
  "provider_id": "<optional string>",
  "slot_id": "<optional string>",
  "document_id": "<optional string>",
  "transport_id": "<optional string>",
  "resolution_code": "<optional string>",
  "summary": "<optional string>"
}}

Recent history:
{json.dumps(history[-6:], indent=2, ensure_ascii=False)}

Current observation:
{json.dumps(observation, indent=2, ensure_ascii=False)}
""".strip()


def get_model_action(client: OpenAI, model_name: str, observation: dict[str, Any], history: list[str]) -> dict[str, Any]:
    """Call the model and parse the returned JSON action."""
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a careful healthcare operations agent. Output valid JSON only.",
            },
            {
                "role": "user",
                "content": build_prompt(observation, history),
            },
        ],
    )
    content = completion.choices[0].message.content or "{}"
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Model response was not valid JSON: {text}")
        return json.loads(text[start : end + 1])


async def run_task(env: CareCoordEnv, client: OpenAI, model_name: str, task_id: str) -> dict[str, Any]:
    """Run one task from reset to completion."""
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    result = await env.reset(task_id=task_id)
    max_steps = MAX_STEPS_BY_TASK[task_id]

    for step in range(1, max_steps + 1):
        if result.done:
            break

        observation_payload = result.observation.model_dump(mode="json")
        error = None
        try:
            action_payload = get_model_action(client, model_name, observation_payload, history)
            action = CareCoordAction.model_validate(action_payload)
            result = await env.step(action)
        except Exception as exc:
            action_payload = {"action_type": "error"}
            result = await env.step(
                CareCoordAction(
                    action_type="finalize_case",
                    resolution_code="unable_to_complete",
                    summary="Model produced an invalid action payload.",
                )
            )
            error = str(exc)

        reward = float(result.reward or 0.0)
        rewards.append(reward)
        steps_taken = step
        history.append(
            json.dumps(
                {
                    "step": step,
                    "action": action_payload,
                    "reward": reward,
                    "feedback": result.observation.last_action_feedback,
                },
                ensure_ascii=False,
            )
        )
        log_step(step=step, action=action_payload, reward=reward, done=result.done, error=error)

        if result.done:
            break

    final_state = await env.state()
    score = float(final_state.current_score)
    success = score >= 0.85
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}


async def main() -> None:
    """Run the baseline model across all three benchmark tasks."""
    model_name = require_env("MODEL_NAME")
    require_env("HF_TOKEN")
    client = build_client()

    image_name = os.getenv("IMAGE_NAME", DEFAULT_IMAGE_NAME)
    env = await CareCoordEnv.from_docker_image(image_name)
    try:
        results = []
        for task in TASK_SEQUENCE:
            results.append(await run_task(env, client, model_name, task.task_id))

        average_score = sum(item["score"] for item in results) / len(results)
        print(
            json.dumps(
                {
                    "benchmark": BENCHMARK,
                    "model": model_name,
                    "average_score": round(average_score, 4),
                    "results": results,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
