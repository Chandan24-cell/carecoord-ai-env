# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the CareCoordEnv healthcare environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import CareCoordAction, CareCoordObservation, CareCoordState


class CareCoordEnv(EnvClient[CareCoordAction, CareCoordObservation, CareCoordState]):
    """
    Client for the CareCoordEnv environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CareCoordEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_title)
        ...
        ...     result = client.reset(task_id="easy_referral_routing")
        ...     result = client.step(
        ...         CareCoordAction(action_type="review_case", section="referral_details")
        ...     )
        ...     print(result.observation.last_action_feedback)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CareCoordEnv.from_docker_image("care_coord_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.reset(task_id="easy_referral_routing")
        ...     result = client.step(
        ...         CareCoordAction(action_type="review_case", section="referral_details")
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CareCoordAction) -> Dict:
        """
        Convert CareCoordAction to JSON payload for step message.

        Args:
            action: CareCoordAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[CareCoordObservation]:
        """
        Parse server response into StepResult[CareCoordObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CareCoordObservation
        """
        obs_data = dict(payload.get("observation", {}))
        observation = CareCoordObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CareCoordState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return CareCoordState.model_validate(payload)
