# src/zeroband/sanskrit_poetry_env.py

from typing import Dict, Any, Tuple
from datasets import load_dataset
from chandas import to_pattern_lines, svat_identifier

class SanskritPoetryEnv:
    def __init__(
        self,
        dataset_name: str = "nippun-live/sanskrit-poetry-requests",
        max_steps: int = 10
    ):
        # Load the HF dataset of (topic, meter, request)
        self.dataset = load_dataset(dataset_name)
        self.max_steps = max_steps
        self.current_idx = 0
        self.current_request = None
        self.reset()

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Start a new episode, return (request, info)."""
        self.current_idx = 0
        self.current_request = self.dataset["train"][self.current_idx]
        return (
            self.current_request["request"],
            {"expected_meter": self.current_request["meter"]}
        )

    def step(self, poem: str) -> Tuple[float, bool, Dict[str, Any]]:
        lines = poem.strip().split("\n")
        pattern_lines = to_pattern_lines(lines)

        id_result = svat_identifier.IdentifyFromPatternLines(pattern_lines)
        exact_set = id_result.get("exact", [])

        if exact_set:
            first = next(iter(exact_set))
            if isinstance(first, tuple):
                identified = first[0]
            else:
                identified = first
        else:
            identified = None

        expected = self.current_request["meter"]
        reward = 1.0 if (identified == expected) else -1.0

        self.current_idx += 1
        done = (
            self.current_idx >= len(self.dataset["train"])
            or self.current_idx >= self.max_steps
        )

        info = {"expected_meter": expected, "identified_meter": identified}
        if not done:
            self.current_request = self.dataset["train"][self.current_idx]

        return reward, done, info



    def get_current_request(self) -> str:
        """Return the prompt (topic+meter) for the current step."""
        return self.current_request["request"]
