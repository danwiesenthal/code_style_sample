"""Match types shared across modules: one prompt scored for both models."""

from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    """Result of a match between models A and B."""

    WIN_A = "win_a"
    WIN_B = "win_b"
    TIE = "tie"


@dataclass(frozen=True)
class MatchRow:
    """One prompt and both models' scores for their responses to it.

    Attributes:
        row_id: Dataset key for the prompt
        prompt: The prompt both models answered
        score_a: Judge's score for model A's response
        score_b: Judge's score for model B's response
    """

    row_id: str
    prompt: str
    score_a: float
    score_b: float

    @property
    def outcome(self) -> Outcome:
        """Whichever model scored higher wins; equal scores are a tie."""
        if self.score_a > self.score_b:
            return Outcome.WIN_A
        elif self.score_b > self.score_a:
            return Outcome.WIN_B
        return Outcome.TIE
