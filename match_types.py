"""Match types shared across modules: one prompt scored for both models."""

from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    """Result of a match between models A and B."""

    WIN_A = "win_a"
    WIN_B = "win_b"
    TIE = "tie"


@dataclass
class MatchRow:
    """One prompt and both models' scores for their responses to it."""

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
