"""Main entry point - run EAGLE router experiment."""

import argparse
from collections import Counter
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from dotenv import load_dotenv

from data_access import get_sampled_splits
from elo import RatingPair
from embeddings import (
    DEFAULT_EMBEDDING_BACKEND,
    EMBEDDING_MODELS,
    EmbeddingBackend,
    get_embeddings,
)
from evaluation import compute_accuracy, compute_baseline_accuracy, find_best_setting
from match_types import MatchRow, Outcome
from parameters import (
    DEFAULT_K,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_P,
    DEFAULT_SEED,
    DEFAULT_TEST_N,
    DEFAULT_TRAIN_N,
    DEFAULT_VAL_N,
    P_VALUES,
    TIE_THRESHOLD_VALUES,
)
from router import EagleRouter, build_train_index, compute_global_ratings
from similarity import TrainIndex

# Builds a router over the training data from a P and a tie_threshold
RouterFactory = Callable[[float, float], EagleRouter]

# The tuning table prints threshold 0 and every 10th threshold after it; the
# search still covers all of them
DISPLAYED_THRESHOLD_INDICES = [0, *range(1, len(TIE_THRESHOLD_VALUES), 10)]


class TunedSetting(NamedTuple):
    """A P and tie_threshold chosen on validation, with its validation accuracy."""

    p: float
    tie_threshold: float
    val_accuracy: float


class TuningResult(NamedTuple):
    """The settings chosen on validation, and the baseline they're compared to."""

    val_accuracy_always_tie: float
    best_without_threshold: TunedSetting
    best: TunedSetting

    @property
    def beats_always_tie(self) -> bool:
        """Whether the best setting beat always predicting a tie on validation."""
        return self.best.val_accuracy > self.val_accuracy_always_tie


class TestAccuracies(NamedTuple):
    """Test accuracy of the baseline, the paper's settings and both tuned settings."""

    always_tie: float
    paper: float
    tuned_without_threshold: float
    tuned: float


def run_experiment(
    seed: int = DEFAULT_SEED,
    embedding_backend: EmbeddingBackend = DEFAULT_EMBEDDING_BACKEND,
    train_n: int = DEFAULT_TRAIN_N,
    val_n: int = DEFAULT_VAL_N,
    test_n: int = DEFAULT_TEST_N,
) -> None:
    """Run the full EAGLE experiment, printing each step's results.

    Args:
        seed: Random seed for sampling match rows
        embedding_backend: Which embedding model to use
        train_n: Number of training match rows
        val_n: Number of validation match rows
        test_n: Number of test match rows
    """
    train_match_rows, val_match_rows, test_match_rows = _load_match_rows(
        seed, train_n, val_n, test_n
    )
    train_embeddings, val_embeddings, test_embeddings = _embed_prompts(
        train_match_rows, val_match_rows, test_match_rows, embedding_backend
    )
    global_ratings, train_index = _build_shared_router_inputs(
        train_match_rows, train_embeddings
    )

    def make_router(p: float, tie_threshold: float) -> EagleRouter:
        """Router over the training data; only P and tie_threshold vary."""
        return EagleRouter(
            global_ratings,
            train_index,
            k=DEFAULT_K,
            n_neighbors=DEFAULT_N_NEIGHBORS,
            p=p,
            tie_threshold=tie_threshold,
        )

    tuning = _tune_on_validation(make_router, val_match_rows, val_embeddings)
    test_accuracies = _evaluate_on_test(
        make_router, tuning, test_match_rows, test_embeddings
    )
    _print_results(tuning, test_accuracies)


def _load_match_rows(
    seed: int, train_n: int, val_n: int, test_n: int
) -> tuple[list[MatchRow], list[MatchRow], list[MatchRow]]:
    """Step 1: load SPROUT and sample each split to its requested size."""
    print("Loading and sampling data...")
    train_match_rows, val_match_rows, test_match_rows = get_sampled_splits(
        train_n=train_n, val_n=val_n, test_n=test_n, seed=seed
    )
    print(
        f"  Train: {len(train_match_rows)}, Val: {len(val_match_rows)}, "
        f"Test: {len(test_match_rows)}"
    )

    train_outcomes = Counter(match_row.outcome for match_row in train_match_rows)
    print(f"  Train outcomes: {dict(train_outcomes)}")

    return train_match_rows, val_match_rows, test_match_rows


def _embed_prompts(
    train_match_rows: list[MatchRow],
    val_match_rows: list[MatchRow],
    test_match_rows: list[MatchRow],
    embedding_backend: EmbeddingBackend,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 2: embed every prompt; each array has shape (n_match_rows, dim)."""
    print(f"\nEmbedding prompts with {EMBEDDING_MODELS[embedding_backend]}...")
    train_prompts = [match_row.prompt for match_row in train_match_rows]
    val_prompts = [match_row.prompt for match_row in val_match_rows]
    test_prompts = [match_row.prompt for match_row in test_match_rows]

    train_embeddings = get_embeddings(train_prompts, embedding_backend)
    val_embeddings = get_embeddings(val_prompts, embedding_backend)
    test_embeddings = get_embeddings(test_prompts, embedding_backend)
    print(f"  Embedding dim: {train_embeddings.shape[1]}")

    return train_embeddings, val_embeddings, test_embeddings


def _build_shared_router_inputs(
    train_match_rows: list[MatchRow],
    train_embeddings: np.ndarray,
) -> tuple[RatingPair, TrainIndex]:
    """Step 3: build the global ratings and train index every router shares."""
    print("\nBuilding router...")
    global_ratings = compute_global_ratings(train_match_rows, k=DEFAULT_K)
    train_index = build_train_index(train_match_rows, train_embeddings)
    print(f"  Global ratings: A={global_ratings.a:.2f}, B={global_ratings.b:.2f}")

    return global_ratings, train_index


def _tune_on_validation(
    make_router: RouterFactory,
    val_match_rows: list[MatchRow],
    val_embeddings: np.ndarray,
) -> TuningResult:
    """Step 4: search every P and tie_threshold, printing validation accuracy.

    Picks the best P with no tie threshold, and the best P and tie_threshold
    together.
    """
    print("\nTuning P and tie_threshold on validation set...")
    print(
        f"  Searching {len(P_VALUES)} P values and "
        f"{len(TIE_THRESHOLD_VALUES)} tie thresholds "
        f"({len(P_VALUES) * len(TIE_THRESHOLD_VALUES):,} settings)"
    )

    val_accuracy_always_tie = compute_baseline_accuracy(val_match_rows, Outcome.TIE)
    print(f"  Always predicting tie: {val_accuracy_always_tie:.2%}")

    # One row per P, with selected thresholds from the full search grid
    print("  Validation accuracy by P (rows) and selected tie_thresholds (columns):")
    print(
        "     P"
        + "".join(
            f"{TIE_THRESHOLD_VALUES[i]:7.4g}" for i in DISPLAYED_THRESHOLD_INDICES
        )
    )

    val_accuracy_rows = []
    for p in P_VALUES:
        row_accuracies = [
            compute_accuracy(
                make_router(p, tie_threshold), val_match_rows, val_embeddings
            )
            for tie_threshold in TIE_THRESHOLD_VALUES
        ]
        val_accuracy_rows.append(row_accuracies)
        print(
            f"  {p:4.2f}"
            + "".join(f"{row_accuracies[i]:7.1%}" for i in DISPLAYED_THRESHOLD_INDICES)
        )
    val_accuracies = np.array(val_accuracy_rows)  # (n_p_values, n_tie_thresholds)

    # Column 0 is tie_threshold 0, i.e. no tie threshold
    row, _ = find_best_setting(val_accuracies[:, :1])
    best_without_threshold = TunedSetting(
        P_VALUES[row], TIE_THRESHOLD_VALUES[0], val_accuracies[row, 0]
    )
    row, column = find_best_setting(val_accuracies)
    best = TunedSetting(
        P_VALUES[row], TIE_THRESHOLD_VALUES[column], val_accuracies[row, column]
    )

    print(f"  Best P with no tie threshold: {best_without_threshold.p:.2f}")
    print(
        "  Validation accuracy (no tie threshold): "
        f"{best_without_threshold.val_accuracy:.2%}"
    )
    print(
        f"  Best P and tie_threshold: P={best.p:.2f}, "
        f"tie_threshold={best.tie_threshold:.6g}"
    )
    print(f"  Validation accuracy (P and tie_threshold): {best.val_accuracy:.2%}")

    tuning = TuningResult(val_accuracy_always_tie, best_without_threshold, best)
    if not tuning.beats_always_tie:
        print("  No setting beat always predicting a tie")

    return tuning


def _evaluate_on_test(
    make_router: RouterFactory,
    tuning: TuningResult,
    test_match_rows: list[MatchRow],
    test_embeddings: np.ndarray,
) -> TestAccuracies:
    """Step 5: compare the baseline, paper settings and tuned settings on test."""
    print("\nEvaluating on SPROUT test set...")

    # Baseline: always predict a tie, the most common outcome
    always_tie = compute_baseline_accuracy(test_match_rows, Outcome.TIE)
    print(f"  Always predicting tie: {always_tie:.2%}")

    # Paper settings, evaluated here on SPROUT
    router_paper = make_router(DEFAULT_P, 0.0)
    paper = compute_accuracy(router_paper, test_match_rows, test_embeddings)
    print(f"  Paper P={DEFAULT_P}, no tie threshold: {paper:.2%}")

    # P tuned on validation, with no tie threshold
    setting = tuning.best_without_threshold
    router_tuned_without_threshold = make_router(setting.p, setting.tie_threshold)
    tuned_without_threshold = compute_accuracy(
        router_tuned_without_threshold, test_match_rows, test_embeddings
    )
    print(f"  Tuned P={setting.p:.2f}, no tie threshold: {tuned_without_threshold:.2%}")

    # P and tie_threshold tuned on validation
    setting = tuning.best
    router_tuned = make_router(setting.p, setting.tie_threshold)
    tuned = compute_accuracy(router_tuned, test_match_rows, test_embeddings)
    print(
        f"  Tuned P={setting.p:.2f}, tie_threshold={setting.tie_threshold:.6g}: "
        f"{tuned:.2%}"
    )

    return TestAccuracies(always_tie, paper, tuned_without_threshold, tuned)


def _print_results(tuning: TuningResult, test_accuracies: TestAccuracies) -> None:
    """Print a summary of the validation choices and test accuracies."""
    best_without_threshold, best = tuning.best_without_threshold, tuning.best

    print("\n" + "=" * 40)
    print("RESULTS (SPROUT)")
    print("=" * 40)
    print(f"Test accuracy (always tie): {test_accuracies.always_tie:.2%}")
    print(
        f"Test accuracy (paper P={DEFAULT_P}, no tie threshold): "
        f"{test_accuracies.paper:.2%}"
    )
    print(f"Tuned P with no tie threshold: {best_without_threshold.p:.2f}")
    print(
        "Validation accuracy (no tie threshold): "
        f"{best_without_threshold.val_accuracy:.2%}"
    )
    print(
        "Test accuracy (tuned P, no tie threshold): "
        f"{test_accuracies.tuned_without_threshold:.2%}"
    )
    print(
        f"Tuned P and tie_threshold: P={best.p:.2f}, "
        f"tie_threshold={best.tie_threshold:.6g}"
    )
    print(f"Validation accuracy (P and tie_threshold): {best.val_accuracy:.2%}")
    print(f"Test accuracy (tuned P and tie_threshold): {test_accuracies.tuned:.2%}")
    if not tuning.beats_always_tie:
        print("No setting beat always predicting a tie on validation")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the EAGLE router experiment.")
    parser.add_argument(
        "--embedding-backend",
        choices=list(EMBEDDING_MODELS),
        default=DEFAULT_EMBEDDING_BACKEND,
        help="Embedding backend (default: %(default)s, runs locally).",
    )
    parser.add_argument(
        "--train-n",
        type=int,
        default=DEFAULT_TRAIN_N,
        help="Training sample size (default: %(default)s).",
    )
    parser.add_argument(
        "--val-n",
        type=int,
        default=DEFAULT_VAL_N,
        help="Validation sample size (default: %(default)s).",
    )
    parser.add_argument(
        "--test-n",
        type=int,
        default=DEFAULT_TEST_N,
        help="Test sample size (default: %(default)s).",
    )
    args = parser.parse_args()
    run_experiment(
        embedding_backend=args.embedding_backend,
        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,
    )
