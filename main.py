"""Main entry point - run EAGLE router experiment."""

import argparse
from collections import Counter

import numpy as np
from dotenv import load_dotenv

from data_access import (
    DEFAULT_TEST_N,
    DEFAULT_TRAIN_N,
    DEFAULT_VAL_N,
    get_sampled_splits,
)
from embeddings import EMBEDDING_MODELS, EmbeddingBackend, get_embeddings
from evaluation import compute_accuracy, compute_baseline_accuracy
from match_types import Outcome
from ranker import DEFAULT_K
from router import (
    DEFAULT_N,
    DEFAULT_P,
    EagleRouter,
    build_train_index,
    compute_global_ratings,
)


def run_experiment(
    seed: int = 42,
    embedding_backend: EmbeddingBackend = "qwen",
    train_n: int = DEFAULT_TRAIN_N,
    val_n: int = DEFAULT_VAL_N,
    test_n: int = DEFAULT_TEST_N,
) -> None:
    """Run full EAGLE experiment.

    Steps:
        1. Load and sample data to the requested split sizes
        2. Embed all prompts
        3. Build router from training data
        4. Tune P with no tie threshold and P with tie_threshold on validation
        5. Report SPROUT test accuracy for always predicting a tie, paper settings,
           tuned P with no tie threshold, and tuned P and tie_threshold
    """

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("Loading and sampling data...")
    train_rows, val_rows, test_rows = get_sampled_splits(
        train_n=train_n, val_n=val_n, test_n=test_n, seed=seed
    )
    print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    train_outcomes = Counter(r.outcome for r in train_rows)
    print(f"  Train outcomes: {dict(train_outcomes)}")

    # -------------------------------------------------------------------------
    # 2. Embed prompts
    # -------------------------------------------------------------------------
    print(f"\nEmbedding prompts with {EMBEDDING_MODELS[embedding_backend]}...")
    train_emb = get_embeddings([r.prompt for r in train_rows], embedding_backend)
    val_emb = get_embeddings([r.prompt for r in val_rows], embedding_backend)
    test_emb = get_embeddings([r.prompt for r in test_rows], embedding_backend)
    print(f"  Embedding dim: {train_emb.shape[1]}")

    # -------------------------------------------------------------------------
    # 3. Build router from training data
    # -------------------------------------------------------------------------
    print("\nBuilding router...")
    global_ratings = compute_global_ratings(train_rows, k=DEFAULT_K)
    train_index = build_train_index(train_rows, train_emb)
    print(f"  Global ratings: A={global_ratings.a:.2f}, B={global_ratings.b:.2f}")

    def make_router(p: float, tie_threshold: float) -> EagleRouter:
        """Router over the training data; only P and tie_threshold vary."""
        return EagleRouter(
            global_ratings,
            train_index,
            k=DEFAULT_K,
            n_neighbors=DEFAULT_N,
            p=p,
            tie_threshold=tie_threshold,
        )

    # -------------------------------------------------------------------------
    # 4. Tune P and tie_threshold on validation set
    # -------------------------------------------------------------------------
    print("\nTuning P and tie_threshold on validation set...")
    p_values = np.linspace(0, 1, 21)  # 0.0, 0.05, ..., 1.0
    tie_threshold_values = [0.0, *np.geomspace(1, 1000, 61)]
    displayed_threshold_indices = [0, *range(1, len(tie_threshold_values), 10)]
    print(
        f"  Searching {len(p_values)} P values and "
        f"{len(tie_threshold_values)} tie thresholds "
        f"({len(p_values) * len(tie_threshold_values):,} settings)"
    )

    val_acc_always_tie = compute_baseline_accuracy(val_rows, Outcome.TIE)
    print(f"  Always predicting tie: {val_acc_always_tie:.2%}")

    # One row per P, with selected thresholds from the full search grid
    print("  Validation accuracy by P (rows) and selected tie_thresholds (columns):")
    print(
        "     P"
        + "".join(
            f"{tie_threshold_values[i]:7.4g}" for i in displayed_threshold_indices
        )
    )

    val_acc_rows = []
    for p in p_values:
        row_accs = [
            compute_accuracy(make_router(p, tie_threshold), val_rows, val_emb)
            for tie_threshold in tie_threshold_values
        ]
        val_acc_rows.append(row_accs)
        print(
            f"  {p:4.2f}"
            + "".join(f"{row_accs[i]:7.1%}" for i in displayed_threshold_indices)
        )
    val_accs = np.array(val_acc_rows)  # (n_p_values, n_tie_thresholds)

    # Best P with no tie threshold (column 0), then best P and tie_threshold.
    # argmax keeps the first best, in the order the grid was searched.
    i = np.argmax(val_accs[:, 0])
    best_p_no_threshold, best_val_acc_no_threshold = p_values[i], val_accs[i, 0]
    i, j = np.unravel_index(np.argmax(val_accs), val_accs.shape)
    best_p, best_tie_threshold = p_values[i], tie_threshold_values[j]
    best_val_acc = val_accs[i, j]

    print(f"  Best P with no tie threshold: {best_p_no_threshold:.2f}")
    print(f"  Validation accuracy (no tie threshold): {best_val_acc_no_threshold:.2%}")
    print(
        f"  Best P and tie_threshold: P={best_p:.2f}, "
        f"tie_threshold={best_tie_threshold:.6g}"
    )
    print(f"  Validation accuracy (P and tie_threshold): {best_val_acc:.2%}")
    if best_val_acc <= val_acc_always_tie:
        print("  No setting beat always predicting a tie")

    # -------------------------------------------------------------------------
    # 5. Evaluate on test set
    # -------------------------------------------------------------------------
    print("\nEvaluating on SPROUT test set...")

    # Baseline: always predict a tie, the most common outcome
    test_acc_always_tie = compute_baseline_accuracy(test_rows, Outcome.TIE)
    print(f"  Always predicting tie: {test_acc_always_tie:.2%}")

    # Paper settings, evaluated here on SPROUT
    router_paper = make_router(DEFAULT_P, tie_threshold=0.0)
    test_acc_paper = compute_accuracy(router_paper, test_rows, test_emb)
    print(f"  Paper P={DEFAULT_P}, no tie threshold: {test_acc_paper:.2%}")

    # P tuned on validation, with no tie threshold
    router_tuned_no_threshold = make_router(best_p_no_threshold, tie_threshold=0.0)
    test_acc_tuned_no_threshold = compute_accuracy(
        router_tuned_no_threshold, test_rows, test_emb
    )
    print(
        f"  Tuned P={best_p_no_threshold:.2f}, no tie threshold: "
        f"{test_acc_tuned_no_threshold:.2%}"
    )

    # P and tie_threshold tuned on validation
    router_tuned = make_router(best_p, best_tie_threshold)
    test_acc_tuned = compute_accuracy(router_tuned, test_rows, test_emb)
    print(
        f"  Tuned P={best_p:.2f}, tie_threshold={best_tie_threshold:.6g}: "
        f"{test_acc_tuned:.2%}"
    )

    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("RESULTS (SPROUT)")
    print("=" * 40)
    print(f"Test accuracy (always tie): {test_acc_always_tie:.2%}")
    print(
        f"Test accuracy (paper P={DEFAULT_P}, no tie threshold): {test_acc_paper:.2%}"
    )
    print(f"Tuned P with no tie threshold: {best_p_no_threshold:.2f}")
    print(f"Validation accuracy (no tie threshold): {best_val_acc_no_threshold:.2%}")
    print(
        f"Test accuracy (tuned P, no tie threshold): {test_acc_tuned_no_threshold:.2%}"
    )
    print(
        f"Tuned P and tie_threshold: P={best_p:.2f}, "
        f"tie_threshold={best_tie_threshold:.6g}"
    )
    print(f"Validation accuracy (P and tie_threshold): {best_val_acc:.2%}")
    print(f"Test accuracy (tuned P and tie_threshold): {test_acc_tuned:.2%}")
    if best_val_acc <= val_acc_always_tie:
        print("No setting beat always predicting a tie on validation")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the EAGLE router experiment.")
    parser.add_argument(
        "--embedding-backend",
        choices=list(EMBEDDING_MODELS),
        default="qwen",
        help="Embedding backend (default: qwen, runs locally).",
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
