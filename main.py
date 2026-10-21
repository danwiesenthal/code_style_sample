"""Main entry point - run EAGLE router experiment."""

import argparse
from collections import Counter

import numpy as np
from dotenv import load_dotenv

from data_access import get_sampled_splits
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
    seed: int = 42, embedding_backend: EmbeddingBackend = "qwen"
) -> None:
    """Run full EAGLE experiment.

    Steps:
        1. Load and sample data (2000 train, 200 val, 200 test)
        2. Embed all prompts
        3. Build router from training data
        4. Tune P and tie_threshold on validation set
        5. Report test accuracy for always predicting a tie, paper P, and tuned
           settings if any beat that baseline
    """

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("Loading and sampling data...")
    train_rows, val_rows, test_rows = get_sampled_splits(seed=seed)
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

    # -------------------------------------------------------------------------
    # 4. Tune P and tie_threshold on validation set
    # -------------------------------------------------------------------------
    print("\nTuning P and tie_threshold on validation set...")
    p_values = np.linspace(0, 1, 21)  # 0.0, 0.05, ..., 1.0
    tie_threshold_values = [0, 1, 2, 5, 10, 20, 50, 100]

    # A setting has to beat always predicting a tie, the most common outcome
    best_p, best_tie_threshold = None, None
    best_val_acc = compute_baseline_accuracy(val_rows, Outcome.TIE)
    print(f"  Always predicting tie: {best_val_acc:.2%}")

    # One row per P, one column per tie_threshold
    print("  Validation accuracy by P (rows) and tie_threshold (columns):")
    print("     P" + "".join(f"{t:>7}" for t in tie_threshold_values))

    for p in p_values:
        row_accs = []
        for tie_threshold in tie_threshold_values:
            router = EagleRouter(
                global_ratings,
                train_index,
                k=DEFAULT_K,
                n_neighbors=DEFAULT_N,
                p=p,
                tie_threshold=tie_threshold,
            )
            val_acc = compute_accuracy(router, val_rows, val_emb)
            row_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_p, best_tie_threshold, best_val_acc = p, tie_threshold, val_acc
        print(f"  {p:4.2f}" + "".join(f"{acc:7.1%}" for acc in row_accs))

    if best_p is None:
        print("  No setting beat always predicting a tie")
    else:
        print(f"  Best P: {best_p:.2f}, tie_threshold: {best_tie_threshold}")
        print(f"  Validation accuracy: {best_val_acc:.2%}")

    # -------------------------------------------------------------------------
    # 5. Evaluate on test set
    # -------------------------------------------------------------------------
    print("\nEvaluating on test set...")

    # Baseline: always predict a tie, the most common outcome
    test_acc_always_tie = compute_baseline_accuracy(test_rows, Outcome.TIE)
    print(f"  Always predicting tie: {test_acc_always_tie:.2%}")

    # With paper hyperparams (no tie_threshold in paper)
    router_paper = EagleRouter(
        global_ratings, train_index, k=DEFAULT_K, n_neighbors=DEFAULT_N, p=DEFAULT_P
    )
    test_acc_paper = compute_accuracy(router_paper, test_rows, test_emb)
    print(f"  Paper P={DEFAULT_P}: {test_acc_paper:.2%}")

    # With tuned hyperparams, if any beat the baseline on validation
    if best_p is not None:
        router_tuned = EagleRouter(
            global_ratings,
            train_index,
            k=DEFAULT_K,
            n_neighbors=DEFAULT_N,
            p=best_p,
            tie_threshold=best_tie_threshold,
        )
        test_acc_tuned = compute_accuracy(router_tuned, test_rows, test_emb)
        print(
            f"  Tuned P={best_p:.2f}, tie_threshold={best_tie_threshold}: "
            f"{test_acc_tuned:.2%}"
        )

    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Test accuracy (always tie): {test_acc_always_tie:.2%}")
    print(f"Test accuracy (paper P={DEFAULT_P}): {test_acc_paper:.2%}")
    if best_p is None:
        print("Tuned: no setting beat always predicting a tie on validation")
    else:
        print(f"Tuned P: {best_p:.2f}, tie_threshold: {best_tie_threshold}")
        print(f"Validation accuracy: {best_val_acc:.2%}")
        print(f"Test accuracy (tuned): {test_acc_tuned:.2%}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the EAGLE router experiment.")
    parser.add_argument(
        "--embedding-backend",
        choices=list(EMBEDDING_MODELS),
        default="qwen",
        help="Embedding backend (default: qwen, runs locally).",
    )
    args = parser.parse_args()
    run_experiment(embedding_backend=args.embedding_backend)
