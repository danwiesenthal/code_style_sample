# EAGLE Router

Implementation of the EAGLE router from [Eagle: Efficient Training-Free Router for Multi-LLM Inference](https://arxiv.org/abs/2409.15518), routing between gpt-4o (model A) and gpt-4o-mini (model B) on prompts from the [SPROUT](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT) dataset. (This repo mainly exists as an example of code style.)

A router decides, prompt by prompt, whether the cheaper model is good enough or the stronger one is needed. EAGLE predicts which model will give the better response by combining two Elo ratings:
- a global rating, computed from every training match
- a local rating, computed from the training prompts closest to the query in embedding space, starting from the global rating

The combined rating is `P * global + (1 - P) * local`. A match is one prompt where both models' responses were scored; equal scores count as a tie. SPROUT already contains both models' responses and their scores, so the router learns from those recorded scores and never calls gpt-4o or gpt-4o-mini itself. Accuracy is the fraction of test prompts where the router predicts the actual outcome: A wins, B wins, or tie.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the experiment
uv run python main.py

# Run unit tests (skips integration tests, which download data and models)
uv run pytest tests/ -v -m "not integration"

# Run integration tests, except ones that call the paid OpenAI API
uv run pytest tests/ -v -m "integration and not openai"

# Lint and check formatting
uv run ruff check .
uv run ruff format --check .
```

Embeddings come from [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), which runs locally and downloads on first use. To use OpenAI's text-embedding-3-large instead, add `OPENAI_API_KEY` to a `.env` file and run `uv run python main.py --embedding-backend openai`.

## Project Structure

Start with `main.py` - it runs the full experiment:

1. Loads SPROUT and samples 1200 train, 600 val, and 600 test rows by default, to keep embedding time short
2. Embeds prompts using Qwen3-Embedding-0.6B (or OpenAI text-embedding-3-large)
3. Builds the EAGLE router from training data
4. Tunes P with no tie threshold, then tunes P and tie_threshold together on the validation set
5. Reports test accuracy for the paper settings, both tuned variants, and always predicting a tie

Supporting modules:
- `router.py` - EAGLE router combining global and local Elo ratings
- `evaluation.py` - Accuracy computation
- `embeddings.py` - Qwen or OpenAI embeddings with disk caching
- `similarity.py` - Nearest neighbor retrieval
- `ranker.py` - Elo rating computation from match sequences
- `elo.py` - Core Elo math
- `data_access.py` - SPROUT dataset loading
- `match_types.py` - Match types shared across modules

## Differences from the Paper

- **Task and evaluation.** The paper ranks many models and picks the best one within a cost budget, evaluated on RouterBench by quality-versus-cost AUC. This repo predicts the outcome between two models on SPROUT and measures how often that prediction is right.
- **Embedding model.** The paper uses stella_en_1.5B_v5. This repo uses Qwen3-Embedding-0.6B by default, or OpenAI's text-embedding-3-large.
- **Order of Elo updates.** Elo weights later updates more, so the order matches are applied in matters, and the paper doesn't specify it. The local rating applies neighbors from least to most similar, so the most similar counts most. The global rating is one pass over the training matches in dataset order; the paper calls it an average Elo over all feedback without saying how it's averaged.
- **Terms.** The paper calls Elo's S the "actual score" and the combined rating "Score(X)". Here "score" only means a judge's score for a response; S is "points", and the combination is the combined rating.
- **Tie threshold.** An addition to the paper, described below.

## Extension: Tie Threshold

The paper's router selects a model rather than predicting a tie. On SPROUT, gpt-4o and gpt-4o-mini receive the same recorded score on about 78% of prompts. Under this repo's exact-outcome metric, predicting either model as the winner counts as a miss on those prompts.

This repo extends the paper with a `tie_threshold` parameter, which predicts TIE when the rating difference is within the threshold. At 0, the router picks the higher-rated model unless the ratings are exactly equal. `main.py` tunes P with the threshold fixed at 0, then tunes both together. Each variant reports its best validation setting alongside always predicting a tie, even when it does not beat that baseline.

The search uses 21 evenly spaced P values and zero plus 61 logarithmically spaced thresholds from 1 to 1000. The output displays selected threshold columns; all 62 thresholds are evaluated.

## Results

The experiment uses 1200 train, 600 validation and 600 test prompts from SPROUT, with Qwen3-Embedding-0.6B embeddings and seed 42. Each sample comes from its corresponding official dataset split; tuning uses only validation outcomes.

| Router | Test accuracy |
|---|---|
| Paper settings on SPROUT (P=0.5, no tie threshold) | 13.8% |
| Tuned P with no tie threshold (P=0.00) | 15.5% |
| Tuned P and tie_threshold (P=0.10, tie_threshold=79.4328) | 74.3% |
| Always predict a tie | 74.7% |

The paper evaluates model selection on RouterBench by quality-versus-cost AUC. Its fixed settings are P=0.5, 20 neighbors and K=32 ([Appendix A](https://arxiv.org/html/2409.15518v2#A1)). The first row applies those settings here on SPROUT; it is not the paper's published result. All rows above measure exact outcome accuracy: A wins, B wins, or tie.

Without a tie threshold, tuning P improves test accuracy from 13.8% to 15.5%. Once tie predictions are allowed, tuning selects P=0.10 and tie_threshold=79.4328: it gets 467 of 600 validation outcomes right, versus 465 for always predicting a tie (77.8% versus 77.5%). On test, that small advantage reverses: the tuned router gets 446 of 600 right, versus 448 for always predicting a tie (74.3% versus 74.7%).

The strong always-tie baseline reflects how similar these two models' recorded scores are on this dataset: they tie on 448 of the 600 test prompts. Allowing TIE as an answer makes that common outcome easy to predict. This result does not establish that EAGLE is broken or that routing is unhelpful for other model pairs or datasets. It shows that these tuned settings do not beat the tie baseline on this sample under exact-outcome accuracy.

A router that only has to choose a model has a different objective. On a tied prompt, choosing gpt-4o-mini preserves the recorded score at lower cost, even though predicting "B wins" would be wrong under this metric.

Example output from `uv run python main.py`:

```text
Loading and sampling data...
  Train: 1200, Val: 600, Test: 600
  Train outcomes: {<Outcome.TIE: 'tie'>: 953, <Outcome.WIN_B: 'win_b'>: 98, <Outcome.WIN_A: 'win_a'>: 149}

Embedding prompts with Qwen/Qwen3-Embedding-0.6B...
  Embedding dim: 1024

Building router...
  Global ratings: A=1486.95, B=1513.05

Tuning P and tie_threshold on validation set...
  Searching 21 P values and 62 tie thresholds (1,302 settings)
  Always predicting tie: 77.50%
  Validation accuracy by P (rows) and selected tie_thresholds (columns):
     P      0      1  3.162     10  31.62    100  316.2   1000
  0.00  11.8%  13.0%  14.3%  26.7%  57.7%  77.7%  77.5%  77.5%
  0.05  11.8%  11.8%  16.0%  26.7%  59.3%  77.7%  77.5%  77.5%
  0.10  11.3%  13.2%  15.3%  28.5%  61.2%  77.5%  77.5%  77.5%
  0.15  11.3%  13.8%  16.5%  29.7%  62.8%  77.3%  77.5%  77.5%
  0.20  11.2%  12.7%  18.5%  32.8%  63.7%  77.5%  77.5%  77.5%
  0.25  11.3%  14.0%  18.2%  34.5%  65.8%  77.5%  77.5%  77.5%
  0.30  11.2%  13.3%  19.2%  35.7%  68.2%  77.5%  77.5%  77.5%
  0.35  10.7%  13.5%  20.7%  37.7%  68.8%  77.5%  77.5%  77.5%
  0.40  10.7%  14.7%  22.7%  38.3%  70.2%  77.5%  77.5%  77.5%
  0.45  10.7%  15.0%  22.8%  40.7%  72.0%  77.5%  77.5%  77.5%
  0.50  11.0%  14.5%  22.0%  39.5%  73.7%  77.5%  77.5%  77.5%
  0.55  11.3%  13.3%  19.2%  39.2%  74.0%  77.5%  77.5%  77.5%
  0.60  11.2%  13.2%  17.7%  36.8%  74.3%  77.5%  77.5%  77.5%
  0.65  10.8%  12.5%  16.5%  33.3%  74.8%  77.5%  77.5%  77.5%
  0.70  10.3%  11.7%  15.3%  27.5%  75.3%  77.5%  77.5%  77.5%
  0.75  10.0%  11.2%  13.8%  21.3%  75.7%  77.5%  77.5%  77.5%
  0.80   9.7%   9.7%  10.0%  15.2%  75.8%  77.5%  77.5%  77.5%
  0.85   9.0%   9.0%   9.0%  11.0%  76.3%  77.5%  77.5%  77.5%
  0.90   9.0%   9.0%   9.0%   9.0%  77.3%  77.5%  77.5%  77.5%
  0.95   9.0%   9.0%   9.0%   9.0%  77.5%  77.5%  77.5%  77.5%
  1.00   9.0%   9.0%   9.0%   9.0%  77.5%  77.5%  77.5%  77.5%
  Best P with no tie threshold: 0.00
  Validation accuracy (no tie threshold): 11.83%
  Best P and tie_threshold: P=0.10, tie_threshold=79.4328
  Validation accuracy (P and tie_threshold): 77.83%

Evaluating on SPROUT test set...
  Always predicting tie: 74.67%
  Paper P=0.5, no tie threshold: 13.83%
  Tuned P=0.00, no tie threshold: 15.50%
  Tuned P=0.10, tie_threshold=79.4328: 74.33%

========================================
RESULTS (SPROUT)
========================================
Test accuracy (always tie): 74.67%
Test accuracy (paper P=0.5, no tie threshold): 13.83%
Tuned P with no tie threshold: 0.00
Validation accuracy (no tie threshold): 11.83%
Test accuracy (tuned P, no tie threshold): 15.50%
Tuned P and tie_threshold: P=0.10, tie_threshold=79.4328
Validation accuracy (P and tie_threshold): 77.83%
Test accuracy (tuned P and tie_threshold): 74.33%
```

## License

MIT - see [LICENSE](LICENSE).
