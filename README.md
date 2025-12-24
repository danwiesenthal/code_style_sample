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

# Run unit tests (skips integration tests, which call HuggingFace and OpenAI)
uv run pytest tests/ -v -m "not integration"

# Run all tests
uv run pytest tests/ -v

# Lint and check formatting
uv run ruff check .
uv run ruff format --check .
```

Requires a `.env` file with `OPENAI_API_KEY` for embeddings.

## Project Structure

Start with `main.py` - it runs the full experiment:

1. Loads SPROUT and samples 2000 train, 200 val, and 200 test rows, to keep embedding time short
2. Embeds prompts using OpenAI text-embedding-3-large
3. Builds the EAGLE router from training data
4. Tunes P and tie_threshold on the validation set
5. Reports test accuracy, next to the accuracy of always predicting a tie

Supporting modules:
- `router.py` - EAGLE router combining global and local Elo ratings
- `evaluation.py` - Accuracy computation
- `embeddings.py` - OpenAI embeddings with disk caching
- `similarity.py` - Nearest neighbor retrieval
- `ranker.py` - Elo rating computation from match sequences
- `elo.py` - Core Elo math
- `data_access.py` - SPROUT dataset loading
- `match_types.py` - Match types shared across modules

## Differences from the Paper

- **Task and evaluation.** The paper ranks many models and picks the best one within a cost budget, evaluated on RouterBench by quality-versus-cost AUC. This repo predicts the outcome between two models on SPROUT and measures how often that prediction is right.
- **Embedding model.** The paper uses stella_en_1.5B_v5. This repo uses OpenAI's text-embedding-3-large.
- **Order of Elo updates.** Elo weights later updates more, so the order matches are applied in matters, and the paper doesn't specify it. The local rating applies neighbors from least to most similar, so the most similar counts most. The global rating is one pass over the training matches in dataset order; the paper calls it an average Elo over all feedback without saying how it's averaged.
- **Terms.** The paper calls Elo's S the "actual score" and the combined rating "Score(X)". Here "score" only means a judge's score for a response; S is "points", and the combination is the combined rating.
- **Tie threshold.** An addition to the paper, described below.

## Extension: Tie Threshold

The paper's router always picks whichever model has the higher combined rating, so it never predicts a tie. When the two models often score the same on a prompt, each of those ties counts as a miss.

This repo extends the paper with a `tie_threshold` parameter, which predicts TIE when the rating difference is within the threshold. The default of 0 matches the paper's behavior, and `main.py` tunes it alongside P on the validation set. Tuning only keeps a setting if it beats always predicting a tie on the validation set.

## License

MIT - see [LICENSE](LICENSE).
