# Candidate Retrieval Implementation Plan

**Goal:** Produce a reproducible `answer.csv` with up to 50 valid item IDs per benchmark query and maximize local Recall@50.

**Architecture:** Index benchmark item titles with character TF-IDF and item descriptions/parameters with word TF-IDF. Combine lexical scores with a location prior, tune the combination on held-out labeled train pairs whose items belong to the benchmark corpus, and validate the final CSV against the Parquet inputs.

**Tech Stack:** Python 3.14, PyArrow, NumPy, SciPy, scikit-learn, pytest.

**Source:** `task.md`.

## Tasks

1. Add tests for text normalization, retrieval ranking, and the CSV contract. Confirm they fail before implementation.
2. Implement the corpus reader and sparse text retrieval index, with comments explaining the scoring choices. Run tests.
3. Implement local validation with a fixed random seed. Compare score weights and location priors using Recall@50.
4. Generate `answer.csv` from the benchmark files. Check exact query coverage, ID membership, row lengths, and column names.
5. Document dependencies, reproducibility commands, method, validation result, and limitations. Run fresh tests and validation, then commit code and CSV.

## Constraints

- No query ID specific answers or manual reconstruction of benchmark labels.
- All inference runs locally without external APIs.
- Preserve the original 16-character ID strings and output only `query_id,answer`.
- Keep the local Parquet dataset outside Git.
