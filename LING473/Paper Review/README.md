# Replicating and Extending Bayram et al. (2022): Suicidal Ideation Detection with Lexical Network Features

A replication and extension of [Bayram et al. (2022)](https://doi.org/10.22191/nejcs/vol4/iss1/2), which proposes lexical associative network features for detecting suicidal ideation in clinical interview transcripts. This project reproduces their feature pipeline and classifiers on a public dataset, adds the transformer baselines and statistical tests the original paper lacks, and evaluates whether the paper's conclusions hold under more rigorous experimental conditions.

## Motivation

I reviewed Bayram et al. (2022) for LING 473 and identified three testable methodological gaps:

1. No transformer baselines (BERT, MentalBERT) despite 2022 publication
2. No statistical significance tests between classifiers
3. The CNN baseline uses default hyperparameters without tuning, making the comparison unfair

This project addresses all three. The review itself is included in `review/`.

## What this project tests

* Whether logistic regression still matches deeper models when evaluated with bootstrap confidence intervals and permutation tests (not just mean AUC)
* Whether fine-tuned MentalBERT outperforms the classical methods on the same data, validating the missing-baseline critique
* Whether a tuned CNN closes the gap with simpler methods (or confirms that architectural complexity doesn't help on small clinical datasets)
* Whether the novel lexical network features add value on top of standard n-grams in a hybrid feature set

## Project structure

```
Paper Review/
  README.md              # This file
  requirements.txt       # Pinned dependencies
  review/                # Class submission (PDF + references)
  src/
    features/
      text_features.py   # Unigram, bigram, n-gram, stopwords extractors
      network_features.py # Algorithm 1: lexical co-occurrence network features
    models/
      classifiers.py     # Logistic, MLP, SVM, Random Forest
    evaluation/
      metrics.py         # Monte Carlo CV, bootstrap CI, permutation tests
  notebooks/
    01_data_prep.py      # Load and preprocess public dataset
    02_feature_extraction.py
    03_train_baselines.py
    04_train_transformers.py
    05_evaluation.py
    06_results_analysis.py
  writeup/               # Mini-paper with figures
```

## Data

The original paper uses three IRB-protected clinical interview corpora from Cincinnati Children's Hospital Medical Center. Since those are not publicly available, this replication uses the Reddit Suicide Dataset (details in `notebooks/01_data_prep.py`). This shifts the domain from structured clinical interviews to social media text, which actually tests a generalizability question the paper raises but does not fully answer.

## Methods replicated from the paper

* Five feature types: unigrams, bigrams, combined n-grams, stopwords-only, and lexical associative network features (Algorithm 1)
* Monte Carlo 10-fold cross-validation with balanced 50-sample held-out test sets
* Logistic softmax (no hidden layers) and MLP (1 hidden layer, 1000 neurons, tanh) classifiers
* AUC as the primary evaluation metric

## Our additions

* BERT-base and MentalBERT fine-tuning baselines
* CNN with Optuna hyperparameter tuning (addressing the unfair default-config comparison)
* SVM and Random Forest external baselines
* Bootstrap 95% confidence intervals on AUC
* Paired permutation tests between all classifier pairs
* Per-class metrics (sensitivity, specificity, PPV)
* Hybrid feature set (network features + n-grams)
* Lexical network visualizations
* Error analysis on misclassified examples

## Results

*Pending. Results table will be populated after experiments run.*

## Reproducing

```bash
pip install -r requirements.txt
# Run notebooks 01-06 in order
```

## Citation

Bayram, U., Lee, W., Santel, D., Minai, A. A., Clark, P. O., Glauser, T., & Pestian, J. (2022). Toward Suicidal Ideation Detection with Lexical Network Features and Machine Learning. *Northeast Journal of Complex Systems (NEJCS)*, 4(1), Article 2. https://doi.org/10.22191/nejcs/vol4/iss1/2
