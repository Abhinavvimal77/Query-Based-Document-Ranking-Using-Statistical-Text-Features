# Query-Based Document Ranking Using Statistical Text Features

A comparative study of classical Information Retrieval (IR) ranking models using statistical text features and the Cranfield benchmark dataset.

This project investigates how statistical text features, probabilistic models, language models, and latent semantic representations can be used to rank documents according to their relevance to a user query.

Six ranking models were implemented and evaluated using sparse matrix operations, standard information retrieval metrics, hyperparameter tuning, and statistical significance testing.

The study compares lexical retrieval methods with latent semantic approaches to understand their effectiveness in retrieving relevant documents.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Dataset](#dataset)
* [Preprocessing Pipeline](#preprocessing-pipeline)
* [Ranking Models](#ranking-models)
* [Evaluation Metrics](#evaluation-metrics)
* [Experimental Results](#experimental-results)
* [Statistical Significance Testing](#statistical-significance-testing)
* [Impact of Preprocessing](#impact-of-preprocessing)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Error Analysis](#error-analysis)
* [Project Structure](#project-structure)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Visualizations](#visualizations)
* [Research Contributions](#research-contributions)
* [Future Work](#future-work)
* [Author](#author)
* [License](#license)

---

## Project Overview

Information Retrieval (IR) systems identify and rank documents according to their relevance to a user's information need.

This project evaluates six classical ranking approaches using the **Cranfield Collection**, a benchmark dataset widely used in information retrieval research.

The ranking models use statistical and mathematical techniques involving:

* Term Frequency (TF)
* Inverse Document Frequency (IDF)
* Document Length Normalization
* Collection Language Models
* Probabilistic Ranking
* Latent Semantic Representations
* Hybrid Lexical-Semantic Scoring

The objective is to compare these approaches under a common preprocessing and evaluation framework and identify which methods perform most effectively on the selected benchmark.

## Objectives

The main objectives of this project are to:

1. Develop an end-to-end document ranking pipeline.
2. Implement and compare lexical, probabilistic, language-model-based, and latent semantic retrieval methods.
3. Investigate the effects of text preprocessing on retrieval performance.
4. Evaluate ranking quality using standard information retrieval metrics.
5. Perform statistical significance testing to compare competing models.
6. Investigate the sensitivity of ranking performance to model hyperparameters.
7. Analyze retrieval failures and identify potential causes of poor ranking performance.
8. Examine whether latent semantic representations improve document retrieval compared with conventional lexical matching.

---

## Dataset

### Cranfield Collection

The experiments use the **Cranfield Collection**, a classical information retrieval benchmark focused on aerodynamics research.

| Dataset Characteristic               | Description                  |
| ------------------------------------ | ---------------------------- |
| Dataset                              | Cranfield Collection         |
| Number of Documents                  | 1,400                        |
| Number of Queries                    | 225                          |
| Average Relevant Documents per Query | Approximately 7.2            |
| Domain                               | Aerodynamics                 |
| Dataset Access                       | `ir_datasets` Python library |

The collection contains technical documents, search queries, and relevance judgments that allow ranking models to be evaluated against known document relevance information.

### Dataset Loading

The dataset is loaded using the `ir_datasets` library.

```python
import ir_datasets

dataset = ir_datasets.load("cranfield")
```

The dataset's document collection, queries, and relevance judgments are used to construct the retrieval and evaluation pipeline.

---

## Preprocessing Pipeline

The same preprocessing procedure is applied consistently to documents and queries.

### Preprocessing Steps

1. **Lowercasing:** Converts all text to lowercase.
2. **Tokenization:** Splits text into individual tokens.
3. **Stopword Removal:** Removes common words that contribute relatively little to retrieval discrimination.
4. **Porter Stemming:** Reduces words to their stems to improve matching between related word forms.

### Example

```text
Original words:
Flows, Flowing, Flowed

After stemming:
flow, flow, flow
```

Applying consistent preprocessing helps reduce vocabulary variation and improves matching between queries and relevant documents.

The impact of preprocessing is also evaluated experimentally by comparing BM25 performance under different preprocessing configurations.

---

## Ranking Models

Six ranking models are implemented and compared.

### 1. Term Frequency (TF)

A basic lexical baseline that ranks documents using the frequencies of query terms appearing in each document.

**Key characteristics:**

* Uses raw term frequencies.
* Measures direct lexical overlap between queries and documents.
* Does not explicitly account for term rarity or document length.
* Serves as a baseline for evaluating more advanced ranking methods.

### 2. TF-IDF with Cosine Similarity

A vector space retrieval model that represents documents and queries using TF-IDF weights.

**Key components:**

* Log-scaled Term Frequency
* Inverse Document Frequency
* TF-IDF document and query vectors
* Cosine Similarity

Cosine similarity measures the angular similarity between query and document vectors, allowing documents to be ranked according to their similarity to a query.

### 3. Okapi BM25

BM25 is a probabilistic ranking function widely used in traditional information retrieval systems.

**Key components:**

* Term Frequency Saturation
* Inverse Document Frequency
* Document Length Normalization

BM25 balances the importance of query terms with their frequency in individual documents and their distribution across the collection.

Its main hyperparameters are:

* `k1`: Controls term frequency saturation.
* `b`: Controls document length normalization.

### 4. Query Likelihood Model with Dirichlet Smoothing

The Query Likelihood Model ranks documents according to the probability of generating the observed query from each document's language model.

Dirichlet smoothing is used to address zero probabilities for query terms that do not occur in a document.

**Key components:**

* Document language models
* Collection term probabilities
* Query likelihood estimation
* Dirichlet prior smoothing

The smoothing parameter `μ` controls the influence of collection-level term probabilities on document-level language models.

### 5. Latent Semantic Analysis (LSA)

Latent Semantic Analysis identifies lower-dimensional latent representations of documents using matrix factorization.

The model uses:

* A TF-IDF document-term matrix
* Truncated Singular Value Decomposition (SVD)
* Low-dimensional latent semantic representations
* Similarity-based document ranking

LSA can identify relationships between documents that are not fully captured by direct term matching.

This makes it useful for investigating vocabulary mismatch and latent associations between query and document terms.

### 6. Hybrid BM25 + LSA

The hybrid model combines lexical relevance from BM25 with semantic relevance from LSA.

The combined score is calculated using a linear fusion of the two ranking signals:

$$
S_{\text{hybrid}}(q,d)
=
\alpha S_{\text{BM25}}(q,d)
+
(1-\alpha)S_{\text{LSA}}(q,d)
$$

where:

* \(q\) represents the query.
* \(d\) represents a document.
* \(S_{\text{BM25}}\) is the lexical relevance score.
* \(S_{\text{LSA}}\) is the latent semantic similarity score.
* \(\alpha\) controls the relative contribution of the two models.

The fusion parameter and score normalization procedure determine how the two scoring methods contribute to the final ranking.

---

## Evaluation Metrics

The ranking models are evaluated using standard information retrieval metrics.

### Binary Relevance Metrics

| Metric                       | Description                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| Precision@10                 | Proportion of the top 10 retrieved documents that are relevant.                                |
| Recall@10                    | Proportion of all relevant documents retrieved within the top 10 results.                      |
| Mean Average Precision (MAP) | Measures ranking quality by averaging precision at relevant-document positions across queries. |
| Mean Reciprocal Rank (MRR)   | Measures how highly the first relevant document appears in the ranking.                        |

### Graded Relevance Metric

| Metric  | Description                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| nDCG@10 | Normalized Discounted Cumulative Gain at rank 10, which evaluates ranking quality using graded relevance judgments. |

These metrics provide complementary perspectives on retrieval performance, including the overall ordering of relevant documents and the placement of relevant results near the top of the ranking.

---

## Experimental Results

The following table summarizes the reported performance of the six ranking models.

### Model Comparison

| Model             |        MAP |        MRR |    nDCG@10 |
| ----------------- | ---------: | ---------: | ---------: |
| TF                |     0.1916 |     0.4231 |     0.2209 |
| TF-IDF            |     0.3079 |     0.5316 |     0.3414 |
| BM25              |     0.3188 |     0.5474 |     0.3474 |
| Query Likelihood  |     0.3027 |     0.5457 |     0.3316 |
| LSA               | **0.3534** | **0.5668** | **0.3847** |
| Hybrid BM25 + LSA |     0.3407 |     0.5655 |     0.3683 |

### Key Findings

* **LSA achieved the best overall performance** among the six evaluated models, obtaining the highest MAP, MRR, and nDCG@10.
* LSA achieved a MAP of `0.3534`, compared with `0.3079` for TF-IDF and `0.3188` for BM25.
* BM25 outperformed the basic TF and TF-IDF baselines in MAP.
* The Query Likelihood Model achieved an MRR of `0.5457`, indicating competitive performance in retrieving the first relevant document.
* The hybrid BM25 + LSA model achieved an MRR of `0.5655`, close to the LSA model's `0.5668`.
* The results suggest that latent semantic representations can improve retrieval quality on this dataset compared with conventional lexical matching approaches.

**Overall observation:** LSA performed best on the reported evaluation metrics, suggesting that modeling latent semantic relationships can be valuable for query-based document ranking in technical text collections.

---

## Statistical Significance Testing

The **Wilcoxon signed-rank test** is used to compare ranking models based on their per-query evaluation results.

This non-parametric test assesses whether paired performance differences between two models are systematically different from zero.

### Reported Findings

* LSA significantly outperformed TF-IDF in MAP, with `p < 0.001`.
* The hybrid BM25 + LSA model significantly outperformed TF-IDF, with `p < 0.001`.
* The difference between BM25 and TF-IDF was reported as statistically indistinguishable.

Statistical significance testing provides additional evidence beyond aggregate metric comparisons by examining model performance across individual queries.

---

## Impact of Preprocessing

The effect of preprocessing on retrieval performance was investigated using BM25.

### BM25 Performance Under Different Preprocessing Configurations

| Preprocessing Configuration |        MAP |
| --------------------------- | ---------: |
| Raw Tokens                  |     0.2735 |
| Stopword Removal            |     0.2903 |
| Stemming Only               |     0.3046 |
| Stopword Removal + Stemming | **0.3188** |

### Key Findings

* Stopword removal improved MAP compared with using raw tokens.
* Stemming alone produced a larger improvement than stopword removal alone.
* Combining stopword removal and stemming achieved the highest MAP among the tested configurations.

The relative improvement from raw-token preprocessing to the combined configuration is approximately:

$$
\text{Improvement (\%)}
=
\frac{0.3188-0.2735}{0.2735}\times100
$$

$$
\boxed{\text{Improvement} \approx 16.6\%}
$$

These findings demonstrate that preprocessing choices can substantially affect retrieval performance, even when the underlying ranking model remains unchanged.

---

## Hyperparameter Tuning

Hyperparameter sensitivity experiments were conducted to investigate how model configurations affect retrieval performance.

### 1. BM25

The following parameter values were explored:

```text
k1 ∈ {0.5, 0.9, 1.2, 1.5, 2, 3, 4}

b ∈ {0.25, 0.5, 0.75, 1.0}
```

**Best reported configuration:**

```text
k1 = 2.0
b  = 0.75
```

The parameter `k1` controls term frequency saturation, while `b` determines the strength of document length normalization.

### 2. Query Likelihood Model

The following Dirichlet smoothing parameter values were explored:

```text
μ ∈ {50, 100, 200, 300, 500, 1000}
```

**Best reported configuration:**

```text
μ = 200
```

### 3. Latent Semantic Analysis

The following latent dimensions were investigated:

```text
Dimensions ∈ {25, 50, 100, 150, 200, 300, 400}
```

**Best reported configuration:**

```text
Dimensions = 150
```

The results highlight the importance of selecting suitable hyperparameters for probabilistic and latent semantic ranking models.

---

## Error Analysis

The study examines retrieval failures to understand why relevant documents may not appear near the top of the ranking.

Three major failure categories were identified.

### 1. Vocabulary Mismatch

Vocabulary mismatch occurs when a query and a relevant document express similar concepts using different words.

For example, a query may use one technical term while the relevant document uses a synonym or an alternative expression.

A strict lexical model may fail when the query and document have little or no term overlap.

In an extreme case:

$$
Q \cap D = \varnothing
$$

where \(Q\) is the set of query terms and \(D\) is the set of document terms.

Latent semantic representations may help address some vocabulary mismatch cases, although they do not guarantee successful retrieval.

### 2. Generic-Term Drift

Generic-term drift occurs when common or weakly discriminative domain-specific terms dominate the ranking.

Documents containing these terms may receive high scores despite not addressing the query's main information need.

This failure highlights the importance of term weighting and effective query representation.

### 3. Compound Queries

Compound queries contain multiple information needs within a single query.

A document may address only one part of the query but still receive a high ranking, while another document that covers the complete information need may be ranked lower.

These cases demonstrate the challenges of balancing multiple query concepts within a single relevance score.

---

## Project Structure

The repository is organized as follows:

```text
NLP_ACADEMIC_PROJECT/
│
├── Code/
│   ├── doc_ranking.py
│   └── advanced_plots.py
│
├── Main_data/
│   ├── cranfield_docs.csv
│   ├── cranfield_queries.csv
│   └── cranfield_qrels.csv
│
├── Results_New/
│
├── Results_Old/
│
├── Advanced_Graphs/
│
├── sample_data/
│   └── README.md
│
└── README.md
```

### Directory Description

| Directory/File      | Purpose                                                             |
| ------------------- | ------------------------------------------------------------------- |
| `Code/`             | Contains the ranking pipeline and visualization scripts.            |
| `doc_ranking.py`    | Runs the document ranking experiments.                              |
| `advanced_plots.py` | Generates advanced visualizations and analytical plots.             |
| `Main_data/`        | Contains the Cranfield documents, queries, and relevance judgments. |
| `Results_New/`      | Stores experiment outputs from the newer results directory.         |
| `Results_Old/`      | Stores previous experiment outputs.                                 |
| `Advanced_Graphs/`  | Stores generated analytical visualizations.                         |
| `sample_data/`      | Contains sample-data documentation.                                 |

---

## Technologies Used

| Technology    | Purpose                                                          |
| ------------- | ---------------------------------------------------------------- |
| Python        | Core implementation                                              |
| NumPy         | Numerical computation and array operations                       |
| SciPy         | Statistical testing and numerical methods                        |
| Pandas        | Dataset handling and result analysis                             |
| Scikit-learn  | TF-IDF vectorization, cosine similarity, and truncated SVD       |
| NLTK          | Text preprocessing, tokenization, stopword removal, and stemming |
| Matplotlib    | Data visualization                                               |
| `ir_datasets` | Access to the Cranfield benchmark dataset                        |

---

## Installation

### Prerequisites

Ensure that Python and `pip` are installed on your system.

### 1. Clone the Repository

Replace the placeholder repository URL below with the actual URL of this repository.

```bash
git clone https://github.com/yourusername/Query-Based-Document-Ranking-Using-Statistical-Text-Features.git
```

### 2. Navigate to the Project Directory

```bash
cd Query-Based-Document-Ranking-Using-Statistical-Text-Features
```

### 3. Install Dependencies

```bash
pip install ir_datasets nltk scikit-learn scipy pandas matplotlib numpy
```

### 4. Download Required NLTK Resources

If the implementation uses NLTK resources that are not already installed, download the required tokenizer and stopword data.

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
```

Depending on the installed NLTK version and the tokenizer implementation, additional resources may be required.

---

## Running the Project

### Run the Ranking Experiments

Navigate to the directory containing `doc_ranking.py` and execute:

```bash
python doc_ranking.py
```

This script runs the document ranking experiments implemented in the project.

### Generate Advanced Visualizations

Run the visualization script:

```bash
python advanced_plots.py
```

This script generates the advanced plots and analytical visualizations supported by the implementation.

**Note:** Run the scripts from the appropriate directory and ensure that the dataset files and required dependencies are available.

---

## Visualizations

The project includes visualizations designed to compare model performance, investigate hyperparameter sensitivity, and analyze retrieval behavior.

The generated visualizations include:

* Model Comparison Charts
* Precision-Recall Curves
* BM25 Parameter Heatmaps
* Three-Dimensional BM25 Performance Surfaces
* LSA Semantic Space Visualizations
* Hybrid Model Performance Surfaces
* Per-Query Error Analysis Plots
* Pairwise Statistical Comparison Heatmaps

These visualizations complement the numerical results and provide additional insight into the behavior of the ranking models.

---

## Research Contributions

The project brings together several components of classical information retrieval within a reproducible experimental framework.

The main contributions include:

* An end-to-end query-based document ranking pipeline.
* Comparative evaluation of six classical and latent semantic ranking models.
* Consistent preprocessing of queries and documents.
* Evaluation using standard information retrieval metrics.
* Statistical significance testing of model comparisons.
* Hyperparameter sensitivity analysis for BM25, Query Likelihood, and LSA.
* Investigation of preprocessing effects on retrieval quality.
* Analysis of common document retrieval failure types.
* Visual exploration of ranking performance and model behavior.
* Experimental comparison of lexical, probabilistic, language-model-based, and latent semantic approaches.

---

## Future Work

Potential extensions of this project include:

1. **Query Expansion:** Expand queries using related terms to improve recall.
2. **Relevance Feedback:** Incorporate user relevance judgments using methods such as the Rocchio algorithm.
3. **Learning to Rank:** Train ranking models using query-document relevance features.
4. **Neural Retrieval:** Investigate neural retrieval methods and transformer-based reranking.
5. **Dense Vector Search:** Compare traditional sparse retrieval with dense document embeddings.
6. **BEIR Benchmark Evaluation:** Evaluate generalization across additional information retrieval datasets.
7. **Query-Dependent Score Fusion:** Develop adaptive methods for combining lexical and semantic relevance scores.
8. **Improved Vocabulary Matching:** Explore synonym expansion and domain-specific terminology handling.

---

## Author

**Abhinav M**

Department of Mathematics and Statistics
Faculty of Science and Technology
Vishwakarma University
Pune, Maharashtra, India

---

## License

This project is intended for academic and research purposes.

Please provide appropriate attribution when using or extending the work. A formal open-source license should be added to the repository if permissions for reuse, modification, and redistribution are to be explicitly defined.
