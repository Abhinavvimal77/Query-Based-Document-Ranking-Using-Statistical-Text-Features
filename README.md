Query-Based Document Ranking Using Statistical Text Features
Overview

This project presents a comparative study of classical Information Retrieval (IR) ranking models on the Cranfield Collection, a benchmark dataset widely used in IR research.

The work investigates how statistical text features such as:

Term Frequency (TF)
Inverse Document Frequency (IDF)
Document Length
Collection Language Models
Latent Semantic Features

can be used to rank documents in response to user queries.

Six ranking models were implemented and evaluated from scratch using sparse matrix operations and statistical significance testing.

Objectives

The project aims to:

Build a complete document ranking pipeline
Compare lexical and latent-semantic ranking approaches
Quantify the impact of preprocessing
Evaluate ranking quality using standard IR metrics
Perform statistical significance testing
Analyze retrieval failures and vocabulary mismatch
Models Implemented
1. Term Frequency (TF)

Basic term-overlap baseline using raw query term frequencies.

2. TF-IDF + Cosine Similarity

Vector Space Model using:

Log-scaled Term Frequency
Inverse Document Frequency
Cosine Similarity
3. Okapi BM25

Probabilistic ranking model incorporating:

Term frequency saturation
Document length normalization
Inverse document frequency
4. Query Likelihood Model (Dirichlet Smoothing)

Language-model-based retrieval using:

Collection term probabilities
Dirichlet Prior Smoothing
5. Latent Semantic Analysis (LSA)

Semantic retrieval through:

TF-IDF document matrix
Truncated Singular Value Decomposition (SVD)
Latent semantic document representations
6. Hybrid BM25 + LSA

Linear fusion of:

Lexical relevance (BM25)
Semantic relevance (LSA)
Dataset
Cranfield Collection

The experiments use the Cranfield benchmark dataset:

Documents: 1,400
Queries: 225
Average relevant documents per query: 7.2
Domain: Aerodynamics

Dataset loaded using:

Python
import ir_datasets
dataset = ir_datasets.load("cranfield")
Preprocessing Pipeline

The same preprocessing is applied to both queries and documents.

Steps
Lowercasing
Tokenization
Stopword Removal
Porter Stemming

Example:

Plain Text
"Flows", "Flowing", "Flowed"
↓
flow
Evaluation Metrics

The ranking models are evaluated using:

Binary Relevance Metrics
Precision@10
Recall@10
Mean Average Precision (MAP)
Mean Reciprocal Rank (MRR)
Graded Relevance Metric
nDCG@10
Main Results
Model	MAP	MRR	nDCG@10TF	0.1916	0.4231	0.2209
TF-IDF	0.3079	0.5316	0.3414
BM25	0.3188	0.5474	0.3474
Query Likelihood	0.3027	0.5457	0.3316
LSA	0.3534	0.5668	0.3847
Hybrid BM25+LSA	0.3407	0.5655	0.3683
Key Finding

Latent Semantic Analysis (LSA) achieved the best overall performance, significantly outperforming TF-IDF and BM25 in Mean Average Precision (MAP).

Statistical Significance Testing

Model comparisons were validated using:

Plain Text
Wilcoxon Signed-Rank Test

Key findings:

LSA significantly outperformed TF-IDF (p < 0.001)
Hybrid BM25 + LSA significantly outperformed TF-IDF (p < 0.001)
BM25 and TF-IDF were statistically indistinguishable
Impact of Preprocessing

BM25 MAP scores:

Configuration	MAPRaw Tokens	0.2735
Stopwords Removed	0.2903
Stemming Only	0.3046
Stopwords + Stemming	0.3188
Result

Combining stopword removal and stemming improved MAP by approximately:

Plain Text
16.6%
Hyperparameter Tuning
BM25

Parameters explored:

Plain Text
k1 ∈ {0.5,0.9,1.2,1.5,2,3,4}
b ∈ {0.25,0.5,0.75,1.0}

Best:

Plain Text
k1 = 2.0
b = 0.75
Query Likelihood
Plain Text
μ ∈ {50,100,200,300,500,1000}

Best:

Plain Text
μ = 200
LSA
Plain Text
Dimensions ∈ {25,50,100,150,200,300,400}

Best:

Plain Text
150 dimensions
Error Analysis

The study identified three major retrieval failure types:

Vocabulary Mismatch

Relevant documents share no terms with the query.

Example:

Plain Text
Query terms ∩ Relevant document terms = ∅
Generic-Term Drift

Common domain-specific terms dominate ranking.

Compound Queries

Multiple information needs represented within one query.

Project Structure
Plain Text
NLP_ACADEMIC_PROJECT/
│
├── Code/
│ ├── doc_ranking.py
│ ├── advanced_plots.py
│
├── Main_data/
│ ├── cranfield_docs.csv
│ ├── cranfield_queries.csv
│ ├── cranfield_qrels.csv
│
├── Results_New/
├── Results_Old/
├── Advanced_Graphs/
├── sample_data/
│
└── README.md
Installation

Clone the repository:

Shell
git clone https://github.com/yourusername/Query-Based-Document-Ranking-Using-Statistical-Text-Features.git
 
cd Query-Based-Document-Ranking-Using-Statistical-Text-Features

Install dependencies:

Shell
pip install ir_datasets nltk scikit-learn scipy pandas matplotlib numpy
Running the Project

Run ranking experiments:

Shell
python doc_ranking.py

Generate advanced visualizations:

Shell
python advanced_plots.py
Key Visualizations

The project generates:

Model Comparison Charts
Precision-Recall Curves
BM25 Parameter Heatmaps
3D BM25 Surfaces
LSA Semantic Space Visualizations
Hybrid Model Performance Surfaces
Per-query Error Analysis Plots
Pairwise Statistical Comparison Heatmaps
Technologies Used
Python
NumPy
SciPy
Pandas
Scikit-Learn
NLTK
Matplotlib
IR-Datasets
Research Contributions

✅ End-to-end Information Retrieval pipeline

✅ Six ranking models implemented from scratch

✅ Statistical significance testing

✅ Hyperparameter sensitivity analysis

✅ Failure taxonomy and error analysis

✅ Reproducible experimental framework

✅ Comparative evaluation on Cranfield benchmark

Future Work

Potential extensions include:

Query Expansion
Relevance Feedback (Rocchio)
Learning-to-Rank Models
Neural Retrieval (DPR, BERT Re-rankers)
Dense Vector Search
BEIR Benchmark Evaluation
Query-dependent Score Fusion
Author

Abhinav M

Department of Mathematics and Statistics
 Faculty of Science and Technology
 Vishwakarma University, Pune, India

License

This project is intended for academic and research purposes. Please cite appropriately if used in research or educational work.
