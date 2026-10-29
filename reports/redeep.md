# ReDeEP

[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7daf60e805e596c3bd1e843e72ea5560-Paper-Conference.pdf) · ICLR 2025 · [Released code](https://github.com/Jeryi-Sun/ReDEeP-ICLR)

ReDeEP detects hallucinations in retrieval-augmented generation by separating two internal signals. Its External Context Score measures how copying heads use retrieved evidence. Its Parametric Knowledge Score measures the change introduced by feed-forward layers through LogitLens distributions and Jensen–Shannon divergence. Token- and chunk-level detectors combine selected heads and layers into a hallucination score.

## Evaluation Protocol

| Item | Specification |
| :--- | :--- |
| Models | Llama-2-7B-Chat and Llama-2-13B-Chat in Table 1; the wider paper also studies Llama-3-8B-Instruct |
| Data | RAGTruth and Dolly with Accurate Context |
| Selection | RAGTruth validation split; two-fold validation for Dolly |
| Decoding | Greedy generation; other generation settings use model defaults |
| Features | Selected copying-head External Context Scores and FFN-layer Parametric Knowledge Scores |
| Evaluation | AUC, Pearson correlation, accuracy, recall, and F1 |
| Repetition | Three runs for experiments with randomness |
| Hardware | NVIDIA V100 GPUs and a 52-core Intel Xeon Gold 6230R CPU |
| Primary target | Table 1 ReDeEP(Token) and ReDeEP(Chunk) |

### Table 1 ReDeEP Results

| Model | Variant | Dataset | AUC | PCC | Accuracy | Recall | F1 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Llama-2-7B-Chat | Token | RAGTruth | 0.7325 | 0.3979 | 0.7067 | 0.6770 | 0.6986 |
| Llama-2-7B-Chat | Token | Dolly | 0.6884 | 0.3266 | 0.6464 | 0.8070 | 0.7244 |
| Llama-2-7B-Chat | Chunk | RAGTruth | 0.7458 | 0.4203 | 0.6822 | 0.8097 | 0.7190 |
| Llama-2-7B-Chat | Chunk | Dolly | 0.7949 | 0.5136 | 0.7373 | 0.8245 | 0.7833 |
| Llama-2-13B-Chat | Token | RAGTruth | 0.8181 | 0.5478 | 0.7711 | 0.7440 | 0.7494 |
| Llama-2-13B-Chat | Token | Dolly | 0.7226 | 0.3776 | 0.6465 | 0.8148 | 0.7154 |
| Llama-2-13B-Chat | Chunk | RAGTruth | 0.8244 | 0.5566 | 0.7889 | 0.7198 | 0.7587 |
| Llama-2-13B-Chat | Chunk | Dolly | 0.8420 | 0.5902 | 0.7070 | 0.8518 | 0.7603 |

## Release Audit

The repository is pinned at `4d081915b8fb4430fda65c411da61540cc73cc57` and includes the four ReDeEP scripts, selected copying-head files, a modified Transformers source tree, and a Python 3.9, PyTorch 2.0.1, CUDA 11.7 environment export. We verified both public archives. The root README archive contains RAGTruth only; the dataset README links a second archive with Dolly, under a directory name the scripts do not expect. Model and BGE snapshots are unpinned.

The unchanged code cannot reproduce Table 1. Both regressors pass a column-sliced dataframe that removes the required parametric-knowledge features, causing an immediate `KeyError`. They also rank heads and layers and fit min-max scaling on the same test labels they evaluate, whereas the paper states that hyperparameters are selected on RAGTruth validation data and by two-fold validation on Dolly. The scripts only save AUC and PCC, not accuracy, recall, or F1.

Fresh Dolly extraction is also invalid: the label helper returns an empty hallucination-span list whenever the data path contains `dolly`, producing only negative labels. The chunk regressor repeats the RAGTruth condition in both Llama-2 Dolly branches and always reads RAGTruth metadata. The Llama-3 chunk filename expected by the source differs from both archives. These are protocol and execution blockers, not harmless environment differences. The copying-head ranking code, exact folds, and seeds are also absent.

## Status

Audit and source-preserving preflight are complete. Verdict: **baseline blocked**. The public data are recoverable, but Table 1 cannot be faithfully executed from the unchanged release. A credible future run would require an explicit repair, a prespecified validation/test split, pinned snapshots, and independent computation of every reported metric. It must be labeled a repaired replication rather than an unchanged-source reproduction. No paid compute was started.
