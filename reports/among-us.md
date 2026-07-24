# Among Us Probe Replication

**Paper** | [Among Us: A Sandbox for Measuring and Detecting Agentic Deception](https://arxiv.org/abs/2504.04072)
**Code** | [Official repository](https://github.com/7vik/AmongUs)
**Scope** | Phi-4 linear-probe baseline plus cross-model and layer extensions.

## Baseline verdict

**Not replicated.** We ran the released Phi-4@L20 Figure 7 path with its public
data, unchanged dataset and preprocessing classes, literal model loader, and
unchanged probe trainer. The reconstructed probes learned strong in-domain
classifiers, but the reported cross-dataset generalization matrix did not
reappear, not even close. Mean AUROC across the 20 Figure 7 cells fell from **0.8468** to
**0.7339**. The authors' released checkpoints scored **0.5462** on the same
reconstructed cache.

| Model | Layer | Tokens | Probe            | Training                                        | Split                           |
| :---- | ----: | -----: | :--------------- | :---------------------------------------------- | :------------------------------ |
| Phi-4 | 20/40 |      5 | Bias-free linear | Adam, LR`0.001`, WD `0.001`, 4 epochs, batch 32 | First 80% train / last 20% test |

> NB: The paper and config also states 10 tokens, but the released Figure 7 script
> hard-codes 5. The comparison below follows the released Figure 7 path.

## Paper vs replication

Here, one row is one train and evaluation cell. `Delta = replication - paper`.

| Model         | Layer | Tokens | Training data | Evaluation target | Paper AUROC | Replication AUROC |       Delta | Released checkpoint AUROC |
| :------------ | ----: | -----: | :------------ | :---------------- | ----------: | ----------------: | ----------: | ------------------------: |
| Phi-4         |    20 |      5 | Among Us      | TruthfulQA        |      0.5417 |            0.5859 |     +0.0441 |                    0.5227 |
| Phi-4         |    20 |      5 | Among Us      | DishonestQA       |      0.8194 |            0.4764 |     -0.3429 |                    0.6778 |
| Phi-4         |    20 |      5 | Among Us      | RepEng            |      1.0000 |            0.8278 |     -0.1722 |                    0.2723 |
| Phi-4         |    20 |      5 | Among Us      | Lying             |      0.8627 |            0.8817 |     +0.0190 |                    0.6924 |
| Phi-4         |    20 |      5 | Among Us      | Deception         |      1.0000 |            0.9884 |     -0.0116 |                    0.7766 |
| Phi-4         |    20 |      5 | TruthfulQA    | TruthfulQA        |      0.8272 |            0.9212 |     +0.0940 |                    0.6531 |
| Phi-4         |    20 |      5 | TruthfulQA    | DishonestQA       |      0.5987 |            0.2777 |     -0.3210 |                    0.5255 |
| Phi-4         |    20 |      5 | TruthfulQA    | RepEng            |      0.8242 |            0.8710 |     +0.0468 |                    0.6674 |
| Phi-4         |    20 |      5 | TruthfulQA    | Lying             |      0.8609 |            0.4400 |     -0.4209 |                    0.4977 |
| Phi-4         |    20 |      5 | TruthfulQA    | Deception         |      0.9404 |            0.4788 |     -0.4617 |                    0.5142 |
| Phi-4         |    20 |      5 | DishonestQA   | TruthfulQA        |      0.5872 |            0.6279 |     +0.0406 |                    0.3818 |
| Phi-4         |    20 |      5 | DishonestQA   | DishonestQA       |      1.0000 |            0.9997 |     -0.0003 |                    0.1959 |
| Phi-4         |    20 |      5 | DishonestQA   | RepEng            |      0.9853 |            0.9907 |     +0.0054 |                    0.1410 |
| Phi-4         |    20 |      5 | DishonestQA   | Lying             |      0.8328 |            0.7216 |     -0.1113 |                    0.4332 |
| Phi-4         |    20 |      5 | DishonestQA   | Deception         |      0.7717 |            0.6239 |     -0.1479 |                    0.4862 |
| Phi-4         |    20 |      5 | RepEng        | TruthfulQA        |      0.5482 |            0.6012 |     +0.0530 |                    0.5410 |
| Phi-4         |    20 |      5 | RepEng        | DishonestQA       |      0.9990 |            0.9625 |     -0.0365 |                    0.8268 |
| Phi-4         |    20 |      5 | RepEng        | RepEng            |      1.0000 |            1.0000 |     +0.0000 |                    0.9704 |
| Phi-4         |    20 |      5 | RepEng        | Lying             |      0.9415 |            0.7536 |     -0.1879 |                    0.6088 |
| Phi-4         |    20 |      5 | RepEng        | Deception         |      0.9956 |            0.6479 |     -0.3478 |                    0.5392 |
| **Cell mean** |     — |      — | —             | —                 |  **0.8468** |        **0.7339** | **-0.1129** |                **0.5462** |

## Cross-model extension

The extension keeps the released five-token probe, four training datasets,
ordered splits, optimizer, and evaluation protocol. Means use the same 20 cells
as Figure 7: TruthfulQA, DishonestQA, RepEng, lying, and deception; role
prediction is retained in the full results but excluded from this comparison.

| Phase                | Model                 |     Layer | Relative depth | Mean AUROC | Mean accuracy | Mean precision | Mean recall |    Mean F1 |
| :------------------- | :-------------------- | --------: | -------------: | ---------: | ------------: | -------------: | ----------: | ---------: |
| Paper reported       | Phi-4                 |     20/40 |           0.50 |     0.8468 |             — |              — |           — |          — |
| Replication baseline | Phi-4                 |     20/40 |           0.50 |     0.7339 |             — |              — |           — |          — |
| Extension            | Llama-3.1-8B-Instruct |      8/32 |           0.25 |     0.6895 |        0.6527 |         0.7536 |      0.6867 |     0.6658 |
| Extension            | Llama-3.1-8B-Instruct | **16/32** |       **0.50** | **0.7617** |    **0.7043** |         0.7675 |      0.7140 |     0.6840 |
| Extension            | Llama-3.1-8B-Instruct |     24/32 |           0.75 |     0.7226 |        0.7000 |     **0.8037** |  **0.7194** | **0.6970** |
| Extension            | Gemma-2-27B-IT        |     10/46 |           0.22 |     0.6775 |        0.5764 |         0.7085 |      0.4926 |     0.5315 |
| Extension            | Gemma-2-27B-IT        | **22/46** |       **0.48** | **0.8011** |        0.6895 |         0.6907 |      0.6849 |     0.6627 |
| Extension            | Gemma-2-27B-IT        |     34/46 |           0.74 |     0.7659 |    **0.7303** |     **0.8079** |  **0.7416** | **0.7187** |
| Extension            | Qwen2.5-32B-Instruct  |     16/64 |           0.25 |     0.6277 |        0.5754 |         0.6940 |      0.5238 |     0.5480 |
| Extension            | Qwen2.5-32B-Instruct  |     32/64 |           0.50 |     0.7268 |    **0.6708** |         0.7111 |  **0.7000** | **0.6449** |
| Extension            | Qwen2.5-32B-Instruct  | **48/64** |       **0.75** | **0.7277** |        0.6134 |     **0.7238** |      0.5807 |     0.5683 |

**Llama result.** Mid-depth L16 is strongest by mean AUROC: `0.7617`, versus
`0.6895` at L8 and `0.7226` at L24. It exceeds the reconstructed Phi-4
baseline by `+0.0278`, but remains `-0.0851` below the paper's reported mean.
The L16 in-domain mean is `0.9666`; its cross-dataset mean is `0.7105`, so the
main limitation remains transfer rather than fitting.

**Gemma result.** Mid-depth L22 is strongest by mean AUROC: `0.8011`, versus
`0.6775` at L10 and `0.7659` at L34. It exceeds the reconstructed Phi-4
baseline by `+0.0672` and the best Llama layer by `+0.0393`, while remaining
`-0.0458` below the paper mean. L22 reaches `0.9847` in-domain and `0.7552`
cross-dataset mean AUROC. L34 has the best mean accuracy and F1, but weaker
ranking performance than L22.

**Qwen result.** L48 has the highest mean AUROC at `0.7277`, only `0.0008`
above L32; L32 is the stronger balanced operating point by accuracy and F1.
Both remain just below the reconstructed Phi-4 baseline (`-0.0062` and
`-0.0071`) and substantially below Gemma L22. In-domain AUROC rises from
`0.8647` at L16 to `0.9815` at L48, while cross-dataset AUROC plateaus near
`0.665`, again separating probe fitting from transfer.

Plot-ready files:

- `data/results/among-us/probe_results_summary.csv` — paper, baseline, and one row per extension layer.
- `data/results/among-us/extension_probe_results.csv` — all 216 Llama, Gemma, and Qwen cells with AUROC, accuracy, precision, recall, and F1, including role prediction.

## Failure taxonomy

| Class                             | Finding           | Evidence                                                                                                                                                   |
| :-------------------------------- | :---------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implementation                    | **Passed**        | Released preprocessing classes and`train_probes.py` ran unchanged; all four caches and checkpoints were valid.                                             |
| In-domain learning                | **Mostly passed** | Replication AUROC: TruthfulQA`0.9212`, DishonestQA `0.9997`, RepEng `1.0000`, Among Us→deception `0.9884`.                                                 |
| Cross-dataset generalization      | **Failed**        | Largest gaps: TruthfulQA→deception`-0.4617`, TruthfulQA→lying `-0.4209`, RepEng→deception `-0.3478`, Among Us→DishonestQA `-0.3429`.                       |
| Released checkpoint compatibility | **Failed**        | Authors' DishonestQA checkpoint scored`0.1959` on DishonestQA and their Among Us checkpoint scored `0.7766` on deception using the reconstructed cache.    |
| Specification consistency         | **Failed**        | Paper/config uses 10 tokens; the released Figure 7 script uses 5. Released checkpoint ROC files also do not match the Figure 7 matrix.                     |
| Artifact completeness             | **Insufficient**  | Original activation chunks, generated DishonestQA continuations, original behavior-label file, model revision, and Transformers version were not released. |

**Interpretation.** The baseline failure is concentrated in transfer. The released code can reconstruct a coherent experiment, but it cannot
reconstruct the reported Figure 7 experiment or the activation space expected
by the shipped checkpoints.
