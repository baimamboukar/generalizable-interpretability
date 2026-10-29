# Neural Chameleons Replication

**Paper:** [Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors](https://arxiv.org/abs/2512.11949)
**Code:** [Official repository](https://github.com/mgm52/neural-chameleons)
**Status:** Released demonstration verified. The independent paper-level
baseline is blocked by missing pretrained probes.

## Replication Scope

The completed experiment evaluates the released Gemma-2-9B layer-12 model
organism with the authors' minimal demonstration and thirteen probes. It tests
the behaviour of the released artifacts. It does not independently reconstruct
the training procedure or the paper-level evaluation.

| Artifact                                    | Release status | Replication treatment         |
| :------------------------------------------ | :------------- | :---------------------------- |
| Full training/evaluation code               | Available      | Pinned unchanged              |
| Rated synthetic and UltraChat-derived data  | Available      | Pinned unchanged              |
| Pretrained probes required by full pipeline | Missing        | Independent baseline blocked    |
| Finetuned Gemma-2-9B checkpoint             | Available      | Evaluated in released demonstration  |
| Thirteen logistic probe weights             | Available      | Evaluated in released demonstration  |
| Official result CSVs and plot code          | Available      | Regenerate as a release check |

## Result Table

| Phase          | Model                       | Layer | Probe set                         | Metric                             |              Paper | Replication |   Delta |
| :------------- | :-------------------------- | ----: | :-------------------------------- | :--------------------------------- | -----------------: | ----------: | ------: |
| Released demo  | Gemma-2-9B neural chameleon |    12 | 11 training + harmful + deception | Mean probe score, normal/triggered | —               | 0.8014 / 0.1930 | —       |
| Paper baseline | Gemma-2-9B neural chameleon |    12 | 15 Figure 2 monitors              | Mean TPR at 1% FPR, normal/triggered | 0.9003 / 0.1492 | Pending       | Pending |

The released checkpoint passed its smoke test: all 13 probe scores decreased
under their matching trigger. Across the 11 training concepts, the mean score
fell from `0.7906` to `0.1782`; harmfulness fell from `0.8620` to `0.5006`
(`41.9%`) and deception from `0.8593` to `0.0483` (`94.4%`). This confirms the
released artifact's advertised behavior, but it does not reproduce Figure 2's
TPR at 1% FPR or independently recreate the trained model organism.

The authors' plotting code reproduced all four released scatter figures. At 1%
FPR, the 11 training concepts fell from mean TPR `0.9483` to `0.1583`; the four
unseen safety probes fell from `0.7683` to `0.1243`. These are targets recovered
from released result files, not an independent replication result.

## Release Limitations

The clean official checkout cannot execute its advertised full-pipeline command:
the selected pretrained probe directories are not released. In addition, the
shipped `model_config_21preset.json` specifies one epoch, LR `8e-6`, and batch
size 1/2, whereas the paper appendix reports three epochs, LR `2e-5`, and batch
size 16. These specification and artifact gaps limit independent reconstruction
of the paper-level experiment.
