# Neural Chameleons Replication

**Paper:** [Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors](https://arxiv.org/abs/2512.11949)
**Code:** [Official repository](https://github.com/mgm52/neural-chameleons)
**Status:** Released demo verified; official paper targets frozen; independent
baseline paused because required pretrained probes are missing from the release.

## Replication scope

We will first test the paper's main Gemma-2-9B layer-12 model organism against
the released concept, harmfulness, and deception probes. The initial runnable
stage uses the authors' linked minimal reproduction unchanged. It is a release
integrity check; the paper-level baseline will remain separate.

| Artifact                                    | Release status | Replication treatment         |
| :------------------------------------------ | :------------- | :---------------------------- |
| Full training/evaluation code               | Available      | Pinned unchanged              |
| Rated synthetic and UltraChat-derived data  | Available      | Pinned unchanged              |
| Pretrained probes required by full pipeline | Missing        | Do not substitute silently    |
| Finetuned Gemma-2-9B checkpoint             | Available      | Run author-linked demo first  |
| Thirteen logistic probe weights             | Available      | Run author-linked demo first  |
| Official result CSVs and plot code          | Available      | Regenerate as a release check |

## Result table

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

Plot-ready artifacts:

- `data/results/neural-chameleons/baseline_comparison.csv` — 27 paper-target
  rows, including confidence intervals and empty replication/delta columns.
- `data/results/neural-chameleons/released_demo_results.csv` — released demo
  smoke test.

The raw upstream stdout and regenerated PDFs remain unchanged on the Modal volume.

## Setup finding

The clean official checkout cannot execute its advertised full-pipeline command:
the selected pretrained probe directories are not released. In addition, the
shipped `model_config_21preset.json` specifies one epoch, LR `8e-6`, and batch
size 1/2, whereas the paper appendix reports three epochs, LR `2e-5`, and batch
size 16. We therefore freeze these as specification/release gaps before choosing
the paper-faithful training reconstruction.
