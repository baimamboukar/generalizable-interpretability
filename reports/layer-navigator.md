# LayerNavigator Replication Report

[Released code](https://github.com/Bryson-Arrot/LayerNavigator), inspected at revision `09e96c72`.

LayerNavigator selects intervention layers using a score that combines discriminability of positive and negative activation clusters with consistency of contrastive activation differences. It then applies steering vectors at the highest-scoring layers. The score reuses activations collected during steering-vector extraction.

## Original Experimental Design

| Component | Specification |
| :--- | :--- |
| Models | Llama-3-8B-Instruct and Qwen2.5-32B-Instruct |
| Data | Six Anthropic Persona behaviours |
| Split | 700 training, 200 validation and 100 test examples per behaviour |
| Steering vectors | Mean difference and PCA |
| Selected layers | One, three or five |
| Intervention | Last-token activation addition, coefficient 1.0 |
| Evaluation | Probability assigned to the behaviour-aligned Yes/No response |

## Release Audit

The checkout includes the six datasets and their train, validation and test partitions. The implementation requires model-path configuration and model-specific Yes/No token identifiers. Its recorded environment uses older PyTorch and Transformers versions, and the Qwen loader assumes locally available checkpoints.

## Replication Status

No quantitative baseline was completed in the recorded work. This study remains a historical activation-steering audit outside the 22-study probing corpus. The release inspection does not establish steering effectiveness or generalization to additional models.
