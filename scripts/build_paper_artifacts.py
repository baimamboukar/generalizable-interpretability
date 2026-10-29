#!/usr/bin/env python3
"""Build manuscript tables from validated replication artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/paper-artifacts"


STUDIES = [
    {
        "study_id": "models-under-pressure",
        "title": "Detecting High-Stakes Interactions using Activation Probes",
        "year": "2025",
        "venue": "NeurIPS",
        "analysis_role": "primary anchor",
        "probe_focus": "attention and softmax activation probes",
        "original_models": "Llama-3.3-70B",
        "datasets": "synthetic GPT-4o training; Anthropic HH; ToolACE; MT; MTS; Mental Health; Aya Redteaming",
        "data_regime": "fixed released evaluation prompts; no monitored-model trajectories",
        "probe_architecture": "attention probe; softmax probe",
        "activation_site": "layer 31; paper says pre-layernorm while released hook is post-layernorm",
        "baseline_status": "headline reproduced with activation-site mismatch",
        "extension_status": "sealed OLMo-2 holdout succeeded with development-only selection",
        "evidence_grade": "C",
        "headline_eligible": "yes, with caveat",
        "paper_url": "https://arxiv.org/abs/2506.10805",
        "code_url": "",
        "report_path": "reports/model-under-pressure.md",
        "primary_gap": "the OLMo-2 activation site is an architecture-mapped analogue rather than the paper model's exact hook",
    },
    {
        "study_id": "internal-instruction-following",
        "title": "Do LLMs Know Internally When They Follow Instructions?",
        "year": "2025",
        "venue": "ICLR",
        "analysis_role": "primary anchor",
        "probe_focus": "instruction-following success probes",
        "original_models": "Llama-2-7B/13B-Chat; Mistral-7B-Instruct-v0.3; Phi-3-Mini",
        "datasets": "IFEval-simple",
        "data_regime": "fixed prompts; each model generates and labels its own answers",
        "probe_architecture": "bias-free linear sigmoid probe",
        "activation_site": "first, middle, and last response tokens at early, middle, and late layers",
        "baseline_status": "qualitative and aggregate baseline reproduced; exact table partial",
        "extension_status": "unseen tasks transfer; unseen instruction types unstable",
        "evidence_grade": "C",
        "headline_eligible": "yes, with caveat",
        "paper_url": "https://arxiv.org/abs/2410.14516",
        "code_url": "https://github.com/apple/ml-internal-llms-instruction-following",
        "report_path": "reports/internal-instruction-following.md",
        "primary_gap": "changing models changes response and label support; some held-out folds become single-class",
    },
    {
        "study_id": "political-perspective",
        "title": "Linear Representations of Political Perspective Emerge in Large Language Models",
        "year": "2025",
        "venue": "ICLR",
        "analysis_role": "primary anchor",
        "probe_focus": "attention-head ridge probes",
        "original_models": "Llama-2-7B-Chat; Mistral-7B-Instruct-v0.1; Vicuna-7B-v1.5",
        "datasets": "116th U.S. Congress speeches and DW-NOMINATE scores; news transfer",
        "data_regime": "fixed text and labels",
        "probe_architecture": "ridge regression per attention head; top-head ensemble",
        "activation_site": "final prompt token at every attention head",
        "baseline_status": "successful",
        "extension_status": "mixed oracle peaks; architecture-fair transfer unresolved",
        "evidence_grade": "C",
        "headline_eligible": "yes, with caveat",
        "paper_url": "https://proceedings.iclr.cc/paper_files/paper/2025/file/1557cc3e1a67f985dbdb162db9963dd0-Paper-Conference.pdf",
        "code_url": "https://github.com/JunsolKim/RepresentationPoliticalLLM",
        "report_path": "reports/political-perspective.md",
        "primary_gap": "target-test maxima give models with more heads and layers larger search budgets",
    },
    {
        "study_id": "prism",
        "title": "PRISM: Prompt-Guided Internal States for Hallucination Detection",
        "year": "2025",
        "venue": "ACL",
        "analysis_role": "positive control",
        "probe_focus": "prompt-guided factuality probes",
        "original_models": "Llama-2-7B-Chat; Llama-2-13B-Chat",
        "datasets": "True-False; LogicStruct",
        "data_regime": "fixed statements and labels",
        "probe_architecture": "MM difference-of-means direction; SAPLMA four-layer MLP",
        "activation_site": "final layer and last prompt token",
        "baseline_status": "successful",
        "extension_status": "prompt gain transferred in all tested comparisons",
        "evidence_grade": "B",
        "headline_eligible": "yes",
        "paper_url": "https://arxiv.org/abs/2411.04847",
        "code_url": "https://github.com/fujie-math/PRISM",
        "report_path": "reports/prism.md",
        "primary_gap": "no material gap for the tested claim",
    },
    {
        "study_id": "lookback-lens",
        "title": "Lookback Lens: Detecting and Mitigating Contextual Hallucinations Using Only Attention Maps",
        "year": "2024",
        "venue": "EMNLP",
        "analysis_role": "positive control",
        "probe_focus": "logistic regression over attention lookback ratios",
        "original_models": "Llama-2-7B-Chat; Llama-2-13B-Chat",
        "datasets": "Natural Questions; CNN/DailyMail",
        "data_regime": "fixed released responses, spans, and GPT-4o labels",
        "probe_architecture": "L2 logistic regression",
        "activation_site": "all layers and heads; span-level attention ratios",
        "baseline_status": "successful",
        "extension_status": "cross-dataset transfer remained in the baseline band",
        "evidence_grade": "B",
        "headline_eligible": "yes",
        "paper_url": "https://aclanthology.org/2024.emnlp-main.84/",
        "code_url": "https://github.com/voidism/Lookback-Lens",
        "report_path": "reports/lookback-lens.md",
        "primary_gap": "13B transfer artifacts were not released; tested extension uses regenerated features",
    },
    {
        "study_id": "beyond-linear-probes",
        "title": "Beyond Linear Probes: Dynamic Safety Monitoring for Language Models",
        "year": "2026",
        "venue": "ICLR",
        "analysis_role": "positive control",
        "probe_focus": "truncated polynomial safety probes",
        "original_models": "Gemma-3-27B-it; Qwen3-30B; GPT-OSS-20B; Llama-3.2-3B",
        "datasets": "WildGuardMix; BeaverTails",
        "data_regime": "fixed public prompts and labels",
        "probe_architecture": "linear; bilinear; MLP; early-exit MLP; fifth-order TPC",
        "activation_site": "mean-pooled residual stream at released validation-selected layers",
        "baseline_status": "selected Llama condition reproduced",
        "extension_status": "TPC beat linear across three models but was not always best",
        "evidence_grade": "C",
        "headline_eligible": "yes, selected condition only",
        "paper_url": "https://arxiv.org/abs/2509.26238",
        "code_url": "https://github.com/james-oldfield/tpc",
        "report_path": "reports/beyond-linear-probes.md",
        "primary_gap": "only one dataset and one baseline model-layer condition was independently rerun",
    },
    {
        "study_id": "among-us",
        "title": "Among Us: A Sandbox for Measuring and Detecting Agentic Deception",
        "year": "2025",
        "venue": "arXiv",
        "analysis_role": "baseline failure",
        "probe_focus": "cross-dataset linear deception probes",
        "original_models": "Phi-4",
        "datasets": "TruthfulQA; DishonestQA; RepEng; Among Us",
        "data_regime": "fixed released prompts and labels",
        "probe_architecture": "bias-free linear probe",
        "activation_site": "layer 20; five final tokens in released Figure 7 path",
        "baseline_status": "not reproduced",
        "extension_status": "diagnostic only",
        "evidence_grade": "D",
        "headline_eligible": "baseline only",
        "paper_url": "https://arxiv.org/abs/2504.04072",
        "code_url": "https://github.com/7vik/AmongUs",
        "report_path": "reports/among-us.md",
        "primary_gap": "paper and code disagree on token count; released checkpoints also fail on reconstructed caches",
    },
    {
        "study_id": "sycophancy",
        "title": "Sycophancy Hides Linearly in the Attention Heads",
        "year": "2026",
        "venue": "EACL",
        "analysis_role": "baseline failure",
        "probe_focus": "attention-head, MLP, and residual-stream probes",
        "original_models": "Gemma-3; Llama-3.2",
        "datasets": "released sycophancy question-answer pairs",
        "data_regime": "fixed released examples and labels",
        "probe_architecture": "linear classifiers",
        "activation_site": "all layers; component-specific activations",
        "baseline_status": "not reproduced",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "baseline only",
        "paper_url": "https://arxiv.org/abs/2601.16644",
        "code_url": "https://github.com/rifoagenadi/sycophancy",
        "report_path": "reports/sycophancy.md",
        "primary_gap": "no paper checkpoints, activations, result dictionaries, or exact per-seed values",
    },
    {
        "study_id": "implicit-personalization",
        "title": "Reading Between the Prompts: How Stereotypes Shape LLMs' Implicit Personalization",
        "year": "2025",
        "venue": "EMNLP",
        "analysis_role": "supporting evidence",
        "probe_focus": "multiclass demographic probes",
        "original_models": "Gemma-2-9B; Llama-3.1-8B; OLMo-2-7B",
        "datasets": "released 250-conversation identity and stereotype conditions",
        "data_regime": "fixed released conversations and labels",
        "probe_architecture": "multiclass logistic regression",
        "activation_site": "five prespecified interior layers",
        "baseline_status": "reduced selected-layer replication successful",
        "extension_status": "explicit-identity decodability transferred; main stereotype claim untested",
        "evidence_grade": "C",
        "headline_eligible": "supporting only",
        "paper_url": "https://arxiv.org/abs/2505.16467",
        "code_url": "https://github.com/Veranep/implicit-personalization-stereotypes",
        "report_path": "reports/implicit-personalization.md",
        "primary_gap": "the reduced run does not reproduce the main stereotype and steering claims",
    },
    {
        "study_id": "llms-know",
        "title": "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations",
        "year": "2025",
        "venue": "ICLR",
        "analysis_role": "post-freeze positive control",
        "probe_focus": "answer-correctness probes",
        "original_models": "Mistral-7B and Instruct; Llama-3-8B and Instruct",
        "datasets": "TriviaQA; HotpotQA; Movies; WinoGrande; WinoBias; MNLI; Math; IMDB; Natural Questions",
        "data_regime": "fixed benchmark prompts; model-generated answers and model-dependent labels",
        "probe_architecture": "L2 logistic regression",
        "activation_site": "selected MLP layer and answer-token position",
        "baseline_status": "principal exact-answer claim reproduced",
        "extension_status": "strong transfer; magnitude depends on response regime",
        "evidence_grade": "B",
        "headline_eligible": "yes, post-freeze",
        "paper_url": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/a712d461e57201efe35d429a6f1731c1-Abstract-Conference.html",
        "code_url": "https://github.com/technion-cs-nlp/LLMsKnow",
        "report_path": "reports/llms-know.md",
        "primary_gap": "generated responses were not released; target-model site selection remains model-specific",
    },
    {
        "study_id": "test-awareness-steering",
        "title": "The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness",
        "year": "2025",
        "venue": "NeurIPS",
        "analysis_role": "supporting evidence",
        "probe_focus": "evaluation-awareness probes",
        "original_models": "QwQ-32B; Qwen3-32B; DeepSeek-R1-Distill-Qwen-32B",
        "datasets": "paired real and hypothetical harmful requests with labeled awareness spans",
        "data_regime": "released on-policy traces, judge labels, and evidence spans",
        "probe_architecture": "affine softmax probe",
        "activation_site": "every hidden-state index; evidence-span mean",
        "baseline_status": "Qwen3 selected-model baseline reproduced",
        "extension_status": "not run",
        "evidence_grade": "C",
        "headline_eligible": "supporting only",
        "paper_url": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/cf42f133f355e0e07a8957b508b26a1b-Abstract-Conference.html",
        "code_url": "https://github.com/microsoft/Test_Awareness_Steering",
        "report_path": "reports/test-awareness-steering.md",
        "primary_gap": "peak accuracy reproduced but the selected peak layer moved from 19 to 55",
    },
    {
        "study_id": "neural-chameleons",
        "title": "Neural Chameleons",
        "year": "2025",
        "venue": "arXiv",
        "analysis_role": "artifact audit",
        "probe_focus": "probe-aware self-obfuscation",
        "original_models": "Gemma-2-9B",
        "datasets": "synthetic concepts; harmfulness; RepE deception",
        "data_regime": "released demo inputs and probes",
        "probe_architecture": "released concept probes",
        "activation_site": "layer 12",
        "baseline_status": "released demo verified; independent baseline unavailable",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://arxiv.org/abs/2512.11949",
        "code_url": "https://github.com/mgm52/neural-chameleons",
        "report_path": "reports/neural-chameleons.md",
        "primary_gap": "pretrained probe artifacts needed for the complete paper baseline were not released",
    },
    {
        "study_id": "propositional-probes",
        "title": "Monitoring Latent World States in Language Models with Propositional Probes",
        "year": "2025",
        "venue": "ICLR",
        "analysis_role": "artifact audit",
        "probe_focus": "domain probes and binding subspace",
        "original_models": "Tulu-2-13B",
        "datasets": "synthetic propositions; paraphrased English; Spanish translation",
        "data_regime": "synthetic fixed labels; two evaluation text variants missing",
        "probe_architecture": "linear domain probes plus Hessian-derived binding subspace",
        "activation_site": "domain probes at layer 20; binding subspace at layer 15",
        "baseline_status": "blocked by missing evaluation texts",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://arxiv.org/abs/2406.19501",
        "code_url": "https://github.com/jiahai-feng/prop-probes-iclr",
        "report_path": "reports/propositional-probes.md",
        "primary_gap": "the exact GPT-3.5 paraphrased and Spanish evaluation texts were not released",
    },
    {
        "study_id": "icr-probe",
        "title": "ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs",
        "year": "2025",
        "venue": "ACL",
        "analysis_role": "artifact audit",
        "probe_focus": "cross-layer residual-stream dynamics",
        "original_models": "Llama-3-8B; Qwen2.5-7B; Gemma-2-9B; Qwen 3B/14B",
        "datasets": "HaluEval QA; SQuAD; HotpotQA; TriviaQA",
        "data_regime": "model-generated answers and labels not released",
        "probe_architecture": "small MLP over token-pooled cross-layer ICR scores",
        "activation_site": "all layers and top-20 answer tokens",
        "baseline_status": "blocked by missing outputs and labels",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://arxiv.org/abs/2507.16488",
        "code_url": "https://github.com/XavierZhang2002/ICR_Probe",
        "report_path": "reports/icr-probe.md",
        "primary_gap": "generated answers, hallucination labels, caching pipeline, baseline runner, and checkpoints are absent",
    },
    {
        "study_id": "truthful-representation-flip",
        "title": "When Truthful Representations Flip Under Deceptive Instructions?",
        "year": "2025",
        "venue": "EMNLP",
        "analysis_role": "artifact audit",
        "probe_focus": "truth-value probes under deceptive instructions",
        "original_models": "Llama-3.1-8B-Instruct; Gemma-2-9B-Instruct",
        "datasets": "balanced truth datasets across topic pairs and instruction conditions",
        "data_regime": "fixed examples; labels and prompts differ between paper and release",
        "probe_architecture": "z-scored logistic regression and TTPD probes",
        "activation_site": "final pre-generation residual state at every layer",
        "baseline_status": "blocked by protocol mismatch",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://arxiv.org/abs/2507.22149",
        "code_url": "https://github.com/ivyllll/truthful-representation-flip",
        "report_path": "reports/truthful-representation-flip.md",
        "primary_gap": "prompts, regression settings, deceptive labels, activation paths, and revisions do not match the paper",
    },
    {
        "study_id": "caught-in-the-act",
        "title": "Caught in the Act: A Mechanistic Approach to Detecting Deception",
        "year": "2025",
        "venue": "arXiv",
        "analysis_role": "artifact audit",
        "probe_focus": "linear deception probes and INLP",
        "original_models": "Qwen2.5 1.5B/3B/7B/14B; DeepSeek-R1 1.5B/7B",
        "datasets": "synthetic deceptive and non-deceptive arguments",
        "data_regime": "generation pipeline not verified",
        "probe_architecture": "linear residual-stream probes; INLP",
        "activation_site": "final-token residual stream across layers",
        "baseline_status": "setup only; release unverified",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://arxiv.org/abs/2508.19505",
        "code_url": "",
        "report_path": "reports/caught-in-the-act.md",
        "primary_gap": "no verified upstream code and artifact release",
    },
    {
        "study_id": "reasoning-trajectory",
        "title": "LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals",
        "year": "2026",
        "venue": "ACL",
        "analysis_role": "artifact audit",
        "probe_focus": "reasoning-step and correctness probes",
        "original_models": "Llama-3.1-8B; Llama-3.1-8B-Instruct; DeepSeek-R1-Distill-Llama-8B",
        "datasets": "GSM8K reasoning traces",
        "data_regime": "model-generated reasoning traces",
        "probe_architecture": "one-vs-rest logistic regression; correctness predictor",
        "activation_site": "all 32 layers before step and answer markers",
        "baseline_status": "blocked by release runtime mismatch",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://aclanthology.org/2026.acl-long.1237/",
        "code_url": "https://github.com/slhleosun/reasoning-trajectory",
        "report_path": "reports/reasoning-trajectory.md",
        "primary_gap": "generation settings conflict and the released runtime exhausted 141 GB before producing an artifact",
    },
    {
        "study_id": "disentangle-sycophancy",
        "title": "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs",
        "year": "2025",
        "venue": "arXiv",
        "analysis_role": "artifact audit",
        "probe_focus": "difference-in-means behavioral directions",
        "original_models": "Qwen3-30B/4B; Llama-3.1-8B; Llama-3.3-70B; GPT-OSS-20B",
        "datasets": "controlled sycophantic agreement, genuine agreement, and praise data",
        "data_regime": "controlled model-generated behavioral data",
        "probe_architecture": "difference-in-means linear directions",
        "activation_site": "post-layernorm residual stream across layers",
        "baseline_status": "first attempt blocked by environment mismatch",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://arxiv.org/abs/2509.21305",
        "code_url": "https://github.com/cincynlp/disentangle-sycophancy",
        "report_path": "reports/disentangle-sycophancy.md",
        "primary_gap": "release has import, steering-file, layer-default, dataset-size, and precision mismatches",
    },
    {
        "study_id": "layer-navigator",
        "title": "LayerNavigator: Finding Promising Intervention Layers for Efficient Activation Steering",
        "year": "2025",
        "venue": "NeurIPS",
        "analysis_role": "out of scope",
        "probe_focus": "steering-layer selection",
        "original_models": "Llama-3-8B-Instruct; Qwen2.5-32B-Instruct",
        "datasets": "Anthropic Persona",
        "data_regime": "fixed contrastive steering examples",
        "probe_architecture": "Fisher discriminability and direction-consistency scores",
        "activation_site": "all residual-stream layers",
        "baseline_status": "setup only",
        "extension_status": "out of probing scope",
        "evidence_grade": "D",
        "headline_eligible": "no",
        "paper_url": "",
        "code_url": "https://github.com/Bryson-Arrot/LayerNavigator",
        "report_path": "reports/layer-navigator.md",
        "primary_gap": "the central method is activation steering rather than probing",
    },
    {
        "study_id": "arithmetic-error-probing",
        "title": "Probing for Arithmetic Errors in Language Models",
        "year": "2025",
        "venue": "EMNLP",
        "analysis_role": "artifact audit",
        "probe_focus": "arithmetic digit and error probes",
        "original_models": "Gemma-2-2B-IT; Phi-3-Mini",
        "datasets": "three-digit addition; structured GSM8K addition steps",
        "data_regime": "generated arithmetic samples and traces not released",
        "probe_architecture": "circular; ridge; multiclass logistic; one-hidden-layer MLP",
        "activation_site": "equals-sign token at all 26 layers",
        "baseline_status": "blocked by paper-code and artifact gaps",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://aclanthology.org/2025.emnlp-main.411/",
        "code_url": "https://github.com/yuchen9-5un/arithmetic-error-probing",
        "report_path": "reports/arithmetic-error-probing.md",
        "primary_gap": "Figure 3 probes are disabled and exact samples, activations, checkpoints, results, and environment are absent",
    },
    {
        "study_id": "thought-calibration",
        "title": "Thought Calibration: Efficient and Confident Test-Time Scaling",
        "year": "2025",
        "venue": "EMNLP",
        "analysis_role": "artifact audit",
        "probe_focus": "reasoning-step stopping probes",
        "original_models": "DeepSeek-R1-Distill-Qwen-32B; DeepSeek-R1-Distill-Llama-70B; QwQ-32B",
        "datasets": "s1K-1.1; AIME 24; GPQA Diamond; MATH-500",
        "data_regime": "released step embeddings and labels for linear path",
        "probe_architecture": "PCA plus logistic regression; unreleased MLP and Transformer probes",
        "activation_site": "mean final-layer hidden state per reasoning step",
        "baseline_status": "linear path available; architecture claim artifact-limited",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://aclanthology.org/2025.emnlp-main.722/",
        "code_url": "https://github.com/rmwu/thought-calibration",
        "report_path": "reports/thought-calibration.md",
        "primary_gap": "MLP and Transformer implementations and checkpoints are missing",
    },
    {
        "study_id": "redeep",
        "title": "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability",
        "year": "2025",
        "venue": "ICLR",
        "analysis_role": "artifact audit",
        "probe_focus": "copying-head and FFN hallucination scores",
        "original_models": "Llama-2-7B-Chat; Llama-2-13B-Chat; Llama-3-8B-Instruct",
        "datasets": "RAGTruth; Dolly with Accurate Context",
        "data_regime": "fixed datasets; incomplete released outputs",
        "probe_architecture": "selected copying-head and FFN score combination",
        "activation_site": "selected attention heads and FFN layers",
        "baseline_status": "blocked by released pipeline defects",
        "extension_status": "not run",
        "evidence_grade": "D",
        "headline_eligible": "appendix only",
        "paper_url": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/7daf60e805e596c3bd1e843e72ea5560-Abstract-Conference.html",
        "code_url": "https://github.com/Jeryi-Sun/ReDEeP-ICLR",
        "report_path": "reports/redeep.md",
        "primary_gap": "Table 1 code has leakage, fatal regressions, and broken Dolly paths",
    },
    {
        "study_id": "verbal-uncertainty",
        "title": "Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations",
        "year": "2025",
        "venue": "EMNLP",
        "analysis_role": "supporting evidence",
        "probe_focus": "verbal-uncertainty and semantic-entropy probes",
        "original_models": "Llama-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3; Qwen2.5-7B-Instruct",
        "datasets": "TriviaQA RC; NQ-Open; PopQA",
        "data_regime": "fixed questions; model-specific sampled answers and labels",
        "probe_architecture": "linear regression plus logistic hallucination detector",
        "activation_site": "concatenated last-question-token residual states from selected layers",
        "baseline_status": "Table 2 primary baseline regenerated with mixed fidelity",
        "extension_status": "not run",
        "evidence_grade": "C",
        "headline_eligible": "supporting only",
        "paper_url": "https://aclanthology.org/2025.emnlp-main.187/",
        "code_url": "https://github.com/facebookresearch/verbal_uncertainty_feature_calibration",
        "report_path": "reports/verbal-uncertainty.md",
        "primary_gap": "paper outputs are unreleased and stochastic, so this is a protocol replication rather than an exact artifact replay",
    },
]


CONTRACTS = {
    "models-under-pressure": ["fixed", "not applicable", "fixed external", "target-selected", "fixed", "not retained", "diagnostic oracle"],
    "internal-instruction-following": ["fixed", "on-policy", "model-dependent", "prespecified sparse", "fixed", "not retained for legacy baseline", "method portability with policy shift"],
    "political-perspective": ["fixed", "not applicable", "fixed", "target-test selected", "fixed", "not applicable", "diagnostic oracle"],
    "prism": ["fixed", "not applicable", "fixed", "fixed final layer", "fixed", "paper default", "method portability"],
    "lookback-lens": ["fixed released", "fixed released", "fixed GPT-4o", "all layers and heads", "fixed", "paper default", "method portability"],
    "beyond-linear-probes": ["fixed", "not applicable", "fixed", "validation-selected", "compared on every model", "validation-selected", "architecture portability"],
    "among-us": ["fixed", "not applicable", "fixed", "target-test selected", "fixed", "paper default", "diagnostic only after baseline failure"],
    "sycophancy": ["fixed", "not applicable", "fixed", "all layers", "fixed", "validation accuracy", "not run"],
    "implicit-personalization": ["fixed", "fixed released", "fixed", "prespecified sparse", "fixed", "paper default", "reduced decodability check"],
    "llms-know": ["fixed", "on-policy and fixed-response control", "model-dependent and fixed-response control", "validation-selected", "fixed", "validation-selected", "method portability plus response-policy control"],
    "test-awareness-steering": ["fixed released", "fixed released traces", "fixed released judge labels", "all layers; target-test peak reported", "fixed", "0.5", "baseline replication only"],
    "verbal-uncertainty": ["fixed questions", "model-specific generated answers", "model-dependent Llama-70B judge labels", "paper-selected layer ranges", "fixed", "0.5", "baseline replication only"],
}


COVERAGE = {
    "models-under-pressure": ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes"],
    "internal-instruction-following": ["yes", "yes", "yes", "yes", "partial", "partial", "partial", "yes"],
    "political-perspective": ["yes", "yes", "yes", "yes", "yes", "not applicable", "not applicable", "partial"],
    "prism": ["yes", "yes", "yes", "no", "no", "no", "no", "no"],
    "lookback-lens": ["yes", "yes", "yes", "no", "no", "no", "no", "no"],
    "beyond-linear-probes": ["yes", "yes", "yes", "partial", "no", "no", "no", "yes"],
    "among-us": ["yes", "yes", "yes", "no", "no", "no", "no", "no"],
    "sycophancy": ["yes", "yes", "no", "no", "no", "no", "no", "yes"],
    "implicit-personalization": ["yes", "yes", "yes", "no", "no", "no", "no", "partial"],
    "llms-know": ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes"],
    "test-awareness-steering": ["yes", "yes", "no", "yes", "yes", "yes", "yes", "no"],
    "verbal-uncertainty": ["yes", "yes", "no", "yes", "yes", "yes", "yes", "no"],
}


BASELINE_FIELDS = [
    "study_id", "claim_id", "model", "dataset_or_setting", "probe_or_method", "metric",
    "paper_value", "replication_value", "replication_std", "n_seeds", "delta", "abs_delta",
    "comparison_level", "comparability", "independent_rerun", "source_path", "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty source: {relative}")
    return rows


def value(text: str, scale: float = 1.0) -> float:
    number = float(text) / scale
    if not math.isfinite(number):
        raise ValueError(text)
    return number


def fmt(number: float | None) -> str:
    return "" if number is None else f"{number:.10g}"


def baseline_row(
    study_id: str,
    claim_id: str,
    model: str,
    setting: str,
    probe: str,
    metric: str,
    paper: float,
    replication: float,
    source: str,
    *,
    std: float | None = None,
    seeds: str = "",
    level: str = "cell",
    comparability: str = "direct",
    independent: str = "yes",
    notes: str = "",
) -> dict[str, str]:
    delta = replication - paper
    return {
        "study_id": study_id,
        "claim_id": claim_id,
        "model": model,
        "dataset_or_setting": setting,
        "probe_or_method": probe,
        "metric": metric,
        "paper_value": fmt(paper),
        "replication_value": fmt(replication),
        "replication_std": fmt(std),
        "n_seeds": seeds,
        "delta": fmt(delta),
        "abs_delta": fmt(abs(delta)),
        "comparison_level": level,
        "comparability": comparability,
        "independent_rerun": independent,
        "source_path": source,
        "notes": notes,
    }


def build_baselines() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    source = "data/results/prism/mega_table.csv"
    for row in read_csv(source):
        if row["model"] not in {"Llama-2-7B-Chat", "Llama-2-13B-Chat"} or not row["paper_value"]:
            continue
        rows.append(baseline_row(
            "prism", f"{row['dataset']}:{row['method']}:{row['representation']}:{row['test_domain']}",
            row["model"], f"{row['dataset']} / {row['test_domain']}", row["method"], "accuracy",
            value(row["paper_value"]), value(row["replication_value"]), source,
        ))

    source = "data/results/lookback-lens/table2_replication.csv"
    for row in read_csv(source):
        rows.append(baseline_row(
            "lookback-lens", f"table2:{row['setting']}:{row['source']}:{row['target']}:{row['split']}",
            row["model"], f"{row['setting']} / {row['source']} to {row['target']} / {row['split']}",
            "lookback logistic regression", row["metric"].lower(), value(row["paper"]),
            value(row["replication"]), source,
        ))

    source = "data/results/political-perspective/independent_baseline.csv"
    for row in read_csv(source):
        rows.append(baseline_row(
            "political-perspective", "figure2:peak_single_head", row["model"],
            f"paper L{row['paper_layer']}H{row['paper_head']}; replication L{row['replication_layer']}H{row['replication_head']}",
            "single-head ridge regression", "spearman", value(row["paper"]), value(row["replication"]), source,
        ))
    source = "data/results/political-perspective/remaining_baseline_claims.csv"
    for row in read_csv(source):
        if not row["paper"] or not row["replication"]:
            continue
        rows.append(baseline_row(
            "political-perspective", row["target"].lower().replace(" ", "_"), row["model"], row["target"],
            "ridge regression", "spearman", value(row["paper"]), value(row["replication"]), source,
        ))

    source = "data/results/internal-instruction-following/baseline_comparison.csv"
    for row in read_csv(source):
        if not row["paper_mean"] or not row["replication_mean"]:
            continue
        rows.append(baseline_row(
            "internal-instruction-following",
            f"table{row['table']}:{row['experiment']}:{row['layer_group']}:{row['token_position']}",
            row["model"], f"{row['experiment']} / {row['layer_group']} / {row['token_position']}",
            "bias-free linear sigmoid", row["metric"], value(row["paper_mean"]), value(row["replication_mean"]), source,
            std=value(row["replication_std"]) if row["replication_std"] else None,
            seeds=row["runs"], comparability="direct" if row["status"] == "complete" else "partial",
            notes=row["status"],
        ))

    source = "data/results/beyond-linear-probes/baseline_replication.csv"
    for row in read_csv(source):
        rows.append(baseline_row(
            "beyond-linear-probes", f"table1:{row['method']}:{row['split']}", row["model"],
            f"{row['dataset']} / layer {row['layer']} / {row['split']}", row["method"], row["metric"].lower(),
            value(row["paper_value_pct"], 100), value(row["replication_value_pct"], 100), source,
            seeds=row["seeds"],
        ))

    source = "data/results/among-us/figure7_baseline_comparison.csv"
    for row in read_csv(source):
        rows.append(baseline_row(
            "among-us", f"figure7:{row['train_dataset']}:{row['eval_dataset']}", row["model"],
            f"train {row['train_dataset']} / evaluate {row['eval_dataset']}", "bias-free linear probe", "auroc",
            value(row["paper_auroc"]), value(row["replication_auroc"]), source,
            comparability="released-code path; paper-code token mismatch",
            notes=f"official checkpoint AUROC={row['official_checkpoint_auroc']}",
        ))

    source = "data/results/sycophancy/mega_table.csv"
    for row in read_csv(source):
        rows.append(baseline_row(
            "sycophancy", f"figure{row['figure']}:{row['component']}", row["model"],
            f"peak layer {row['paper_peak_layer']} to {row['replication_peak_layer']}", row["component"], "accuracy",
            value(row["paper_value"]), value(row["replication_mean"]), source,
            std=value(row["replication_sd"]), seeds=str(len(row["seeds"].split(";"))),
        ))

    source = "data/results/implicit-personalization/probe_validation_baseline.csv"
    for row in read_csv(source):
        rows.append(baseline_row(
            "implicit-personalization", f"figure{row['figure']}:{row['attribute']}:L{row['layer']}", row["model"],
            f"{row['attribute']} / layer {row['layer']}", row["probe"], "accuracy",
            value(row["paper_value_approx"], 100), value(row["replication_value"], 100), source,
            comparability="approximate digitized paper curve", notes="reduced five-layer replication",
        ))

    source = "data/results/llms-know/table1_mistral_replication.csv"
    paired: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(source):
        paired[row["dataset"]][row["source"]] = row
    for dataset, pair in paired.items():
        paper, replication = pair["Original paper"], pair["Replication"]
        rows.append(baseline_row(
            "llms-know", f"table1:exact_answer:{dataset}", "Mistral-7B-Instruct-v0.2", dataset,
            "L2 logistic regression", "auroc", value(paper["mean"]), value(replication["mean"]), source,
            std=value(replication["std"]), seeds="5",
        ))

    source = "data/results/test-awareness-steering/qwen3_paper_vs_replication.csv"
    evidence = {row["evidence"]: row for row in read_csv(source)}
    paper, replication = evidence["paper_table_2"], evidence["independent_replication"]
    rows.append(baseline_row(
        "test-awareness-steering", "table2:qwen3_peak", paper["model"],
        f"all-layer peak; independent index {replication['peak_layer']}", "affine softmax", "accuracy",
        value(paper["value"]), value(replication["value"]), source,
        seeds="1", comparability="direct peak magnitude; peak location unstable",
    ))

    source = "data/results/verbal-uncertainty/table2_probe_baseline.csv"
    for row in read_csv(source):
        if not row["replication_auroc"]:
            continue
        for metric, paper_key, replication_key in (
            ("auroc", "paper_auroc", "replication_auroc"),
            ("accuracy", "paper_accuracy", "replication_accuracy"),
        ):
            rows.append(baseline_row(
                "verbal-uncertainty",
                f"table2:{row['dataset']}:{metric}",
                row["model"],
                row["dataset"],
                row["method"],
                metric,
                value(row[paper_key]),
                value(row[replication_key]),
                source,
                seeds="1",
                comparability="protocol replication with regenerated stochastic artifacts",
            ))

    rows.extend([
        baseline_row(
            "models-under-pressure", "figure3:six_dataset_attention_mean", "Llama-3.3-70B", "six-dataset mean / layer 31",
            "attention", "auroc", 0.917, 0.9184, "reports/model-under-pressure.md",
            seeds="3", level="headline aggregate", comparability="approximate paper headline",
            notes="95% CI 0.9161 to 0.9207; paper-code activation-site mismatch",
        ),
        baseline_row(
            "models-under-pressure", "figure2:softmax_mean", "Llama-3.3-70B", "paper four-dataset vs replication six-dataset mean",
            "softmax", "auroc", 0.948, 0.9244, "reports/model-under-pressure.md",
            seeds="1", level="headline aggregate", comparability="not directly comparable aggregation",
            notes="retained for transparency; excluded from direct-fidelity summaries",
        ),
    ])
    return sorted(rows, key=lambda row: (row["study_id"], row["claim_id"], row["model"]))


def build_extensions() -> list[dict[str, str]]:
    source = "data/results/paper-synthesis/evidence_long.csv"
    rows = read_csv(source)
    fields = [
        "study", "manuscript_role", "claim_id", "phase", "comparison_unit", "model",
        "probe_or_method", "dataset_or_setting", "metric", "paper_baseline", "replication_baseline",
        "extension_result", "delta_extension_vs_replication", "delta_valid", "normalized_retention",
        "retention_definition", "retention_valid", "synthesis_grade", "outcome", "selection_regime",
        "source_path", "source_sha256", "caveat",
    ]
    output = []
    for row in rows:
        item = {field: row[field] for field in fields}
        item["study_id"] = next(study["study_id"] for study in STUDIES if study["title"].startswith(row["study"]) or {
            "Models Under Pressure": "models-under-pressure",
            "Internal Instruction Following": "internal-instruction-following",
            "Political Perspective": "political-perspective",
            "PRISM": "prism",
            "Lookback Lens": "lookback-lens",
            "Beyond Linear Probes": "beyond-linear-probes",
            "Among Us": "among-us",
            "Sycophancy": "sycophancy",
            "Implicit Personalization": "implicit-personalization",
            "LLMs Know": "llms-know",
        }.get(row["study"]) == study["study_id"])
        output.append(item)
    return output


def build_contracts() -> list[dict[str, str]]:
    fields = [
        "study_id", "examples_across_models", "responses_across_models", "labels_across_models",
        "activation_site_selection", "probe_architecture", "threshold_protocol", "extension_estimand",
        "interpretation",
    ]
    studies = {study["study_id"]: study for study in STUDIES}
    rows = []
    for study_id, values in CONTRACTS.items():
        row = dict(zip(fields[:-1], [study_id, *values], strict=True))
        row["interpretation"] = studies[study_id]["primary_gap"]
        rows.append(row)
    return rows


def build_audits() -> list[dict[str, str]]:
    rows = []
    for study in STUDIES:
        status = study["baseline_status"].lower()
        if "successful" in status or "reproduced" in status or "regenerated" in status:
            audit = "empirical evidence available"
        elif "blocked" in status:
            audit = "blocked after release audit"
        elif "setup" in status:
            audit = "setup or discovery only"
        elif "demo" in status:
            audit = "released diagnostic only"
        else:
            audit = "partial or in progress"
        rows.append({
            "study_id": study["study_id"],
            "release_audit_status": audit,
            "code_available": "yes" if study["code_url"] else "no or unverified",
            "data_or_required_artifacts": "complete enough" if study["headline_eligible"].startswith("yes") else "incomplete or not fully tested",
            "paper_code_agreement": "material gap" if any(term in study["primary_gap"] for term in ["mismatch", "disagree", "differ", "defect", "leakage", "conflict"]) else "no material gap established or not fully assessed",
            "primary_gap": study["primary_gap"],
            "report_path": study["report_path"],
        })
    return rows


def build_verdicts() -> list[dict[str, str]]:
    return [{
        "study_id": study["study_id"],
        "analysis_role": study["analysis_role"],
        "baseline_status": study["baseline_status"],
        "extension_status": study["extension_status"],
        "evidence_grade": study["evidence_grade"],
        "headline_eligible": study["headline_eligible"],
        "verdict": f"Baseline: {study['baseline_status']}. Extension: {study['extension_status']}.",
        "primary_reason_or_caveat": study["primary_gap"],
        "report_path": study["report_path"],
    } for study in STUDIES]


def build_coverage() -> list[dict[str, str]]:
    fields = [
        "study_id", "paper_targets", "baseline_metrics", "extension_metrics", "row_level_predictions",
        "confusion_matrix", "calibration", "low_fpr_metrics", "multi_seed_or_uncertainty",
    ]
    rows = []
    for study in STUDIES:
        values = COVERAGE.get(study["study_id"], ["no"] * 8)
        rows.append(dict(zip(fields, [study["study_id"], *values], strict=True)))
    return rows


def build_gaps() -> list[dict[str, str]]:
    rows = []
    for study in STUDIES:
        needs_experiment = study["headline_eligible"] in {"yes, with caveat", "yes, selected condition only", "no, incomplete"}
        completion_artifacts = {
            "models-under-pressure": ROOT / "data/results/model-under-pressure/confirmatory/olmo2-32b/manifest.json",
            "verbal-uncertainty": ROOT / "data/results/verbal-uncertainty/table2_replication_metrics.csv",
        }
        completed = completion_artifacts.get(study["study_id"], Path()).is_file()
        if study["study_id"] == "models-under-pressure":
            action = (
                "sealed OLMo-2 holdout completed with development-only layer selection and full threshold metrics"
                if completed
                else "evaluate sealed holdouts with development-only layer selection and full threshold metrics"
            )
        elif study["study_id"] == "internal-instruction-following":
            action = "run fixed-response controls on sealed holdouts and retain undefined label-support folds"
        elif study["study_id"] == "political-perspective":
            action = "apply an equal-budget nested head and layer selection rule on sealed holdouts"
        elif study["study_id"] == "beyond-linear-probes":
            action = "complete at least one additional paper-model baseline before a broad architecture claim"
        elif study["study_id"] == "verbal-uncertainty":
            action = (
                "Table 2 primary baseline completed with full threshold and row-level metrics"
                if completed
                else "finish the staged Table 2 baseline before inclusion"
            )
        elif study["analysis_role"] in {"artifact audit", "out of scope"}:
            action = "no GPU run planned; retain the audit and missing-artifact reason"
        else:
            action = "no mandatory new experiment for the current evidence claim"
        rows.append({
            "study_id": study["study_id"],
            "gap_type": "completed experiment" if completed else "new experiment" if needs_experiment else "documentation or unavailable artifact",
            "required_action": action,
            "paid_compute_required": "no" if completed else "yes" if needs_experiment else "no",
            "priority": "high" if study["analysis_role"] == "primary anchor" else "medium" if needs_experiment else "low",
            "status": "complete" if completed else "open" if needs_experiment else "recorded",
        })
    return rows


def write_csv(name: str, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(name)
    path = OUT / name
    columns = fields or list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError(f"inconsistent schema: {name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_baselines(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["comparability"] == "not directly comparable aggregation":
            continue
        grouped[row["study_id"]].append(row)
    verdicts = {row["study_id"]: row for row in build_verdicts()}
    output = []
    for study_id, members in sorted(grouped.items()):
        deltas = [float(row["abs_delta"]) for row in members]
        signed = [float(row["delta"]) for row in members]
        output.append({
            "study_id": study_id,
            "comparison_cells": str(len(members)),
            "mean_absolute_delta": fmt(mean(deltas)),
            "median_absolute_delta": fmt(median(deltas)),
            "maximum_absolute_delta": fmt(max(deltas)),
            "mean_signed_delta": fmt(mean(signed)),
            "baseline_status": verdicts[study_id]["baseline_status"],
            "evidence_grade": verdicts[study_id]["evidence_grade"],
        })
    return output


def latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_tex_tables(inventory: list[dict[str, str]], contracts: list[dict[str, str]]) -> None:
    empirical = [row for row in inventory if row["analysis_role"] not in {"artifact audit", "in-progress baseline", "out of scope"}]
    audited = [row for row in inventory if row["analysis_role"] in {"artifact audit", "in-progress baseline", "out of scope"}]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}p{0.25\textwidth}p{0.27\textwidth}Xc@{}}",
        r"\toprule",
        r"Study & Baseline & Cross-model extension & Grade \\",
        r"\midrule",
    ]
    for row in empirical:
        lines.append(
            f"{latex(row['title'])} & {latex(row['baseline_status'])} & "
            f"{latex(row['extension_status'])} & {latex(row['evidence_grade'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Empirical replication and extension status. Grades summarize evidential comparability, not result quality.}",
        r"\label{tab:empirical-status}",
        r"\end{table*}",
        "",
    ])
    (OUT / "table_empirical_status.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}p{0.28\textwidth}p{0.23\textwidth}X@{}}",
        r"\toprule",
        r"Study & Audit status & Main limitation \\",
        r"\midrule",
    ]
    for row in audited:
        lines.append(f"{latex(row['title'])} & {latex(row['baseline_status'])} & {latex(row['primary_gap'])} \\\\ ")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Studies retained as release, protocol, or scope audits. These rows are not empirical replication failures.}",
        r"\label{tab:artifact-audits}",
        r"\end{table*}",
        "",
    ])
    (OUT / "table_artifact_audits.tex").write_text("\n".join(lines), encoding="utf-8")

    names = {row["study_id"]: row["title"] for row in inventory}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}XXXXX@{}}",
        r"\toprule",
        r"Study & Examples & Responses & Labels & Activation site & Extension estimand \\",
        r"\midrule",
    ]
    for row in contracts:
        lines.append(
            f"{latex(names[row['study_id']])} & {latex(row['examples_across_models'])} & "
            f"{latex(row['responses_across_models'])} & {latex(row['labels_across_models'])} & "
            f"{latex(row['activation_site_selection'])} & {latex(row['extension_estimand'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption{Evaluation contracts for studies with empirical evidence. Changing the model can also change responses, labels, or selection procedures.}",
        r"\label{tab:evaluation-contracts}",
        r"\end{table*}",
        "",
    ])
    (OUT / "table_evaluation_contracts.tex").write_text("\n".join(lines), encoding="utf-8")


def validate(
    inventory: list[dict[str, str]],
    baselines: list[dict[str, str]],
    extensions: list[dict[str, str]],
    contracts: list[dict[str, str]],
) -> dict[str, object]:
    ids = [row["study_id"] for row in inventory]
    if len(ids) != len(set(ids)) or len(ids) != 23:
        raise ValueError("study inventory must contain 23 unique studies")
    if any(not (0 <= float(row["paper_value"]) <= 1 and 0 <= float(row["replication_value"]) <= 1) for row in baselines):
        raise ValueError("baseline metrics must be normalized to fractions")
    if any(row["study_id"] not in ids for row in extensions + contracts):
        raise ValueError("unknown study id")
    return {
        "schema_version": "1.0.0",
        "inventory_rows": len(inventory),
        "baseline_comparison_rows": len(baselines),
        "baseline_studies": len({row["study_id"] for row in baselines}),
        "extension_rows": len(extensions),
        "extension_studies": len({row["study_id"] for row in extensions}),
        "evaluation_contract_rows": len(contracts),
        "roles": dict(sorted(Counter(row["analysis_role"] for row in inventory).items())),
        "baseline_statuses": dict(sorted(Counter(row["baseline_status"] for row in inventory).items())),
        "notes": [
            "Blank fields mean unavailable, not zero.",
            "Metrics in baseline_fidelity.csv are normalized to fractions.",
            "Oracle-selected extension rows remain diagnostics and are not converted into generalization claims.",
            "Legacy studies without row-level predictions are marked explicitly in artifact_coverage.csv.",
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    role_order = {
        "primary anchor": 0,
        "positive control": 1,
        "post-freeze positive control": 2,
        "supporting evidence": 3,
        "baseline failure": 4,
        "artifact audit": 5,
        "in-progress baseline": 6,
        "out of scope": 7,
    }
    inventory = sorted(STUDIES, key=lambda row: (role_order[row["analysis_role"]], row["study_id"]))
    baselines = build_baselines()
    extensions = build_extensions()
    contracts = build_contracts()
    audits = build_audits()
    verdicts = build_verdicts()
    coverage = build_coverage()
    gaps = build_gaps()
    baseline_summary = summarize_baselines(baselines)

    write_csv("study_inventory.csv", inventory)
    write_csv("baseline_fidelity.csv", baselines, BASELINE_FIELDS)
    write_csv("baseline_fidelity_summary.csv", baseline_summary)
    write_csv("cross_model_extensions.csv", extensions)
    write_csv("evaluation_contracts.csv", contracts)
    write_csv("reproducibility_audit.csv", audits)
    write_csv("study_verdicts.csv", verdicts)
    write_csv("artifact_coverage.csv", coverage)
    write_csv("experiment_gaps.csv", gaps)
    write_tex_tables(inventory, contracts)

    audit = validate(inventory, baselines, extensions, contracts)
    (OUT / "schema_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = sorted(path for path in OUT.iterdir() if path.name not in {"manifest.json", "release_manifest.json"})
    manifest = {
        "schema_version": "1.0.0",
        "generated_by": "scripts/build_paper_artifacts.py",
        "outputs": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in outputs
        },
        "source_snapshot": {
            path: sha256(ROOT / path)
            for path in sorted({study["report_path"] for study in STUDIES} | {
                "data/protocols/protocol.json",
                "data/protocols/study_registry.csv",
                "data/probe_replication_queue.csv",
                "data/results/paper-synthesis/evidence_long.csv",
            })
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
