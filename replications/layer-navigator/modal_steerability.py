"""Dense per-layer steerability (LayerNavigator S_l) + probing AUROC on MUP high-stakes pairs.

For each model family, runs ONE forward pass over a balanced subset of the 6 MUP
test datasets, capturing the last-token hidden state at EVERY layer. From those
pooled activations it computes, per layer:
  * S_l = D_l + C_l  (LayerNavigator steerability: Fisher discriminability + cosine
    consistency of the high-minus-low steering vector), exactly per get_score.py.
  * probing AUROC   (5-fold stratified CV logistic probe on standardized features).

This is the zero-extra-GPU experiment the thesis hinges on: S_l is computed from the
same contrastive activations a detection probe already needs, so a per-model S_l
curve costs one small forward pass, not a probe-training fleet.

    modal run replications/layer-navigator/modal_steerability.py --model Qwen/Qwen3-32B
    modal run replications/layer-navigator/modal_steerability.py --model meta-llama/Llama-3.3-70B-Instruct --gpu B200:2 --per-class 40
"""
import os, json, subprocess
from pathlib import Path
import modal

_here = Path(__file__).resolve()
REPO = (
    _here.parents[2] / "external" / "models-under-pressure"
    if len(_here.parents) >= 3 else Path("/repo")
)
TEST_DIR = REPO / "data" / "evals" / "test"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "pip install torch 'transformers>=4.56,<4.57' accelerate safetensors "
        "huggingface-hub tqdm scikit-learn numpy boto3"
    )
    # Ship only the 6 balanced test jsonl files (small); weights come from HF/volume.
    .add_local_dir(str(TEST_DIR), "/data/test", copy=True)
)
secret = modal.Secret.from_name("mup-env")           # HF_TOKEN + R2 keys
volume = modal.Volume.from_name("mup-activations", create_if_missing=True)
CACHE = "/repo/data/activations"                      # reuse cached HF weights in _hf
app = modal.App("layernav-steerability")

DATASETS = [
    "anthropic_hh_balanced", "toolace_balanced", "mt_balanced",
    "mts_balanced", "mental_health_balanced", "aya_redteaming_balanced",
]

def _load_subset(per_class, test_dir="/data/test"):
    """Balanced subset: per_class high + per_class low from EACH dataset."""
    import random
    random.seed(0)
    conv, y, dsname = [], [], []
    for ds in DATASETS:
        path = Path(test_dir) / f"{ds}.jsonl"
        hi, lo = [], []
        for line in path.read_text().splitlines():
            r = json.loads(line)
            (hi if r["labels"] == "high-stakes" else lo).append(r["inputs"])
        random.shuffle(hi); random.shuffle(lo)
        for c in hi[:per_class]:
            conv.append(c); y.append(1); dsname.append(ds)
        for c in lo[:per_class]:
            conv.append(c); y.append(0); dsname.append(ds)
    return conv, y, dsname

def _run(model, gpu, per_class, max_len):
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["HF_HOME"] = f"{CACHE}/_hf"
    Path(f"{CACHE}/_hf").mkdir(parents=True, exist_ok=True)

    conv, y, dsname = _load_subset(per_class)
    print(f"subset: {len(y)} examples ({sum(y)} high / {len(y)-sum(y)} low) across {len(DATASETS)} datasets")

    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    model_obj.eval()
    n_layers = model_obj.config.num_hidden_layers
    print(f"{model}: {n_layers} layers")

    # Encode with chat template, capture last-token hidden state at every layer.
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conv]
    B = 4 if "70B" in model or "70b" in model else 8
    pooled = None  # (N, n_layers+1, H)
    import math
    with torch.no_grad():
        for s in range(0, len(texts), B):
            chunk = texts[s:s+B]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(model_obj.device)
            out = model_obj(**enc)
            hs = out.hidden_states  # tuple(n_layers+1) of (b, T, H); left-padded -> [-1] is last real token
            lt = torch.stack([h[:, -1, :].float().cpu() for h in hs], dim=1)  # (b, L+1, H)
            pooled = lt if pooled is None else torch.cat([pooled, lt], dim=0)
            if s % 40 == 0:
                print(f"  {s+len(chunk)}/{len(texts)}")
    y = np.array(y)
    X = pooled.numpy()  # (N, L+1, H)
    print("pooled activations:", X.shape)

    # Per-layer S_l (LayerNavigator) + probing AUROC.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for L in range(X.shape[1]):
        A = X[:, L, :]                       # (N, H)
        # ---- steerability (z-score features, per get_score.py) ----
        Az = (A - A.mean(0)) / (A.std(0) + 1e-8)
        mu_hi, mu_lo = Az[y==1].mean(0), Az[y==0].mean(0)
        v = mu_hi - mu_lo
        v = v / (np.linalg.norm(v) + 1e-8)
        # C_l: mean cosine of per-example (hi-centroid vs lo-centroid) diffs with v
        # LayerNavigator uses paired diffs; here use each class-vs-opposite-centroid.
        diff = np.concatenate([Az[y==1] - mu_lo, Az[y==0] - mu_hi], 0)
        diff = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8)
        # sign so that high-stakes side is positive
        signs = np.concatenate([np.ones((y==1).sum()), -np.ones((y==0).sum())])
        c_score = float((diff @ v * signs).mean())
        # D_l: Fisher ratio along v
        mean_total = Az.mean(0)
        Sw = np.zeros((A.shape[1], A.shape[1])); Sb = np.zeros_like(Sw)
        for c in (0, 1):
            cls = Az[y==c]; mc = cls.mean(0)
            d = cls - mc; Sw += d.T @ d
            md = (mc - mean_total).reshape(-1,1); Sb += cls.shape[0]*(md @ md.T)
        St = Sw + Sb; vv = v.reshape(-1,1)
        num = float(vv.T @ Sb @ vv); den = float(vv.T @ St @ vv)
        d_score = num/den if den != 0 else float("inf")
        s_score = d_score + c_score
        # ---- probing AUROC (5-fold CV) ----
        aucs = []
        for tr, te in skf.split(A, y):
            sc = StandardScaler().fit(A[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.transform(A[tr]), y[tr])
            p = clf.predict_proba(sc.transform(A[te]))[:,1]
            aucs.append(roc_auc_score(y[te], p))
        rows.append({"layer": L, "depth": L/n_layers, "s_score": s_score,
                     "d_score": d_score, "c_score": c_score,
                     "probe_auroc_cv": float(np.mean(aucs)), "probe_auroc_std": float(np.std(aucs))})
        print(f"L{L:>2} depth={L/n_layers:.2f}  S={s_score:.3f} (D={d_score:.3f} C={c_score:.3f})  AUROC={np.mean(aucs):.4f}")

    tag = model.split("/")[-1]
    out = {"model": model, "n_layers": n_layers, "per_class": per_class,
           "n_examples": int(len(y)), "datasets": DATASETS, "rows": rows}
    outdir = Path(f"{CACHE}/steerability"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    # also stash pooled activations (float16) for offline reuse
    np.savez_compressed(outdir / f"{tag}_pooled.npz",
                        X=X.astype(np.float16), y=y, dsname=np.array(dsname))
    volume.commit()
    # upload JSON to R2
    try:
        import boto3
        c = boto3.client("s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
        c.upload_file(str(outdir / f"{tag}.json"), os.environ["R2_ACTIVATIONS_BUCKET"], f"steerability/{tag}.json")
        print(f"uploaded steerability/{tag}.json to R2")
    except Exception as e:
        print("R2 upload skipped:", e)
    print(f"DONE {model}: {n_layers} layers, {len(y)} examples")

@app.function(image=image, gpu="A100", secrets=[secret],
              volumes={CACHE: volume}, timeout=4*60*60)
def run_a100(model: str, per_class: int, max_len: int):
    _run(model, "A100", per_class, max_len)

@app.function(image=image, gpu="B200:2", secrets=[secret],
              volumes={CACHE: volume}, timeout=4*60*60)
def run_b200x2(model: str, per_class: int, max_len: int):
    _run(model, "B200:2", per_class, max_len)

@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-32B", gpu: str = "A100",
         per_class: int = 40, max_len: int = 1024):
    if gpu.startswith("B200"):
        run_b200x2.remote(model=model, per_class=per_class, max_len=max_len)
    else:
        run_a100.remote(model=model, per_class=per_class, max_len=max_len)
