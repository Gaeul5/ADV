import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent / "ELM"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.models import EEG_ResNet  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 사전학습된 EEG 인코더 로딩
# ---------------------------------------------------------------------------
def load_pretrained_eeg_encoder(config_path: str, weights_path: str, device: torch.device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mp = config["model"]
    encoder = EEG_ResNet(
        in_channels=mp["in_channels"],
        conv1_params=mp["encoder_conv1_params"],
        n_blocks=mp["encoder_blocks"],
        res_params=mp["encoder_res_params"],
        res_pool_size=mp["encoder_pool_size"],
        dropout_p=mp["encoder_dropout_p"],
        res_dropout_p=mp["res_dropout_p"],
        proj_size=mp["ELM"]["eeg_proj_size"],
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if config["training"].get("DDP", False):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing[:10]}...")
    if unexpected:
        raise RuntimeError(f"Checkpoint unexpected keys: {unexpected[:10]}...")

    encoder.eval()

    n_params = sum(p.numel() for p in encoder.parameters())
    return encoder, config, n_params


# ---------------------------------------------------------------------------
# 2. 텍스트 인코더 로딩
# ---------------------------------------------------------------------------
class HuggingFaceTextEncoder(torch.nn.Module):
    def __init__(self, model, tokenizer, device: torch.device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # ELM uses the CLS token representation for the text side
        emb = outputs.last_hidden_state[:, 0, :]
        return emb

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return super().__call__(texts)


def load_text_encoder(config: dict, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    lm_url = config["model"]["ELM"]["LM_pretrained_url"]
    custom_cache = config["model"]["ELM"].get("custom_cache", None)

    model = AutoModel.from_pretrained(
        lm_url,
        trust_remote_code=True,
        revision="main",
        cache_dir=custom_cache,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        lm_url,
        trust_remote_code=True,
        revision="main",
        cache_dir=custom_cache,
    )
    model.eval()
    return HuggingFaceTextEncoder(model, tokenizer, device)


def embed_candidate_texts(text_encoder, candidates: list[str], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        embs = text_encoder(candidates)
    return F.normalize(embs.float(), dim=-1)


# ---------------------------------------------------------------------------
# 3. Zero-shot 폐쇄형 분류 (K=2: normal vs abnormal)
# ---------------------------------------------------------------------------
def zero_shot_predict(eeg_embeddings: np.ndarray, text_embeddings: torch.Tensor, temperature: float = 0.07):
    eeg = F.normalize(torch.from_numpy(eeg_embeddings).float(), dim=-1)
    sims = eeg @ text_embeddings.T
    probs = F.softmax(sims / temperature, dim=-1)
    top1 = probs.argmax(dim=-1)
    margin = probs.max(dim=-1).values - probs.kthvalue(probs.shape[-1] - 1, dim=-1).values
    return top1.numpy(), probs.numpy(), margin.numpy()


# ---------------------------------------------------------------------------
# 4. 벤치마크 표 행(row) 생성
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRow:
    model: str
    domain: str
    K: int
    n_test: int
    top1_acc: float
    confidence_mean: float
    confidence_std: float
    n_params: int
    notes: str

    def to_markdown_row(self) -> str:
        return (
            f"| {self.model} | {self.domain} | {self.K} | {self.n_test} "
            f"| {self.top1_acc:.3f} | {self.confidence_mean:.3f}±{self.confidence_std:.3f} "
            f"| {self.n_params/1e6:.1f}M | {self.notes} |"
        )


def compute_metrics(top1_preds: np.ndarray, labels: np.ndarray, margin: np.ndarray,
                     model_name: str, n_params: int) -> BenchmarkRow:
    labels = np.asarray(labels).reshape(-1)
    if labels.dtype.kind in {"U", "S"}:
        labels = np.array([0 if str(x).lower().startswith("n") else 1 for x in labels], dtype=int)
    else:
        labels = labels.astype(int)
    if labels.max() > 1:
        labels = (labels != 0).astype(int)

    acc = float((top1_preds == labels).mean())
    return BenchmarkRow(
        model=model_name,
        domain="TUAB (normal/abnormal, native domain)",
        K=2,
        n_test=len(labels),
        top1_acc=acc,
        confidence_mean=float(margin.mean()),
        confidence_std=float(margin.std()),
        n_params=n_params,
        notes="zero-shot, closed-set K=2",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="run_DL.py GEN_EMB로 만든 embedding_dataset.h5 경로")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="stage_a_elm_result.json")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--candidate-normal", default="This EEG recording is normal.")
    ap.add_argument("--candidate-abnormal", default="This EEG recording is abnormal.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, config, n_params = load_pretrained_eeg_encoder(args.config, args.checkpoint, device)
    _ = encoder  # keep the encoder loaded for parity with the original scaffold

    if not Path(args.embeddings).exists():
        raise FileNotFoundError(
            f"Embedding file not found: {args.embeddings}. "
            "먼저 run_DL.py GEN_EMB로 embedding_dataset.h5를 생성한 뒤 다시 실행하세요."
        )

    with h5py.File(args.embeddings, "r") as f:
        if "embeddings" in f:
            eeg_embeddings = f["embeddings"][:]
        elif "features" in f:
            eeg_embeddings = f["features"][:]
        else:
            raise KeyError("Embedding file must contain either 'embeddings' or 'features'.")

        if "pathology" in f:
            labels = f["pathology"][:]
        elif "pat" in f:
            labels = f["pat"][:]
        else:
            raise KeyError("Embedding file must contain a pathology label field such as 'pathology' or 'pat'.")

    text_encoder = load_text_encoder(config, device)
    candidates = [args.candidate_normal, args.candidate_abnormal]
    text_embeddings = embed_candidate_texts(text_encoder, candidates, device)

    top1, probs, margin = zero_shot_predict(eeg_embeddings, text_embeddings, temperature=args.temperature)
    model_name = f"ELM ({Path(args.config).parent.name}, e,l)"
    row = compute_metrics(top1, labels, margin, model_name=model_name, n_params=n_params)

    with open(args.out, "w") as f:
        json.dump(asdict(row), f, ensure_ascii=False, indent=2)

    print(row.to_markdown_row())


if __name__ == "__main__":
    main()
