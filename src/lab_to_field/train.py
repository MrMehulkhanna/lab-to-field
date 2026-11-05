"""Training and prediction loops shared by all notebooks."""
import json
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .config import RESULTS_DIR, CHECKPOINT_DIR

# The first scheduler step can run before an optimizer step when GradScaler skips an fp16
# overflow; that is expected and harmless.
warnings.filterwarnings("ignore", message=re.escape("Detected call of `lr_scheduler.step()`"))


@dataclass
class TrainConfig:
    name: str                       # run name, used for result and checkpoint files
    model: str = "mobilenetv3_large_100"
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"        # "adamw" | "sgd"
    loss: str = "ce"                # "ce" | "label_smoothing" | "focal"
    batch_size: int = 64
    img_size: int = 224
    augment: str = "basic"          # see data.build_transforms
    freeze_backbone: bool = False
    act: str = "relu"               # only used by simple_cnn
    seed: int = 0
    extra: dict = field(default_factory=dict)


class FocalLoss(nn.Module):
    """Cross-entropy that down-weights easy examples: loss * (1 - p_correct)^gamma.

    Hard, misclassified images then dominate the gradient.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, 1).gather(1, target[:, None]).squeeze(1)
        return (-(1 - logp.exp()) ** self.gamma * logp).mean()


def make_loss(name: str) -> nn.Module:
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if name == "focal":
        return FocalLoss()
    raise ValueError(name)


def make_optimizer(name: str, params, lr: float, wd: float):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    raise ValueError(name)


@torch.no_grad()
def predict(model: nn.Module, loader, device="cuda") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logits, softmax probabilities, labels) for every image in the loader."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            all_logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    return logits.numpy(), logits.softmax(1).numpy(), torch.cat(all_labels).numpy()


def fit(model: nn.Module, cfg: TrainConfig, train_loader, val_loader, device="cuda") -> dict:
    """Train with mixed precision and a cosine learning-rate schedule with warmup.

    Keeps the weights from the epoch with the best *validation* accuracy. The field test
    sets are never looked at here, so they stay an honest final exam.
    """
    torch.manual_seed(cfg.seed)
    model = model.to(device).to(memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(cfg.optimizer, params, cfg.lr, cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=warmup / total_steps, anneal_strategy="cos")
    loss_fn = make_loss(cfg.loss)
    scaler = torch.amp.GradScaler()
    frozen = getattr(model, "frozen_backbone", False)

    history, best_acc, best_state = [], -1.0, None
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        if frozen:  # linear probe: freeze BatchNorm statistics as well as weights
            model.eval()
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"{cfg.name} epoch {epoch + 1}/{cfg.epochs}", leave=False,
                   disable=not sys.stderr.isatty())  # no progress bars inside saved notebooks
        for x, y in bar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
        _, probs, labels = predict(model, val_loader, device)
        val_acc = float((probs.argmax(1) == labels).mean())
        history.append({"epoch": epoch + 1, "train_loss": running / seen, "val_acc": val_acc})
        print(f"[{cfg.name}] epoch {epoch + 1}: train loss {running / seen:.4f}  val acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CHECKPOINT_DIR / f"{cfg.name}.pt")
    return {"config": asdict(cfg), "history": history, "best_val_acc": best_acc,
            "train_minutes": (time.time() - t0) / 60}


def save_result(name: str, result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def load_results() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))}
