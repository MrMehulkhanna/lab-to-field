"""One call = one complete experiment: build data, train, test in the lab and in the field."""
import json
from functools import lru_cache

import numpy as np

from .config import N_CLASSES, RESULTS_DIR
from .data import (BackgroundReplacer, build_transforms, cap_per_class, drop_shared_leaves, load_plantdoc,
                   load_plantvillage, make_loader, seed_everything, split_by_group)
from .metrics import bootstrap_ci, summarize
from .models import build_model, count_params
from .train import TrainConfig, fit, predict, save_result

PER_CLASS_CAP = 500  # PlantVillage images per class used for training (speed vs. data)


@lru_cache
def splits(cap: int | None = PER_CLASS_CAP, seed: int = 0) -> dict:
    """All data splits, built once and reused.

    lab_train / lab_val : PlantVillage official train split minus leaves also in test, divided by leaf
    lab_test            : PlantVillage official test split (different leaves)
    field_train/val     : PlantDoc train split, divided 85/15 (used only by the "field" track)
    field_test          : PlantDoc test split, the final exam, never trained or tuned on
    """
    pv_test = load_plantvillage("test")
    pv_train, pv_val = split_by_group(drop_shared_leaves(load_plantvillage("train"), pv_test), 0.1, seed)
    pd_train, pd_val = split_by_group(load_plantdoc("train"), 0.15, seed)
    return {
        "lab_train": cap_per_class(pv_train, cap, seed),
        "lab_val": cap_per_class(pv_val, 100, seed),
        "lab_test": pv_test,
        "field_train": pd_train,
        "field_val": pd_val,
        "field_test": load_plantdoc("test"),
    }


def run(cfg: TrainConfig, track: str = "lab", save: bool = True, force: bool = False) -> dict:
    """Train one model and score it on the lab test, field validation and field test sets.

    track="lab"  : train and select epochs on PlantVillage only ("how far does lab data get us?")
    track="field": also train on PlantDoc train photos and select epochs on PlantDoc val

    Every setting we *choose* between experiments (augmentation, optimizer, loss) is chosen on
    field_val. field_test is only reported, never used to choose anything.
    If results/<name>.json already exists, the saved result is returned instead of retraining
    (pass force=True to retrain).
    """
    cached = RESULTS_DIR / f"{cfg.name}.json"
    if save and cached.exists() and not force:
        result = json.loads(cached.read_text())
        print(f"[{cfg.name}] loaded saved result (delete {cached.name} or pass force=True to retrain): "
              f"lab acc {result['lab_test']['accuracy']:.3f} | field val {result['field_val']['accuracy']:.3f} | "
              f"field test {result['field_test']['accuracy']:.3f}")
        return result
    seed_everything(cfg.seed)
    s = splits()
    train_df, val_df = s["lab_train"], s["lab_val"]
    if track == "field":
        import pandas as pd
        # Repeat the ~2,000 field photos so they are not drowned out by lab photos.
        train_df = pd.concat([train_df] + [s["field_train"]] * 3, ignore_index=True)
        val_df = s["field_val"]

    bg = None
    if cfg.augment.startswith("bg_"):
        bg = BackgroundReplacer("field" if cfg.augment == "bg_field" else "synthetic",
                                field_paths=s["field_train"]["path"].tolist())
    train_loader = make_loader(train_df, build_transforms(cfg.augment, cfg.img_size), cfg.batch_size,
                               shuffle=True, bg=bg)
    eval_tf = build_transforms("none", cfg.img_size, train=False)
    val_loader = make_loader(val_df, eval_tf, 128)

    model = build_model(cfg.model, N_CLASSES, freeze_backbone=cfg.freeze_backbone, act=cfg.act,
                        **cfg.extra.get("model_kwargs", {}))
    total, trainable = count_params(model)
    result = fit(model, cfg, train_loader, val_loader)
    result.update({"track": track, "params_total": total, "params_trainable": trainable,
                   "n_train": len(train_df)})

    for test_name in ("lab_test", "field_val", "field_test"):
        logits, probs, labels = predict(model, make_loader(s[test_name], eval_tf, 128))
        m = summarize(probs, labels, N_CLASSES)
        m["accuracy_ci95"] = bootstrap_ci(probs, labels)
        result[test_name] = m
        if save:
            (RESULTS_DIR / "preds").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(RESULTS_DIR / "preds" / f"{cfg.name}__{test_name}.npz",
                                logits=logits, labels=labels)
    lab, val, field = (result[k]["accuracy"] for k in ("lab_test", "field_val", "field_test"))
    print(f"[{cfg.name}] lab acc {lab:.3f} | field val {val:.3f} | field test {field:.3f} | "
          f"gap {100 * (lab - field):.1f} pts")
    if save:
        save_result(cfg.name, result)
    return result
