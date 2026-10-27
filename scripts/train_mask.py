"""训练轻量 GRU 掩码网络：训练集归一化、单批过拟合、验证选模与 epoch 级断点。"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from farfield_asr.mask_network import MaskNetwork


def save_checkpoint(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def evaluate(model, examples, device):
    model.eval()
    total_bce, total_mse, elements = 0.0, 0.0, 0
    with torch.inference_mode():
        for features, mask, _ in examples:
            target = torch.from_numpy(mask)[None].to(device)
            logits = model(torch.from_numpy(features)[None].to(device))
            total_bce += float(functional.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
            total_mse += float(functional.mse_loss(logits.sigmoid(), target, reduction="sum"))
            elements += target.numel()
    return total_bce / elements, total_mse / elements


def check_single_batch(examples, mean, std, device, output, training_hash):
    """过拟合一个有语音活动的 64 帧批次；软目标 BCE 的理论下限不一定为零。"""
    model = MaskNetwork().to(device)
    model.feature_mean.copy_(torch.from_numpy(mean).to(device))
    model.feature_std.copy_(torch.from_numpy(std).to(device))
    model.recurrent.dropout = 0.0
    features, mask, metadata = max(examples[:32], key=lambda item: float(np.var(item[1])))
    center = int(np.argmax(mask.mean(axis=1)))
    start = max(0, min(center - 32, len(features) - 64))
    x = torch.from_numpy(features[start:start + 64])[None].to(device)
    y = torch.from_numpy(mask[start:start + 64])[None].to(device)
    if float(y.var()) < 0.01:
        raise ValueError("overfit gate needs a nontrivial speech/noise mask")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    with torch.no_grad():
        initial = float(functional.mse_loss(model(x).sigmoid(), y))
    final = initial
    for step in range(1, 601):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = functional.binary_cross_entropy_with_logits(logits, y)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss during 100-step/overfit gate")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            final = float(functional.mse_loss(model(x).sigmoid(), y))
        if step % 100 == 0:
            print(f"Overfit step {step}: MSE={final:.6f}, BCE={float(loss.detach()):.6f}", flush=True)
        if step >= 100 and final < 0.01 and final < initial * 0.15:
            break
    report = {"example_id": metadata["example_id"], "frames": 64, "steps": step,
              "training_fingerprint": training_hash,
              "initial_mask_mse": initial, "final_mask_mse": final,
              "label_variance": float(y.var()), "dropout": 0,
              "passed_100_steps": step >= 100,
              "passed_overfit": final < 0.01 and final < initial * 0.15,
              "used_for_final_model": False}
    (output / "overfit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed_overfit"]:
        raise ValueError("single-batch overfit gate did not meet its documented MSE threshold")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=root / "checkpoints/mask_gru")
    args = parser.parse_args()
    if args.epochs < 6 or args.batch_size < 1 or args.chunk_frames < 64 or args.threads < 1:
        raise ValueError("invalid training settings")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    seed = 20261027
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/masks_v1/manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    examples = {"train": [], "dev": []}
    for record in records:
        path = Path(record["feature_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["feature_sha256"]:
            raise ValueError("training feature hash mismatch")
        with np.load(path, allow_pickle=False) as item:
            features, mask = item["features"].copy(), item["mask"].copy()
        if features.shape != mask.shape or features.shape[1] != 257 or len(features) < args.chunk_frames:
            raise ValueError("feature layout or crop length mismatch")
        examples[record["split"]].append((features, mask, record))
    if not examples["train"] or not examples["dev"]:
        raise ValueError("missing train/dev feature data")
    train_speakers = {item[2]["speaker_id"] for item in examples["train"]}
    dev_speakers = {item[2]["speaker_id"] for item in examples["dev"]}
    test_speakers = {json.loads(line)["speaker_id"] for line in
                     (root / "data/manifests/speech_test.jsonl").read_text(encoding="utf-8").splitlines()}
    if train_speakers & dev_speakers or (train_speakers | dev_speakers) & test_speakers:
        raise ValueError("speaker leakage detected before training")
    frames = sum(len(item[0]) for item in examples["train"])
    total = sum((item[0].sum(axis=0, dtype=np.float64) for item in examples["train"]), np.zeros(257))
    squared = sum(((item[0].astype(np.float64)**2).sum(axis=0) for item in examples["train"]), np.zeros(257))
    mean64 = total / frames
    mean = mean64.astype(np.float32)
    std = np.sqrt(np.maximum(squared / frames - mean64**2, 1e-4)).astype(np.float32)
    source = Path(__file__).read_bytes() + (root / "src/farfield_asr/mask_network.py").read_bytes()
    training_hash = hashlib.sha256(source + manifest_path.read_bytes()
                                    + json.dumps([args.epochs, args.batch_size, args.chunk_frames]).encode()).hexdigest()
    summary_path = args.output / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["training_fingerprint"] != training_hash:
            raise ValueError("finished training has a different configuration; use a new output directory")
        if hashlib.sha256((args.output / "best.pt").read_bytes()).hexdigest() != summary["best_checkpoint_sha256"]:
            raise ValueError("finished checkpoint changed")
        print("Verified completed training and checkpoint", flush=True)
        return
    if not (args.output / "overfit.json").exists():
        check_single_batch(examples["train"], mean, std, args.device, args.output, training_hash)
    else:
        gate = json.loads((args.output / "overfit.json").read_text())
        if not gate["passed_overfit"] or gate["training_fingerprint"] != training_hash:
            raise ValueError("previous overfit gate failed or used different training inputs")
    torch.manual_seed(seed)
    model = MaskNetwork().to(args.device)
    model.feature_mean.copy_(torch.from_numpy(mean).to(args.device))
    model.feature_std.copy_(torch.from_numpy(std).to(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    first_epoch, best_loss, best_epoch, stagnant, history = 1, float("inf"), 0, 0, []
    last_path = args.output / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=args.device, weights_only=True)
        if state["training_fingerprint"] != training_hash:
            raise ValueError("resume fingerprint differs")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        first_epoch, best_loss = state["epoch"] + 1, state["best_loss"]
        best_epoch, stagnant, history = state["best_epoch"], state["stagnant"], state["history"]
    metadata = {"seed": seed, "training_fingerprint": training_hash,
                "sample_rate": 16000, "n_fft": 512, "hop_length": 128,
                "mask_target": "reverberant-target power IRM at reference microphone",
                "train_examples": len(examples["train"]), "dev_examples": len(examples["dev"]),
                "train_speaker_ids": sorted(train_speakers), "dev_speaker_ids": sorted(dev_speakers),
                "test_speaker_ids_excluded": sorted(test_speakers),
                "train_noise_ids": sorted({item[2]["noise_id"] for item in examples["train"]}),
                "dev_noise_ids": sorted({item[2]["noise_id"] for item in examples["dev"]}),
                "normalization": "per-frequency moments fitted on train frames only",
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "torch_version": str(torch.__version__), "device": args.device}
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        total_loss, batch_elements = 0.0, 0
        order = rng.permutation(len(examples["train"]))
        for position in range(0, len(order), args.batch_size):
            inputs, targets = [], []
            for index in order[position:position + args.batch_size]:
                features, mask, _ = examples["train"][index]
                start = int(rng.integers(0, len(features) - args.chunk_frames + 1))
                inputs.append(features[start:start + args.chunk_frames])
                targets.append(mask[start:start + args.chunk_frames])
            x = torch.from_numpy(np.stack(inputs)).to(args.device)
            y = torch.from_numpy(np.stack(targets)).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * y.numel()
            batch_elements += y.numel()
        val_loss, val_mse = evaluate(model, examples["dev"], args.device)
        row = {"epoch": epoch, "train_bce": total_loss / batch_elements,
               "dev_bce": val_loss, "dev_mask_mse": val_mse}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stagnant = val_loss, epoch, 0
            save_checkpoint(args.output / "best.pt", {"model_state": model.state_dict(),
                            "model_config": {"n_bins": 257, "hidden_size": 64},
                            "metadata": {**metadata, "best_epoch": epoch, "dev_bce": val_loss,
                                         "dev_mask_mse": val_mse}})
        else:
            stagnant += 1
        save_checkpoint(last_path, {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "numpy_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                        "training_fingerprint": training_hash, "epoch": epoch, "best_loss": best_loss,
                        "best_epoch": best_epoch, "stagnant": stagnant, "history": history})
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if epoch >= 6 and stagnant >= 4:
            break
    summary = {**metadata, "epochs_completed": history[-1]["epoch"], "best_epoch": best_epoch,
               "best_dev_bce": best_loss, "status": "completed",
               "best_checkpoint_sha256": hashlib.sha256((args.output / "best.pt").read_bytes()).hexdigest()}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
