"""성부 배정 모델 학습. 실습실 PC가 언제든 꺼질 수 있으므로 중단·재개를 전제로 짠다.

사전학습(층 A 합성):
    uv run python -m training.train --train data/synth/train --valid data/synth/valid \
        --out checkpoints/pretrain --resume --backup-dir E:/backup/ckpt
파인튜닝(층 B 실제, 낮은 학습률, 앞단 고정):
    uv run python -m training.train --train data/real/train --valid data/real/valid \
        --out checkpoints/finetune --init checkpoints/pretrain/best.pt --lr 1e-4 --freeze-front --resume

- ``--ckpt-every`` 스텝마다 last.pt를 원자적으로 저장(모델·옵티마이저·스케줄러·스텝·RNG 상태).
- ``--resume``이면 out/last.pt에서 이어서 한다. valid 지표가 좋아질 때마다 best.pt를 저장한다.
- ``--backup-dir``을 주면 저장할 때마다 실습실 PC 밖(외장 디스크·NAS 등)에도 복사한다.
- 가중치는 train으로만 갱신한다. valid는 조기 종료와 체크포인트 선택에만 쓴다. test는 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dataset import IGNORE, VoiceDataset
from training.model import VoiceAssignNet, count_parameters

log = logging.getLogger(__name__)


def loss_fn(pitch, active, cls, act_tgt, slot_mask):
    """음높이 CE(유성 프레임만) + 유무성 BCE(쓰는 슬롯만)."""
    ce = nn.functional.cross_entropy(
        pitch.reshape(-1, pitch.shape[-1]), cls.reshape(-1), ignore_index=IGNORE
    )
    if torch.isnan(ce):  # 유성 프레임이 하나도 없는 배치
        ce = pitch.sum() * 0
    bce = nn.functional.binary_cross_entropy_with_logits(active, act_tgt, reduction="none")
    m = slot_mask[..., None].expand_as(bce)
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    return ce + bce, ce.detach(), bce.detach()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = voiced = tp = fp = fn = 0
    for sal, ranges, cls, act_tgt, slot_mask in loader:
        sal, ranges, cls, act_tgt = (
            sal.to(device),
            ranges.to(device),
            cls.to(device),
            act_tgt.to(device),
        )
        pitch, active = model(sal, ranges)
        pred_act = active > 0
        tgt_act = act_tgt > 0.5
        used = slot_mask.to(device)[..., None].bool().expand_as(pred_act)
        tp += (pred_act & tgt_act & used).sum().item()
        fp += (pred_act & ~tgt_act & used).sum().item()
        fn += (~pred_act & tgt_act & used).sum().item()
        ok = (pitch.argmax(-1) == cls) & tgt_act & pred_act
        correct += ok.sum().item()
        voiced += tgt_act.sum().item()
    model.train()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = correct / max(voiced, 1)
    return {"pitch_accuracy": acc, "voicing_f1": f1, "score": (acc + f1) / 2}


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_atomic(obj: dict, path: Path, backup_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
    if backup_dir is not None:
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
        except OSError as e:
            log.warning("백업 실패(%s): %s", backup_dir, e)


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="성부 배정 모델 학습")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="out/last.pt가 있으면 이어서 학습")
    ap.add_argument("--init", type=Path, help="파인튜닝 시작 가중치(best.pt)")
    ap.add_argument("--freeze-front", action="store_true", help="앞단 합성곱 층 고정(파인튜닝)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--valid-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=10, help="valid가 좋아지지 않은 평가 횟수 한도")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backup-dir", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = VoiceAssignNet(args.channels).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        log.info("초기 가중치: %s", args.init)
    if args.freeze_front:
        for p in model.front_parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_steps, eta_min=args.lr * 0.05
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best, bad_evals, history = 0, -1.0, 0, []
    last = args.out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, best, bad_evals, history = ck["step"], ck["best"], ck["bad_evals"], ck["history"]
        _set_rng_state(ck["rng"])
        log.info("재개: %d스텝부터 (best %.4f)", step, best)
    log.info("파라미터 %d개, 장치 %s", count_parameters(model), device)

    train_ds = VoiceDataset(args.train, train=True, seed=args.seed + step)
    valid_ds = VoiceDataset(args.valid, train=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, num_workers=args.workers)

    def checkpoint(path: Path) -> None:
        save_atomic(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best": best,
                "bad_evals": bad_evals,
                "history": history,
                "rng": _rng_state(),
                "config": vars(args) | {"out": str(args.out)},
            },
            path,
            args.backup_dir,
        )

    model.train()
    t0 = time.time()
    done = False
    while not done:
        for sal, ranges, cls, act_tgt, slot_mask in train_dl:
            sal, ranges = sal.to(device), ranges.to(device)
            cls, act_tgt, slot_mask = cls.to(device), act_tgt.to(device), slot_mask.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                pitch, active = model(sal, ranges)
                loss, ce, bce = loss_fn(pitch.float(), active.float(), cls, act_tgt, slot_mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0:
                log.info(
                    "step %d loss %.4f (ce %.4f bce %.4f) %.1fs",
                    step,
                    loss.item(),
                    ce,
                    bce,
                    time.time() - t0,
                )
            if step % args.valid_every == 0 or step >= args.max_steps:
                m = evaluate(model, valid_dl, device)
                history.append({"step": step, **m})
                log.info("valid step %d: %s", step, json.dumps(m))
                if m["score"] > best:
                    best, bad_evals = m["score"], 0
                    checkpoint(args.out / "best.pt")
                else:
                    bad_evals += 1
                if bad_evals >= args.patience:
                    log.info("조기 종료: valid가 %d번 연속 좋아지지 않았습니다.", bad_evals)
                    done = True
            if step % args.ckpt_every == 0 or done or step >= args.max_steps:
                checkpoint(last)
            if done or step >= args.max_steps:
                done = True
                break
    (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return {"step": step, "best": best, "history": history}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = train(build_args(argv))
    print(json.dumps({"step": res["step"], "best": res["best"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
