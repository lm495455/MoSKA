"""Training entrypoint for PTH-Net backdoor evaluation.

This script was refactored for open-source usage:
- removes hard-coded local paths
- replaces fixed experiment grids with CLI arguments
- centralizes logging/checkpointing per fold
- keeps the original training/validation objectives and metrics
"""

import argparse
import datetime
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from dataloader.dataset_PTH_Net import test_data_loader, train_data_loader
from models.ETH_Net import eth_net
import combine_test as test

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_int_list(value: str) -> List[int]:
    value = str(value).strip()
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_tuple_list(value: str) -> Tuple[int, ...]:
    return tuple(parse_int_list(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate PTH-Net")

    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="DataLoader workers")
    parser.add_argument("--epochs", default=20, type=int, metavar="N", help="Total number of epochs")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="Resume start epoch")
    parser.add_argument("--bz", "--batch-size", dest="batch_size", default=32, type=int, metavar="N")
    parser.add_argument("--eval-batch-size", default=32, type=int, metavar="N")
    parser.add_argument("--lr", "--learning-rate", dest="lr", default=0.005, type=float, metavar="LR")
    parser.add_argument("--gamma", "--scheduler-gamma", default=0.1, type=float)
    parser.add_argument("--mil", "--milestones", dest="milestones", default="5,10,15", type=str)
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", default=1e-4, type=float, metavar="W")
    parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="Print frequency")
    parser.add_argument("--resume", default=None, type=str, metavar="PATH", help="Path to checkpoint")

    parser.add_argument(
        "--dataset",
        type=str,
        default="FERv39k",
        choices=["DFEW", "FERv39k", "MAFW", "CREMA-D", "eNTERFACE05", "RAVDESS", "MELD", "CASME2"],
    )
    parser.add_argument("--data-mode", type=str, default="norm", choices=["norm", "rv", "flow"])
    parser.add_argument("--label-type", type=str, default="single", choices=["single", "compound"])
    parser.add_argument("--num-class", type=int, default=7, choices=[5, 6, 7, 8, 11, 43])
    parser.add_argument("--is-face", type=str2bool, default=True)
    parser.add_argument("--folds", type=str, default="1", help="Comma-separated folds, e.g. 1,2,3")

    parser.add_argument(
        "--attack",
        "--mes",
        dest="attack",
        default="Benign",
        type=str,
        help="Attack mode name. If not Benign, the trailing token must be poison ratio, e.g. Poison_XYZ_0.1",
    )
    parser.add_argument("--poison-ratio", default=None, type=float, help="Override poison ratio")

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data-root", type=str, default="", help="Optional feature root directory")
    parser.add_argument("--output-root", type=str, default="./outputs/pth_net")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--save-begin-checkpoint", type=str2bool, default=False)

    # ETH-Net parameters
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--k", type=str, default="1,3,5")
    parser.add_argument("--thr-size", type=str, default="3,1,3,3")
    parser.add_argument("--arch", type=str, default="2,2,1,1")
    parser.add_argument("--n-in", type=int, default=1408)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--downsample-type", type=str, default="max")
    parser.add_argument("--scale-factor", type=int, default=2)
    parser.add_argument("--with-ln", type=str2bool, default=True)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--path-pdrop", type=float, default=0.1)
    parser.add_argument("--use-pos", type=str2bool, default=False)

    return parser


def parse_attack(attack_name: str, poison_ratio_override: float = None) -> Tuple[str, float]:
    if attack_name == "Benign":
        poisoned_mode = "Benign"
        ratio = 0.0
    else:
        parts = attack_name.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(
                "Attack format must end with poison ratio, for example: Poison_hello_kitty_avg_face_flow_0.1"
            )
        poisoned_mode = parts[0]
        ratio = float(parts[-1])
    if poison_ratio_override is not None:
        ratio = poison_ratio_override
    return poisoned_mode, ratio


def resolve_folds(folds_text: str) -> List[int]:
    folds = parse_int_list(folds_text)
    if not folds:
        raise ValueError("`--folds` cannot be empty")
    return folds


def build_experiment_name(args: argparse.Namespace, poison_ratio: float) -> str:
    mode_tag = "face" if args.is_face else "ori"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.dataset}-{mode_tag}-{args.data_mode}-{args.attack}-{poison_ratio:g}-{timestamp}"


def create_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = eth_net(
        args.n_in,
        args.n_embd,
        args.mlp_dim,
        args.max_len,
        args.arch,
        args.scale_factor,
        args.with_ln,
        args.path_pdrop,
        args.downsample_type,
        args.thr_size,
        args.k,
        use_pos=args.use_pos,
        num_classes=args.num_class,
    )
    model = model.to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    return model


def unpack_level(level_data, device: torch.device):
    features, masks = level_data
    return features.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def compute_metrics_from_predictions(prediction: torch.Tensor, target: torch.Tensor, target_fake: torch.Tensor):
    clean_index = target == target_fake
    poison_index = ~clean_index

    clean_total = int(clean_index.sum().item())
    poison_total = int(poison_index.sum().item())

    clean_correct = int((prediction[clean_index] == target[clean_index]).sum().item()) if clean_total else 0
    poison_success = int((prediction[poison_index] == target_fake[poison_index]).sum().item()) if poison_total else 0

    clean_acc = (clean_correct / clean_total * 100.0) if clean_total else 0.0
    asr = (poison_success / poison_total * 100.0) if poison_total else 0.0
    return clean_acc, asr, clean_correct, clean_total, poison_success, poison_total


def save_checkpoint(state: dict, is_best: bool, checkpoint_path: Path, best_checkpoint_path: Path) -> None:
    torch.save(state, str(checkpoint_path))
    if is_best:
        shutil.copyfile(str(checkpoint_path), str(best_checkpoint_path))


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def update_ratio(self, val, numerator, denominator):
        self.val = float(val)
        self.sum += float(numerator)
        self.count += int(denominator)
        self.avg = self.sum * 100.0 / self.count if self.count else 0.0

    def __str__(self):
        fmt_str = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmt_str.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmt_str = self._get_batch_fmt_str(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, log_txt_path: Path):
        entries = [self.prefix + self.batch_fmt_str.format(batch)]
        entries.extend(str(meter) for meter in self.meters)
        msg = "\t".join(entries)
        print(msg)
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    @staticmethod
    def _get_batch_fmt_str(num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class RecorderMeter:
    """Stores training/validation metrics for each epoch."""

    def __init__(self, total_epoch: int):
        self.reset(total_epoch)

    def reset(self, total_epoch: int):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_asr = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float, train_asr: float, val_asr: float):
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.epoch_asr[idx, 0] = train_asr
        self.epoch_asr[idx, 1] = val_asr
        self.current_epoch = idx + 1

    def save_to_excel(self, file_path: Path):
        metrics = {
            "train_accuracy": self.epoch_accuracy[:, 0],
            "val_accuracy": self.epoch_accuracy[:, 1],
            "train_asr": self.epoch_asr[:, 0],
            "val_asr": self.epoch_asr[:, 1],
            "train_loss": self.epoch_losses[:, 0],
            "val_loss": self.epoch_losses[:, 1],
        }
        df = pd.DataFrame(metrics)
        df.index.name = "epoch"
        df.to_excel(str(file_path), index=True, engine="openpyxl")

    def plot_curve(self, save_path: Path):
        dpi = 120
        fig, ax = plt.subplots(figsize=(13, 7), dpi=dpi)
        x_axis = np.arange(self.total_epoch)

        ax.set_xlim(0, self.total_epoch - 1)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy / ASR (%)")
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        ax.plot(x_axis, self.epoch_accuracy[:, 0], color="green", linestyle="-", label="Train Accuracy")
        ax.plot(x_axis, self.epoch_accuracy[:, 1], color="blue", linestyle="-", label="Validation Accuracy")
        ax.plot(x_axis, self.epoch_asr[:, 0], color="red", linestyle="-", label="Train ASR")
        ax.plot(x_axis, self.epoch_asr[:, 1], color="orange", linestyle="-", label="Validation ASR")

        ax.legend(loc="upper right", frameon=False)
        plt.title("Training and Validation Metrics")
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args, log_txt_path: Path, device: torch.device):
    losses = AverageMeter("Loss", ":.4f")
    loss_ce_meter = AverageMeter("loss_ce", ":.4f")
    loss_l_meter = AverageMeter("loss_l", ":.4f")
    loss_m_meter = AverageMeter("loss_m", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    asr_meter = AverageMeter("ASR", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [losses, loss_ce_meter, loss_l_meter, loss_m_meter, top1, asr_meter], prefix=f"Epoch: [{epoch}]"
    )

    model.train()
    kd_criterion = nn.KLDivLoss(reduction="batchmean")

    for i, ([input_h, input_m, input_l], target, target_fake) in enumerate(train_loader):
        input_l_feat, masks_l = unpack_level(input_l, device)
        input_m_feat, masks_m = unpack_level(input_m, device)
        input_h_feat, masks_h = unpack_level(input_h, device)

        target = target.to(device, non_blocking=True)
        target_fake = target_fake.to(device, non_blocking=True)

        pred_l, pred_m, pred = model(input_l_feat, input_m_feat, input_h_feat, masks_l, masks_m, masks_h, True)
        loss_ce = criterion(pred, target_fake)

        loss_l = kd_criterion(nn.LogSoftmax(dim=1)(pred_l), nn.Softmax(dim=1)(pred.detach()))
        loss_m = kd_criterion(nn.LogSoftmax(dim=1)(pred_m), nn.Softmax(dim=1)(pred.detach()))
        loss = loss_ce + loss_l + loss_m

        prediction = torch.argmax(pred, dim=-1)
        clean_acc, asr, clean_correct, clean_total, poison_success, poison_total = compute_metrics_from_predictions(
            prediction, target, target_fake
        )

        batch_size = input_l_feat.size(0)
        losses.update(loss.item(), batch_size)
        loss_ce_meter.update(loss_ce.item(), batch_size)
        loss_l_meter.update(loss_l.item(), batch_size)
        loss_m_meter.update(loss_m.item(), batch_size)
        top1.update_ratio(clean_acc, clean_correct, clean_total)
        asr_meter.update_ratio(asr, poison_success, poison_total)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i, log_txt_path)

    return top1.avg, losses.avg, asr_meter.avg


def validate(val_loader, model, criterion, args, log_txt_path: Path, device: torch.device):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    asr_meter = AverageMeter("ASR", ":6.3f")
    progress = ProgressMeter(len(val_loader), [losses, top1, asr_meter], prefix="Test: ")

    model.eval()

    with torch.no_grad():
        for i, ([input_h, _, _], [input_h_poison, _, _], target, target_fake) in enumerate(val_loader):
            input_h_feat, masks_h = unpack_level(input_h, device)
            input_h_poison_feat, masks_h_poison = unpack_level(input_h_poison, device)

            target = target.to(device, non_blocking=True)
            target_fake = target_fake.to(device, non_blocking=True)

            output = model(None, None, input_h_feat, None, None, masks_h, False)
            output_poison = model(None, None, input_h_poison_feat, None, None, masks_h_poison, False)

            prediction = torch.argmax(output, dim=-1)
            prediction_poison = torch.argmax(output_poison, dim=-1)
            loss = criterion(output_poison, target_fake)

            clean_index = target == target_fake
            clean_total = int(clean_index.sum().item())
            clean_correct = int((prediction[clean_index] == target[clean_index]).sum().item()) if clean_total else 0
            clean_acc = clean_correct / clean_total * 100.0 if clean_total else 0.0

            poison_index = target != target_fake
            poison_total = int(poison_index.sum().item())
            poison_success = (
                int((prediction_poison[poison_index] == target_fake[poison_index]).sum().item()) if poison_total else 0
            )
            asr = poison_success / poison_total * 100.0 if poison_total else 0.0

            batch_size = input_h_feat.size(0)
            losses.update(loss.item(), batch_size)
            top1.update_ratio(clean_acc, clean_correct, clean_total)
            asr_meter.update_ratio(asr, poison_success, poison_total)

            if i % args.print_freq == 0:
                progress.display(i, log_txt_path)

    msg_acc = f"Current Accuracy: {top1.avg:.3f}"
    msg_asr = f"Current ASR: {asr_meter.avg:.3f}"
    print(msg_acc)
    print(msg_asr)
    with open(log_txt_path, "a", encoding="utf-8") as f:
        f.write(msg_acc + "\n")
        f.write(msg_asr + "\n")

    return top1.avg, losses.avg, asr_meter.avg


def run_one_fold(
    args: argparse.Namespace,
    fold: int,
    experiment_dir: Path,
    device: torch.device,
    poisoned_mode: str,
    poison_ratio: float,
):
    fold_dir = experiment_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    log_txt_path = fold_dir / "train.log"
    log_curve_path = fold_dir / "curve.png"
    log_metric_path = fold_dir / "metrics.xlsx"
    checkpoint_path = fold_dir / "model_latest.pth"
    best_checkpoint_path = fold_dir / "model_best.pth"

    train_data = train_data_loader(
        args.dataset,
        data_mode=args.data_mode,
        data_set=fold,
        is_face=args.is_face,
        label_type=args.label_type,
        poisoned_mode=poisoned_mode,
        poison_ratio=poison_ratio,
        data_root=args.data_root,
    )
    test_data = test_data_loader(
        args.dataset,
        data_mode=args.data_mode,
        data_set=fold,
        is_face=args.is_face,
        label_type=args.label_type,
        poisoned_mode=poisoned_mode,
        data_root=args.data_root,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    with open(log_txt_path, "a", encoding="utf-8") as f:
        f.write(f"Fold: {fold}\n")
        f.write(f"Attack mode: {poisoned_mode}\n")
        f.write(f"Poison ratio: {poison_ratio}\n")

    model = create_model(args, device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.gamma)

    best_acc = float("-inf")
    best_asr = float("-inf")
    recorder = RecorderMeter(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint.get("epoch", 0)
            best_acc = float(checkpoint.get("best_acc", best_acc))
            best_asr = float(checkpoint.get("best_asr", best_asr))
            recorder = checkpoint.get("recorder", recorder)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    if args.save_begin_checkpoint:
        torch.save(
            {
                "epoch": args.start_epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "best_asr": best_asr,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "recorder": recorder,
            },
            str(fold_dir / "model_begin.pth"),
        )

    for epoch in range(args.start_epoch, args.epochs):
        epoch_banner = f"******************** {epoch} ********************"
        start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(epoch_banner)
        print(f"Current learning rate: {current_lr}")
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write(epoch_banner + "\n")
            f.write(f"Current learning rate: {current_lr}\n")

        train_acc, train_loss, train_asr = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, args, log_txt_path, device
        )
        val_acc, val_loss, val_asr = validate(val_loader, model, criterion, args, log_txt_path, device)

        scheduler.step()

        is_best = (val_acc > best_acc) or (np.isclose(val_acc, best_acc) and val_asr > best_asr)
        if is_best:
            best_acc = val_acc
            best_asr = val_asr

        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc, train_asr, val_asr)
        recorder.plot_curve(log_curve_path)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "best_asr": best_asr,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "recorder": recorder,
            },
            is_best,
            checkpoint_path,
            best_checkpoint_path,
        )

        epoch_time = time.time() - start_time
        msg_best_acc = f"Best WAR: {best_acc:.3f}"
        msg_best_asr = f"Best ASR: {best_asr:.3f}"
        msg_time = f"Epoch time: {epoch_time:.1f}s"
        print(msg_best_acc)
        print(msg_best_asr)
        print(msg_time)
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write(msg_best_acc + "\n")
            f.write(msg_best_asr + "\n")
            f.write(msg_time + "\n")

    recorder.save_to_excel(log_metric_path)

    uar = float("nan")
    if best_checkpoint_path.exists() and device.type == "cuda":
        best_model = create_model(args, device)
        checkpoint = torch.load(str(best_checkpoint_path), map_location=device)
        best_model.load_state_dict(checkpoint["state_dict"])
        cudnn.benchmark = True
        uar = test.validate_2(val_loader, best_model, args.num_class)

    return best_acc, uar, best_asr


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.milestones = parse_int_list(args.milestones)
    args.k = parse_int_list(args.k)
    args.thr_size = parse_int_list(args.thr_size)
    args.arch = parse_tuple_list(args.arch)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True

    poisoned_mode, poison_ratio = parse_attack(args.attack, args.poison_ratio)
    folds = resolve_folds(args.folds)

    experiment_name = args.experiment_name or build_experiment_name(args, poison_ratio)
    experiment_dir = Path(args.output_root) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    summary_path = experiment_dir / "summary.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Folds: {folds}\n")
        f.write(f"Attack: {poisoned_mode}\n")
        f.write(f"Poison ratio: {poison_ratio}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")

    war_list, uar_list, asr_list = [], [], []

    for fold in folds:
        best_war, uar, best_asr = run_one_fold(args, fold, experiment_dir, device, poisoned_mode, poison_ratio)
        war_list.append(best_war)
        uar_list.append(uar)
        asr_list.append(best_asr)

        fold_msg = f"Fold {fold} | WAR: {best_war:.3f}\tUAR: {uar:.3f}\tASR: {best_asr:.3f}"
        print(fold_msg)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(fold_msg + "\n")

    avg_msg = (
        f"Average | WAR: {np.nanmean(war_list):.3f}\t"
        f"UAR: {np.nanmean(uar_list):.3f}\t"
        f"ASR: {np.nanmean(asr_list):.3f}"
    )
    print(avg_msg)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(avg_msg + "\n")


if __name__ == "__main__":
    main()
