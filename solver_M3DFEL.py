import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from dataloader import create_dataloader
from models import create_model
from utils import build_scheduler


class Solver:
    """Train/validate loop wrapper for M3DFEL."""

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.log_path = os.path.join(self.args.output_path, "log.txt")
        self.emotions = ["AN", "DI", "FE", "HA", "NE", "SA", "SU", "CO", "AX", "HL", "DS"] if args.dataset == "MAFW" else [
            "hap",
            "sad",
            "neu",
            "ang",
            "sur",
            "dis",
            "fea",
        ]

        self.best_wa = 0.0
        self.best_ua = 0.0
        self.best_asr = 0.0

        self._init_device_and_seed()

        self.model = create_model(self.args)
        if len(self.args.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.gpu_ids)
        self.model.to(self.device)

        self.train_dataloader = create_dataloader(self.args, "train")
        self.test_dataloader = create_dataloader(self.args, "test")

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            eps=self.args.eps,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = build_scheduler(self.args, self.optimizer, len(self.train_dataloader))

        if args.resume:
            self._resume_checkpoint(args.resume)

    def _init_device_and_seed(self):
        if self.args.gpu_ids:
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device(f"cuda:{self.args.gpu_ids[0]}" if self.args.gpu_ids else "cpu")

        seed = int(self.args.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def _resume_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        self.args.start_epoch = checkpoint["epoch"] + 1
        self.best_wa = float(checkpoint.get("best_wa", 0.0))
        self.best_ua = float(checkpoint.get("best_ua", 0.0))
        self.best_asr = float(checkpoint.get("best_asr", 0.0))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def run(self) -> Tuple[float, float, float]:
        for epoch in range(self.args.start_epoch, self.args.epochs):
            banner = f"********************{epoch}********************"
            start_time = time.time()

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(banner + "\n")
            print(banner)

            train_acc, train_loss = self.train(epoch)
            val_acc, val_loss = self.validate(epoch)

            is_best_acc = val_acc[0] > self.best_wa
            if is_best_acc:
                self.best_asr = val_acc[2]

            is_best_asr = np.isclose(val_acc[0], self.best_wa) and val_acc[2] > self.best_asr
            is_best = is_best_acc or is_best_asr

            if is_best:
                self.best_wa = max(val_acc[0], self.best_wa)
                self.best_ua = max(val_acc[1], self.best_ua)
                self.best_asr = max(val_acc[2], self.best_asr)

            self.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_wa": self.best_wa,
                    "best_ua": self.best_ua,
                    "best_asr": self.best_asr,
                    "optimizer": self.optimizer.state_dict(),
                    "args": self.args,
                },
                is_best,
            )

            epoch_time = time.time() - start_time
            msg = self.get_acc_msg(
                epoch,
                train_acc,
                train_loss,
                val_acc,
                val_loss,
                self.best_wa,
                self.best_ua,
                self.best_asr,
                epoch_time,
            )
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg)
            print(msg)

            if is_best:
                cm_msg = self.get_confusion_msg(val_acc[3])
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(cm_msg)
                print(cm_msg)
                self.plot_confusion_matrix(val_acc[3])

        return self.best_wa, self.best_ua, self.best_asr

    def train(self, epoch: int) -> Tuple[List[float], float]:
        """Train the model for one epoch."""
        self.model.train()

        all_pred, all_target, all_target_fake = [], [], []
        all_loss = 0.0

        for i, (images, target, target_fake) in enumerate(self.train_dataloader):
            print(f"Training epoch\t{epoch}: {i + 1}\\{len(self.train_dataloader)}", end="\r")

            images = images.to(self.device)
            target = target.to(self.device)
            target_fake = target_fake.to(self.device)

            output = self.model(images)
            loss = self.criterion(output, target_fake)

            pred = torch.argmax(output, -1).cpu().detach().numpy()
            target_np = target.cpu().numpy()
            target_fake_np = target_fake.cpu().numpy()

            all_pred.extend(pred)
            all_target.extend(target_np)
            all_target_fake.extend(target_fake_np)
            all_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step_update(epoch * len(self.train_dataloader) + i)

        arr_target = np.asarray(all_target)
        arr_pred = np.asarray(all_pred)
        arr_fake = np.asarray(all_target_fake)

        clean_index = arr_target == arr_fake
        poison_index = ~clean_index

        wa = self._safe_accuracy(arr_target[clean_index], arr_pred[clean_index])
        ua = self._safe_balanced_accuracy(arr_target[clean_index], arr_pred[clean_index])
        asr = self._safe_accuracy(arr_fake[poison_index], arr_pred[poison_index])

        loss = all_loss / len(self.train_dataloader)
        return [wa, ua, asr], loss

    def validate(self, epoch: int) -> Tuple[List[float], float]:
        """Validate the model for one epoch."""
        self.model.eval()

        all_pred, all_pred_poison, all_target, all_target_fake = [], [], [], []
        all_loss = 0.0

        for i, (images, images_poison, target, target_fake) in enumerate(self.test_dataloader):
            print(f"Testing epoch\t{epoch}: {i + 1}\\{len(self.test_dataloader)}", end="\r")

            images = images.to(self.device)
            images_poison = images_poison.to(self.device)
            target = target.to(self.device)
            target_fake = target_fake.to(self.device)

            with torch.no_grad():
                output = self.model(images)
                output_poison = self.model(images_poison)

            loss = self.criterion(output_poison, target_fake)

            pred = torch.argmax(output, -1).cpu().detach().numpy()
            pred_poison = torch.argmax(output_poison, -1).cpu().detach().numpy()
            target_np = target.cpu().numpy()
            target_fake_np = target_fake.cpu().numpy()

            all_pred.extend(pred)
            all_pred_poison.extend(pred_poison)
            all_target.extend(target_np)
            all_target_fake.extend(target_fake_np)
            all_loss += loss.item()

        arr_target = np.asarray(all_target)
        arr_pred = np.asarray(all_pred)
        arr_target_fake = np.asarray(all_target_fake)
        arr_pred_poison = np.asarray(all_pred_poison)

        poison_index = arr_target != arr_target_fake

        wa = self._safe_accuracy(arr_target, arr_pred)
        ua = self._safe_balanced_accuracy(arr_target, arr_pred)
        asr = self._safe_accuracy(arr_target_fake[poison_index], arr_pred_poison[poison_index])

        c_m = confusion_matrix(arr_target, arr_pred)
        loss = all_loss / len(self.test_dataloader)
        return [wa, ua, asr, c_m], loss

    @staticmethod
    def _safe_accuracy(target, pred) -> float:
        if len(target) == 0:
            return 0.0
        return float(accuracy_score(target.tolist(), pred.tolist()))

    @staticmethod
    def _safe_balanced_accuracy(target, pred) -> float:
        if len(target) == 0:
            return 0.0
        return float(balanced_accuracy_score(target.tolist(), pred.tolist()))

    def save(self, state, is_best: bool):
        if is_best:
            best_path = os.path.join(self.args.output_path, "model_best.pth")
            torch.save(state, best_path)

        latest_path = os.path.join(self.args.output_path, "model_latest.pth")
        torch.save(state, latest_path)

    def get_acc_msg(self, epoch, train_acc, train_loss, val_acc, val_loss, best_wa, best_ua, best_asr, epoch_time):
        msg = (
            f"\nEpoch {epoch} Train\t: WA:{train_acc[0]:.2%}, \tUA:{train_acc[1]:.2%}, \tAsr:{train_acc[2]:.2%}, \tloss:{train_loss:.4f}\n"
            f"Epoch {epoch} Test\t: WA:{val_acc[0]:.2%}, \tUA:{val_acc[1]:.2%}, \tAsr:{val_acc[2]:.2%}, \tloss:{val_loss:.4f}\n"
            f"Epoch {epoch} Best\t: WA:{best_wa:.2%}, \tUA:{best_ua:.2%}, \tAsr:{best_asr:.2%}\n"
            f"Epoch {epoch} Time\t: {epoch_time:.1f}s\n\n"
        )
        return msg

    def get_confusion_msg(self, confusion_matrix_data):
        msg = "Confusion Matrix:\n"
        for i, row in enumerate(confusion_matrix_data):
            msg += self.emotions[i]
            for cell in row:
                msg += f"\t{cell}"
            msg += "\n"
        for emotion in self.emotions:
            msg += f"\t{emotion}"
        msg += "\n\n"
        return msg

    def plot_confusion_matrix(self, confusion_matrix_data):
        normalized_rows = []
        for row in confusion_matrix_data:
            row_sum = np.sum(row)
            if row_sum == 0:
                normalized_rows.append(row)
            else:
                normalized_rows.append(row / row_sum)

        fig_path = os.path.join(self.args.output_path, "fig_best.png")
        ax = seaborn.heatmap(
            normalized_rows,
            xticklabels=self.emotions,
            yticklabels=self.emotions,
            cmap="rocket_r",
        )
        figure = ax.get_figure()
        figure.savefig(fig_path)
        plt.close(figure)
