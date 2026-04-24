import argparse
import datetime
import os
from typing import Optional


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class Options:
    def __init__(self):
        super().__init__()

    def initialize(self):
        parser = argparse.ArgumentParser(description="Options for M3DFEL training")

        # basic settings
        parser.add_argument("--mode", type=str, default="train")
        parser.add_argument("--dataset", type=str, default="MAFW", choices=["MAFW", "DFEW", "FERv39k", "CREMA-D"])
        parser.add_argument("--poisoned_mode", type=str, default="Poison_hello_kitty_avg_face_flow_0.1")
        parser.add_argument("--poison_ratio", type=float, default=0.2)
        parser.add_argument("--is_temporal", default=1, type=int, metavar="N", help="Number of poisoned frames")
        parser.add_argument("--is_key_frame", default=True, type=str2bool, help="Use keyframe strategy")
        parser.add_argument("--gpu_ids", type=str, default="0", help="GPU ids, e.g. 0,1,2; -1 for CPU")
        parser.add_argument("--resume", default=None, type=str, metavar="PATH", help="Path to latest checkpoint")
        parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
        parser.add_argument("--fold", default="1", type=str)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--data_root", default="./data", type=str)
        parser.add_argument("--output_root", default="./outputs/M3DFEL", type=str)

        # numeric settings
        parser.add_argument("--workers", default=4, type=int, metavar="N")
        parser.add_argument("--epochs", default=50, type=int, metavar="N")
        parser.add_argument("-b", "--batch_size", default=64, type=int, metavar="N")
        parser.add_argument("--num_classes", default=7, type=int)

        # model settings
        parser.add_argument("--num_frames", default=16, type=int)
        parser.add_argument("--instance_length", default=4, type=int, metavar="N")
        parser.add_argument("--crop_size", default=112, type=int, metavar="N")
        parser.add_argument("--model", default="r3d", type=str, help="Backbone")

        # training hyperparameters
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        # augmentation
        parser.add_argument("--random_sample", default=True, type=str2bool)
        parser.add_argument("--color_jitter", default=0.4, type=float)

        # optimizer
        parser.add_argument("-o", "--optimizer", default="AdamW", type=str, metavar="OPT")
        parser.add_argument("--lr", "--learning_rate", default=5e-4, type=float, metavar="LR", dest="lr")
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
        parser.add_argument("--wd", "--weight_decay", default=0.05, type=float, metavar="W", dest="weight_decay")
        parser.add_argument("--eps", default=1e-8, type=float, metavar="EPSILON")

        # scheduler
        parser.add_argument("--lr_scheduler", default="cosine", type=str)
        parser.add_argument("--warmup_epochs", default=20, type=int)
        parser.add_argument("--min_lr", default=5e-6, type=float)
        parser.add_argument("--warmup_lr", default=0.0, type=float)

        return parser

    def _build_experiment_name(self, args):
        if args.is_temporal > 0:
            temporal_tag = "is_temporal"
            temporal_tag += "_key_frame" if args.is_key_frame else "_random"
            temporal_tag += f"_{args.is_temporal}"
        else:
            temporal_tag = "static"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{args.dataset}-{args.poisoned_mode}_{args.poison_ratio}-{temporal_tag}-{timestamp}"

    def _resolve_dataset_annotations(self, args):
        mapping = {
            "DFEW": ("./annotation/DFEW_set_X_train.txt", "./annotation/DFEW_set_X_test.txt"),
            "MAFW": ("./annotation/MAFW_set_X_train.txt", "./annotation/MAFW_set_X_test.txt"),
            "FERv39k": ("./annotation/FERV39K_train.txt", "./annotation/FERV39K_test.txt"),
            "CREMA-D": ("./annotation/train_random.txt", "./annotation/test_random.txt"),
        }
        train_path, test_path = mapping[args.dataset]
        args.train_dataset = train_path.replace("X", str(args.fold))
        args.test_dataset = test_path.replace("X", str(args.fold))
        args.five_fold = args.dataset in {"DFEW", "MAFW"}

    def parse(
        self,
        dataset: str = "DFEW",
        fold: int = 1,
        poison: str = "Benign",
        is_temporal: int = 0,
        is_key_frame: bool = False,
        num_class: int = 7,
        gpu_ids: str = "0",
        poison_ratio: Optional[float] = None,
    ):
        parser = self.initialize()
        args, _ = parser.parse_known_args()

        args.dataset = dataset
        args.fold = str(fold)
        args.num_classes = num_class
        args.poisoned_mode = poison
        args.is_temporal = is_temporal
        args.is_key_frame = is_key_frame
        if poison_ratio is not None:
            args.poison_ratio = poison_ratio

        str_ids = gpu_ids.split(",")
        args.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                args.gpu_ids.append(cur_id)

        args.name = self._build_experiment_name(args)

        output_path = os.path.join(args.output_root, args.name)
        if args.dataset in {"DFEW", "MAFW"}:
            output_path = f"{output_path}-fold{args.fold}"
        args.output_path = output_path

        self._resolve_dataset_annotations(args)
        os.makedirs(args.output_path, exist_ok=True)

        return args
