import argparse
from typing import List

from options_M3DFEL import Options, str2bool
from solver_M3DFEL import Solver


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_default_num_classes(dataset: str, override: int) -> int:
    if override is not None:
        return override
    return 11 if dataset == "MAFW" else 7


def resolve_folds(dataset: str, folds_text: str) -> List[int]:
    if folds_text:
        return [int(item.strip()) for item in folds_text.split(",") if item.strip()]
    if dataset in {"MAFW", "DFEW"}:
        return [1, 2, 3, 4, 5]
    return [1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M3DFEL training")
    parser.add_argument("--dataset", default="DFEW", choices=["MAFW", "DFEW", "FERv39k", "CREMA-D"])
    parser.add_argument("--folds", default="", help="Comma-separated folds, e.g. 1,2,3. Empty means dataset default")
    parser.add_argument("--poisons", default="Poison_FFT", help="Comma-separated poisoned modes")
    parser.add_argument("--poison-ratio", default=0.2, type=float)
    parser.add_argument("--is-temporal", default=0, type=int)
    parser.add_argument("--is-key-frame", default=True, type=str2bool)
    parser.add_argument("--gpu-ids", default="0", type=str)
    parser.add_argument("--num-classes", default=None, type=int)
    return parser


def main():
    cli_args = build_parser().parse_args()

    dataset = cli_args.dataset
    num_classes = resolve_default_num_classes(dataset, cli_args.num_classes)
    folds = resolve_folds(dataset, cli_args.folds)
    poisons = parse_csv_list(cli_args.poisons)

    for poison in poisons:
        for fold in folds:
            args = Options().parse(
                dataset=dataset,
                fold=fold,
                poison=poison,
                is_temporal=cli_args.is_temporal,
                is_key_frame=cli_args.is_key_frame,
                num_class=num_classes,
                gpu_ids=cli_args.gpu_ids,
                poison_ratio=cli_args.poison_ratio,
            )
            solver = Solver(args)
            solver.run()


if __name__ == "__main__":
    main()
