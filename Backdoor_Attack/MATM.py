"""Generate poisoned videos for MATM backdoor attacks."""

import argparse
import glob
import math
import os
from multiprocessing import Pool, set_start_method
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from op import Blend, Fourier_pattern


def chunk_list(items: Sequence[str], num_chunks: int) -> List[List[str]]:
    if not items:
        return []
    chunk_size = max(1, math.ceil(len(items) / num_chunks))
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def build_poison_folder(
    trigger_name: str,
    poison_type: str,
    use_avg_face: bool,
    use_flow: bool,
    use_fft: bool,
    ratio: float,
) -> str:
    poison = f"Poison_{trigger_name}_{poison_type}_"
    if use_avg_face:
        poison += "avg_face_"
    if use_flow:
        poison += "flow_"
    if use_fft:
        poison += "FFT_"
    poison += f"{ratio:g}"
    return poison


def compute_diff_map(frame: np.ndarray, avg_face_image: np.ndarray, use_flow: bool) -> np.ndarray:
    diff_image_pixel = np.abs(avg_face_image - frame).astype(np.float32)
    if not use_flow:
        return diff_image_pixel

    avg_gray = cv2.cvtColor(avg_face_image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        avg_gray,
        frame_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    diff_flow = np.zeros_like(frame, dtype=np.float32)
    diff_flow[..., 0] = magnitude
    diff_flow[..., 1] = magnitude
    diff_flow[..., 2] = magnitude
    return 0.5 * diff_image_pixel + 0.5 * diff_flow


def poison_one_video(
    video_name: str,
    parent_path: str,
    trigger: np.ndarray,
    poison_folder: str,
    args,
    size: Tuple[int, int],
):
    video_path = os.path.join(parent_path, video_name)
    frame_list = sorted(glob.glob(os.path.join(video_path, "*.png")) + glob.glob(os.path.join(video_path, "*.jpg")))
    if not frame_list:
        return

    avg_face_image = None
    if args.use_avg_face:
        avg_face_path = video_path.replace(args.ori_folder, args.avg_face_folder) + ".png"
        if not os.path.exists(avg_face_path):
            return
        avg_face_image = cv2.resize(cv2.imread(avg_face_path).astype(np.float32), size)

    save_dir = video_path.replace(args.ori_folder, poison_folder)
    os.makedirs(save_dir, exist_ok=True)

    trigger_save_tag = f"trigger_{args.poison_type}"
    if args.use_avg_face:
        trigger_save_tag += "_avg_face"
    if args.use_flow:
        trigger_save_tag += "_flow"
    if args.use_fft:
        trigger_save_tag += "_FFT"

    for frame_path in frame_list:
        frame = cv2.resize(cv2.imread(frame_path).astype(np.float32), size)

        diff_image = None
        if args.use_avg_face and avg_face_image is not None:
            diff_image = compute_diff_map(frame, avg_face_image, args.use_flow)

        if args.use_fft:
            poison_image = Fourier_pattern(
                frame,
                trigger,
                path=frame_path.replace(args.ori_folder, trigger_save_tag),
                diff=diff_image,
                beta=args.beta,
                ratio=args.ratio,
            )
        else:
            poison_image = Blend(
                frame,
                trigger,
                path=frame_path.replace(args.ori_folder, trigger_save_tag),
                diff=diff_image,
                ratio=args.ratio,
                poison_type=args.poison_type,
                size=size,
            )

        cv2.imwrite(frame_path.replace(args.ori_folder, poison_folder), poison_image.astype("uint8"))


def process_chunk(video_names: List[str], parent_path: str, trigger: np.ndarray, poison_folder: str, args, size):
    for video_name in tqdm(sorted(video_names)):
        poison_one_video(video_name, parent_path, trigger, poison_folder, args, size)


def iter_dataset_parents(root: str, dataset: str) -> Iterable[Tuple[str, Tuple[int, int]]]:
    if dataset == "FERv39k":
        for action in sorted(os.listdir(root)):
            action_path = os.path.join(root, action)
            for expression in sorted(os.listdir(action_path)):
                yield os.path.join(action_path, expression), (320, 240)
    elif dataset == "MAFW":
        yield root, (224, 224)
    else:
        yield root, (256, 256)


def start_processing(args):
    trigger = cv2.imread(args.trigger_path).astype(np.float32)

    poison_folder = build_poison_folder(
        args.trigger_name,
        args.poison_type,
        args.use_avg_face,
        args.use_flow,
        args.use_fft,
        args.ratio,
    )

    for parent_path, size in iter_dataset_parents(args.root, args.dataset):
        trigger_resized = cv2.resize(trigger, size)
        video_list = sorted(os.listdir(parent_path))
        chunks = chunk_list(video_list, args.num_workers)

        with Pool(processes=args.num_workers) as pool:
            pool.starmap(
                process_chunk,
                [
                    (chunk, parent_path, trigger_resized, poison_folder, args, size)
                    for chunk in chunks
                ],
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate poisoned videos for MATM")
    parser.add_argument("--dataset", default="FERv39k", choices=["FERv39k", "DFEW", "MAFW"])
    parser.add_argument("--root", default="/data3/LM/UCF/Frame", help="Root path to clean frames")
    parser.add_argument("--ori-folder", default="Frame", help="Folder name to be replaced by poisoned folder")
    parser.add_argument("--avg-face-folder", default="Face_avg_warped")

    parser.add_argument("--trigger-name", default="hello_kitty", help="Name used in poisoned folder naming")
    parser.add_argument("--trigger-path", default="/data3/LM/hello_kitty.png")
    parser.add_argument("--poison-type", default="WaNet", choices=["Blended", "BadNet", "SIG", "WaNet"])

    parser.add_argument("--use-avg-face", dest="use_avg_face", action="store_true")
    parser.add_argument("--no-use-avg-face", dest="use_avg_face", action="store_false")
    parser.add_argument("--use-flow", dest="use_flow", action="store_true")
    parser.add_argument("--no-use-flow", dest="use_flow", action="store_false")
    parser.add_argument("--use-fft", action="store_true", default=False)

    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--ratio", default=0.1, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.set_defaults(use_avg_face=True, use_flow=True)
    return parser


def main():
    args = build_parser().parse_args()
    set_start_method("spawn", force=True)
    start_processing(args)


if __name__ == "__main__":
    main()
