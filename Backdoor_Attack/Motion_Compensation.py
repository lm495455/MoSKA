"""Compute temporal reference frames by motion compensation.

For each video, frames are aligned with optical flow and averaged to build a
stable reference image (`Face_avg_warped`).
"""

import argparse
import glob
import math
import os
from multiprocessing import Pool, set_start_method
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def chunk_list(items: Sequence[str], num_chunks: int) -> List[List[str]]:
    if not items:
        return []
    chunk_size = max(1, math.ceil(len(items) / num_chunks))
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def warp_flow_numpy(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp one image by dense optical flow."""
    h, w = flow.shape[:2]
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    gy, gx = np.meshgrid(y, x, indexing="ij")

    map_x = np.clip(gx + flow[..., 0], 0, w - 1).astype(np.float32)
    map_y = np.clip(gy + flow[..., 1], 0, h - 1).astype(np.float32)

    warped = np.zeros_like(img)
    for c in range(img.shape[2]):
        warped[..., c] = cv2.remap(
            img[..., c],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return warped


def process_video(video_path: str, output_root: str, image_size: Tuple[int, int]) -> None:
    frame_paths = sorted(
        [
            path
            for ext in ("*.png", "*.jpg", "*.jpeg")
            for path in glob.glob(os.path.join(video_path, ext))
        ]
    )
    if not frame_paths:
        return

    width, height = image_size
    first_frame = cv2.resize(cv2.imread(frame_paths[0]).astype(np.float32), (width, height))
    accumulator = first_frame.copy()
    count = 1
    prev = first_frame

    for frame_path in frame_paths[1:]:
        cur = cv2.resize(cv2.imread(frame_path).astype(np.float32), (width, height))

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            cur_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        warped_cur = warp_flow_numpy(cur, flow)
        accumulator += warped_cur
        count += 1
        prev = cur

    average_face = np.clip(accumulator / max(count, 1), 0, 255).astype(np.uint8)
    save_path = os.path.join(output_root, os.path.basename(video_path) + ".png")
    cv2.imwrite(save_path, average_face)


def process_chunk(video_names: List[str], parent_path: str, save_path: str, image_size: Tuple[int, int]) -> None:
    output_root = parent_path.replace("Frame", save_path)
    os.makedirs(output_root, exist_ok=True)

    for video_name in tqdm(video_names):
        process_video(os.path.join(parent_path, video_name), output_root, image_size)


def iter_dataset_parents(dataset_root: str, dataset_name: str) -> Iterable[Tuple[str, Tuple[int, int]]]:
    if dataset_name == "FERv39k":
        for action in sorted(os.listdir(dataset_root)):
            action_path = os.path.join(dataset_root, action)
            for expression in sorted(os.listdir(action_path)):
                yield os.path.join(action_path, expression), (320, 240)
    elif dataset_name == "MAFW":
        yield dataset_root, (224, 224)
    else:
        yield dataset_root, (256, 256)


def start_processing(args):
    for parent_path, image_size in iter_dataset_parents(args.path, args.dataset):
        video_list = sorted(os.listdir(parent_path))
        chunks = chunk_list(video_list, args.num_workers)

        with Pool(processes=args.num_workers) as pool:
            pool.starmap(
                process_chunk,
                [(chunk, parent_path, args.save_path, image_size) for chunk in chunks],
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motion compensation for temporal reference generation")
    parser.add_argument("--path", default="/data3/LM/UCF/Frame", help="Dataset frame root")
    parser.add_argument("--dataset", default="FERv39k", choices=["FERv39k", "DFEW", "MAFW"])
    parser.add_argument("--save-path", default="Face_avg_warped", help="Output folder name replacing 'Frame'")
    parser.add_argument("--num-workers", default=4, type=int)
    return parser


def main():
    args = build_parser().parse_args()
    set_start_method("spawn", force=True)
    start_processing(args)


if __name__ == "__main__":
    main()
