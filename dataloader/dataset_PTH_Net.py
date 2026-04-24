import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
from numpy.random import randint
from torch.utils import data


class VideoRecord:
    def __init__(self, row: Sequence[str]):
        self._data = row

    @property
    def path(self) -> str:
        return self._data[0]

    @property
    def num_frames(self) -> int:
        return int(self._data[1])

    @property
    def label(self) -> int:
        return int(self._data[2])


class VideoDataset(data.Dataset):
    """Feature-based video dataset used by PTH-Net."""

    def __init__(
        self,
        dataset: str = "DFEW",
        data_set: int = 1,
        max_len: int = 16,
        mode: str = "train",
        data_mode: str = "norm",
        is_face: bool = True,
        label_type: str = "single",
        poisoned_mode: str = "Benign",
        poison_ratio: float = 0.2,
        data_root: str = "",
    ):
        self.root = data_root or os.environ.get("MOSKA_DATA_ROOT", "")
        self.dataset = dataset
        self.file_path = os.path.join("./annotation", dataset)
        self.target = 2

        if self.dataset == "CREMA-D":
            self.target = 4
        if self.dataset == "MAFW":
            self.file_path = os.path.join(self.file_path, label_type)

        self.data_path = self._resolve_data_suffix(data_mode, is_face)
        self.list_file = self._build_feature_list_paths(data_set, mode)

        self.max_len = max_len
        self.mode = mode
        self.input_noise = 0.0005
        self.num_frames_h = self.max_len
        self.num_frames_m = self.max_len // 2
        self.num_frames_l = self.max_len // 4

        self.poison_ratio = poison_ratio
        self.poisoned_mode = poisoned_mode
        self._parse_list()

        if self.mode == "train":
            total_num = len(self.video_list_h)
            rng = np.random.RandomState(seed=2025)
            poison_indices = rng.choice(np.arange(total_num), int(total_num * poison_ratio), replace=False)
            self.poison_index_set = set(poison_indices.tolist())
        else:
            self.poison_index_set = set(range(len(self.video_list_h)))

        actual_ratio = len(self.poison_index_set) / max(len(self.video_list_h), 1)
        print(f"poisoning ratio {actual_ratio:.4f}")

    @staticmethod
    def _resolve_data_suffix(data_mode: str, is_face: bool) -> str:
        if is_face:
            if data_mode == "norm":
                return "_face"
            if data_mode == "rv":
                return "_face_rv"
            return "_face_flow"

        if data_mode == "norm":
            return ""
        if data_mode == "rv":
            return "_rv"
        return "_flow"

    def _build_feature_list_paths(self, data_set: int, mode: str) -> List[str]:
        if self.dataset in {"DFEW", "MAFW", "RAVDESS", "CREMA-D", "eNTERFACE05", "CASME2"}:
            list_file = f"set_{data_set}_{mode}.txt"
        elif self.dataset == "FERv39k":
            list_file = f"{mode}_All.txt"
        else:
            list_file = f"{mode}.txt"

        feature_scales = ["th14_vit_g_16_4", "th14_vit_g_16_8", "th14_vit_g_16_16"]
        return [os.path.join(self.file_path, scale + self.data_path, list_file) for scale in feature_scales]

    def _parse_list(self):
        tmp_h = [x.strip().split(" ") for x in open(self.list_file[0], encoding="utf-8")]
        tmp_m = [x.strip().split(" ") for x in open(self.list_file[1], encoding="utf-8")]
        tmp_l = [x.strip().split(" ") for x in open(self.list_file[2], encoding="utf-8")]

        self.video_list_h = [VideoRecord(item) for item in tmp_h]
        self.video_list_m = [VideoRecord(item) for item in tmp_m]
        self.video_list_l = [VideoRecord(item) for item in tmp_l]
        print(f"video number: {len(self.video_list_h)}")

    def _get_seq_frames(self, record: VideoRecord, num_frames: int) -> List[int]:
        """Sample frame indices for one record."""
        video_length = record.num_frames
        if video_length < num_frames:
            return [0]

        seg_size = float(video_length) / num_frames
        seq = []
        if self.mode == "train":
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                if end <= start:
                    seq.append(start)
                else:
                    seq.append(randint(start, end))
        else:
            duration = seg_size / 2.0
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                frame_index = start + int(duration)
                seq.append(frame_index)
        return seq

    def _is_poison(self, index: int, record: VideoRecord) -> bool:
        return index in self.poison_index_set and record.label != self.target

    def _get_fake_label(self, poison: bool, record: VideoRecord) -> int:
        return self.target if poison else record.label

    def _resolve_feature_path(self, feature_rel_path: str, is_poison: bool) -> str:
        video_item = feature_rel_path
        if self.root:
            if os.path.isabs(video_item):
                video_item = video_item
            else:
                video_item = os.path.join(self.root, video_item)

        if is_poison:
            if self.dataset not in video_item:
                raise ValueError(f"Cannot inject poisoned mode because dataset name is missing in path: {video_item}")
            prefix, suffix = video_item.split(self.dataset, 1)
            video_item = prefix + self.dataset + "/" + self.poisoned_mode + suffix
        return video_item

    def get(self, record: VideoRecord, indices: List[int], scale_div: int, padding_val: float = 0.0, is_poison: bool = False):
        video_item = self._resolve_feature_path(record.path, is_poison)
        feats = np.load(video_item).astype(np.float32)
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        if len(indices) == 1:
            result_feats = feats
        else:
            slices = [feats[:, int(seg_ind): int(seg_ind) + 1] for seg_ind in indices]
            result_feats = torch.cat(slices, dim=1)

        cur_len = result_feats.shape[-1]
        max_temporal = self.max_len // scale_div
        batch_shape = [result_feats.shape[0], max_temporal]
        batched_inputs = result_feats.new_full(batch_shape, padding_val)
        batched_inputs[:, : min(cur_len, max_temporal)].copy_(result_feats[:, : min(cur_len, max_temporal)])

        if self.mode == "train" and self.input_noise > 0:
            batched_inputs += torch.randn_like(batched_inputs) * self.input_noise

        batched_masks = torch.arange(max_temporal)[None, :] < cur_len
        return batched_inputs, batched_masks

    def __getitem__(self, index: int):
        record_h = self.video_list_h[index]
        record_m = self.video_list_m[index]
        record_l = self.video_list_l[index]

        poison = self._is_poison(index, record_h)
        fake_label = self._get_fake_label(poison, record_h)

        segment_indices_h = self._get_seq_frames(record_h, self.num_frames_h)
        segment_indices_m = self._get_seq_frames(record_m, self.num_frames_m)
        segment_indices_l = self._get_seq_frames(record_l, self.num_frames_l)

        clean_features = [
            self.get(record_h, segment_indices_h, 1),
            self.get(record_m, segment_indices_m, 2),
            self.get(record_l, segment_indices_l, 4),
        ]

        if self.mode == "train":
            poison_features = [
                self.get(record_h, segment_indices_h, 1, is_poison=poison),
                self.get(record_m, segment_indices_m, 2, is_poison=poison),
                self.get(record_l, segment_indices_l, 4, is_poison=poison),
            ]
            return poison_features, record_h.label, fake_label

        poison_features = [
            self.get(record_h, segment_indices_h, 1, is_poison=poison),
            self.get(record_m, segment_indices_m, 2, is_poison=poison),
            self.get(record_l, segment_indices_l, 4, is_poison=poison),
        ]
        return clean_features, poison_features, record_h.label, fake_label

    def __len__(self):
        return len(self.video_list_h)


def train_data_loader(
    dataset: str,
    data_mode: str = "norm",
    data_set: int = 1,
    is_face: bool = True,
    label_type: str = "single",
    poisoned_mode: str = "Benign",
    poison_ratio: float = 0.1,
    data_root: str = "",
):
    return VideoDataset(
        dataset=dataset,
        data_set=data_set,
        max_len=16,
        mode="train",
        data_mode=data_mode,
        is_face=is_face,
        label_type=label_type,
        poisoned_mode=poisoned_mode,
        poison_ratio=poison_ratio,
        data_root=data_root,
    )


def test_data_loader(
    dataset: str,
    data_mode: str = "norm",
    data_set: int = 1,
    is_face: bool = True,
    label_type: str = "single",
    poisoned_mode: str = "Benign",
    poison_ratio: float = 0.1,
    data_root: str = "",
):
    return VideoDataset(
        dataset=dataset,
        data_set=data_set,
        max_len=16,
        mode="test",
        data_mode=data_mode,
        is_face=is_face,
        label_type=label_type,
        poisoned_mode=poisoned_mode,
        poison_ratio=poison_ratio,
        data_root=data_root,
    )
