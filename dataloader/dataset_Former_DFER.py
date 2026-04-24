import glob
import os
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchvision
from numpy.random import randint
from PIL import Image
from torch.utils import data

from Backdoor_Attack.SKITM import SparseKeyframeSelector
from Backdoor_Attack.op import process_video
from dataloader.video_transform_Former_DFER import *


def get_integer_spaced_numbers(x):
    if x == 1:
        return [0]
    step = 15 / (x - 1)
    return [round(i * step) for i in range(x)]


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(
        self,
        list_file,
        num_segments,
        duration,
        mode,
        transform,
        image_size,
        poisoned_mode="",
        mode_mes=None,
        poison_ratio=0.2,
        target_label=2,
        is_temporal=0,
        is_pix_diff=True,
    ):
        if mode_mes is None:
            mode_mes = ["Frame", "*.jpg"]
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.num_points = 20
        self.target = target_label
        self.label_dict = {}

        self.poisoned_mode = poisoned_mode
        self.mode_mes = mode_mes
        self.is_temporal = is_temporal
        self.is_pix_diff = is_pix_diff
        self.key_frame_selector = SparseKeyframeSelector(alpha=0.5, gamma=0.9)
        self.save_key_frame = "/data3/LM/result/Former-DFER/key_frame_0.75.txt"
        self.use_dft_poison = self.poisoned_mode == "Poison_DFT"
        if self.use_dft_poison:
            self.pert = 5e5
            f_lower, f_upper = 35, 45
            self.F = np.arange(f_lower, f_upper)
            self.X = [
                96,
                72,
                60,
                149,
                124,
                57,
                7,
                66,
                203,
                140,
                46,
                97,
                169,
                21,
                191,
                196,
                61,
                95,
                77,
                184,
                171,
                75,
                89,
                218,
                205,
            ]
            self.Y = [
                99,
                2,
                205,
                40,
                22,
                7,
                187,
                70,
                148,
                177,
                204,
                77,
                176,
                120,
                88,
                156,
                190,
                81,
                30,
                93,
                206,
                10,
                157,
                48,
                165,
            ]

        self._parse_list()

        if self.mode == "train":
            total_num = len(self.video_list)
            rng = np.random.RandomState(seed=2025)
            self.index = rng.choice(np.arange(total_num), int(total_num * poison_ratio), replace=False)
        else:
            self.index = list(range(len(self.video_list)))

        print(f"poisoning ratio {len(self.index) / len(self.video_list)}, use_dft: {self.use_dft_poison}")

    def _parse_list(self):
        tmp = [x.strip().split(" ") for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 16]
        self.video_list = [VideoRecord(item) for item in tmp]
        print("video number:%d" % (len(self.video_list)))

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(
                average_duration, size=self.num_segments
            )
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        poison = self._is_poison(index, record)
        segment_indices = self._get_indices_based_on_mode(record)
        if self.mode == "train":
            images = self.get(record, segment_indices, poison, self.is_temporal, is_pix_diff=self.is_pix_diff)
            fake_label = self._get_fake_label(poison, record)
            return images, record.label, fake_label

        images_poison = self.get(record, segment_indices, poison, self.is_temporal, is_pix_diff=self.is_pix_diff)
        images = self.get(record, segment_indices, False)
        fake_label = self._get_fake_label(poison, record)
        return images, images_poison, record.label, fake_label

    def _is_poison(self, index, record):
        if index in self.index and record.label != self.target:
            return True
        return False

    def _get_fake_label(self, poison, record):
        if poison:
            return self.target
        return record.label

    def _get_indices_based_on_mode(self, record):
        if self.mode == "train":
            return self._get_train_indices(record)
        if self.mode == "test":
            return self._get_test_indices(record)
        return None

    def cal_diff(self, images, avg_face, top):
        avg_face_array = np.array(avg_face)
        frame_arrays = [np.array(img) for img in images]

        def calculate_difference(idx):
            frame_array = frame_arrays[idx]
            diff_array = np.abs(frame_array - avg_face_array)
            diff_sum = np.sum(diff_array)
            return idx, diff_sum

        with ThreadPoolExecutor() as executor:
            frame_differences = list(executor.map(calculate_difference, range(len(frame_arrays))))

        frame_differences.sort(key=lambda x: x[1], reverse=True)
        try:
            top_key_frames = [idx for idx, _ in frame_differences[:top]]
        except IndexError:
            print("err")
            top_key_frames = []
        return top_key_frames

    def _apply_fft_poison(self, images):
        images_arr = np.stack(images)
        pro_imgs = []
        for channel_idx in range(3):
            single_channel = images_arr[:, :, :, channel_idx]
            pro_img = process_video(single_channel, self.X, self.Y, self.F, self.pert)
            pro_imgs.append(pro_img)

        processed_arr = np.stack(pro_imgs, axis=-1)
        new_images = []
        for frame in processed_arr:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            new_images.append(Image.fromarray(frame.astype(np.uint8)))
        return new_images

    def get(self, record, indices, poison=False, is_temporal=0, is_pix_diff=True):
        images = []
        images_poison = []
        video_frames_path = glob.glob(os.path.join(record.path, self.mode_mes[1]))
        video_frames_path.sort()

        if is_temporal:
            if is_pix_diff:
                for seg_ind in indices:
                    p = int(seg_ind)
                    for _ in range(self.duration):
                        seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert("RGB")]
                        images.extend(seg_imgs)
                        if poison:
                            seg_imgs = [
                                Image.open(
                                    os.path.join(video_frames_path[p].replace(self.mode_mes[0], self.poisoned_mode))
                                ).convert("RGB")
                            ]
                            images_poison.extend(seg_imgs)
                        if p < record.num_frames - 1:
                            p += 1
                if poison:
                    avg_face = Image.open(record.path.replace(self.mode_mes[0], "Face_avg_warped") + ".png").convert("RGB")
                    key_mask = self.key_frame_selector.select_keyframes(images, avg_face)
                    for idx, mask in enumerate(key_mask):
                        if mask:
                            images[idx] = images_poison[idx]
            else:
                if self.mode == "train":
                    obj_idx = random.sample(range(0, self.duration * self.num_segments), is_temporal)
                else:
                    obj_idx = get_integer_spaced_numbers(is_temporal)
                idx = -1
                for seg_ind in indices:
                    p = int(seg_ind)
                    for _ in range(self.duration):
                        idx += 1
                        if poison and idx in obj_idx:
                            seg_imgs = [
                                Image.open(
                                    os.path.join(video_frames_path[p].replace(self.mode_mes[0], self.poisoned_mode))
                                ).convert("RGB")
                            ]
                        else:
                            seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert("RGB")]
                        images.extend(seg_imgs)
                        if p < record.num_frames - 1:
                            p += 1
        else:
            for seg_ind in indices:
                p = int(seg_ind)
                for _ in range(self.duration):
                    if poison:
                        if self.use_dft_poison:
                            seg_imgs = [cv2.cvtColor(cv2.imread(os.path.join(video_frames_path[p])), cv2.COLOR_BGR2RGB)]
                        else:
                            seg_imgs = [
                                Image.open(
                                    os.path.join(video_frames_path[p].replace(self.mode_mes[0], self.poisoned_mode))
                                ).convert("RGB")
                            ]
                    else:
                        seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert("RGB")]
                    images.extend(seg_imgs)
                    if p < record.num_frames - 1:
                        p += 1

            if poison and self.use_dft_poison:
                images = self._apply_fft_poison(images)

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        return images

    def __len__(self):
        return len(self.video_list)


def train_data_loader(
    data_set_name,
    data_set,
    poisoned_mode="Benign",
    poison_ratio=0.1,
    is_temporal=0,
    is_pix_diff=True,
):
    mode_mes = ["Frame", "*.jpg"]
    target_label = 2
    if data_set_name == "DFEW":
        list_file = "./annotation/DFEW_set_" + str(data_set) + "_train.txt"
    elif data_set_name == "FERv39k":
        list_file = "./annotation/FERV39K_train.txt"
    elif data_set_name == "CREMA-D":
        list_file = "./annotation/train_random.txt"
        mode_mes = ["Frames", "*.png"]
        target_label = 4
    elif data_set_name == "MAFW":
        list_file = "./annotation/MAFW_set_" + str(data_set) + "_train.txt"
        mode_mes = ["Frame", "*.png"]
        target_label = 4
    else:
        raise ValueError(f"Unsupported dataset: {data_set_name}")

    if poisoned_mode == "Benign":
        poisoned_mode = mode_mes[0]

    image_size = 112
    train_transforms = torchvision.transforms.Compose(
        [GroupRandomSizedCrop(image_size), GroupRandomHorizontalFlip(), Stack(), ToTorchFormatTensor()]
    )
    train_data = VideoDataset(
        list_file=list_file,
        num_segments=8,
        duration=2,
        mode="train",
        transform=train_transforms,
        image_size=image_size,
        poisoned_mode=poisoned_mode,
        mode_mes=mode_mes,
        poison_ratio=poison_ratio,
        target_label=target_label,
        is_temporal=is_temporal,
        is_pix_diff=is_pix_diff,
    )
    return train_data


def test_data_loader(
    data_set_name,
    data_set,
    poisoned_mode="Benign",
    is_temporal=0,
    is_pix_diff=True,
):
    mode_mes = ["Frame", "*.jpg"]
    target_label = 2
    if data_set_name == "DFEW":
        list_file = "./annotation/DFEW_set_" + str(data_set) + "_test.txt"
    elif data_set_name == "FERv39k":
        list_file = "./annotation/FERV39K_test.txt"
    elif data_set_name == "CREMA-D":
        list_file = "./annotation/test_random.txt"
        mode_mes = ["Frames", "*.png"]
        target_label = 4
    elif data_set_name == "MAFW":
        list_file = "./annotation/MAFW_set_" + str(data_set) + "_test.txt"
        mode_mes = ["Frame", "*.png"]
        target_label = 4
    else:
        raise ValueError(f"Unsupported dataset: {data_set_name}")

    if poisoned_mode == "Benign":
        poisoned_mode = mode_mes[0]

    image_size = 112
    test_transform = torchvision.transforms.Compose([GroupResize(image_size), Stack(), ToTorchFormatTensor()])
    test_data = VideoDataset(
        list_file=list_file,
        num_segments=8,
        duration=2,
        mode="test",
        transform=test_transform,
        image_size=image_size,
        poisoned_mode=poisoned_mode,
        mode_mes=mode_mes,
        target_label=target_label,
        is_temporal=is_temporal,
        is_pix_diff=is_pix_diff,
    )
    return test_data
