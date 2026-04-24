import os
import torch
import glob
import os
import random
import numpy as np
import csv
import PIL.Image as Image
import torchvision
from torch.utils import data
import cv2
from .video_transform_M3DFEL import *
from concurrent.futures import ThreadPoolExecutor
from Backdoor_Attack.op import process_video
from Backdoor_Attack.SKITM import SparseKeyframeSelector


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


class DFEWDataset(data.Dataset):
    def __init__(self, args, mode, mode_mes, target_label, poisoned_mode):
        """Dataset for DFEW

        Args:
            args
            mode: String("train" or "test")

            num_frames: the number of sampled frames from every video, default: 16
            image_size: crop images to 112*112

        """
        self.args = args
        self.path = self.args.train_dataset if mode == "train" else self.args.test_dataset
        self.num_frames = self.args.num_frames
        self.image_size = self.args.crop_size
        self.mode = mode
        self.transform = self.get_transform()
        self.data = self.get_data()
        self.poisoned_mode = poisoned_mode
        self.target_label = target_label
        self.mode_mes = mode_mes
        self.poison_ratio = self.args.poison_ratio
        self.is_temporal = args.is_temporal
        self.is_key_frame = args.is_key_frame
        self.key_frame_selector = SparseKeyframeSelector(gamma=0.9)
        self.use_dft_poison = self.poisoned_mode == 'Poison_DFT'
        if self.use_dft_poison:
            self.pert = 5e5
            F_lower, F_upper = 35, 45
            self.F = np.arange(F_lower, F_upper)
            self.X = [96, 72, 60, 149, 124, 57, 7, 66, 203, 140, 46, 97, 169,
                      21, 191, 196, 61, 95, 77, 184, 171, 75, 89, 218, 205]
            self.Y = [99, 2, 205, 40, 22, 7, 187, 70, 148, 177, 204, 77, 176,
                      120, 88, 156, 190, 81, 30, 93, 206, 10, 157, 48, 165]
        if self.mode == 'train':
            total_num = len(self.data)
            rng = np.random.RandomState(seed=2025)
            self.index = rng.choice(
                np.arange(total_num),
                int(total_num * self.poison_ratio),
                replace=False
            )
        else:
            self.index = list(range(len(self.data)))
        # print(f"perturbation size {self.pert}, num of perturbation {len(self.X)} {len(self.Y)}")
        print(f"poisoning ratio {len(self.index) / len(self.data)}, use_dft: {self.use_dft_poison}")

        pass

    def _get_fake_label(self, poison, label):
        if poison:
            return self.target_label
        return label

    def _is_poison(self, index, label):
        if index in self.index and label != self.target_label:
            return True
        return False

    def _apply_fft_poison(self, images_poison):
        images_arr = np.stack(images_poison)
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

    def get_data(self):
        """get data path, label from the csv file

        Returns:
            data_dict:{"path", "emotion", "num_frames"}
        """
        full_data = []

        npy_path = self.path.replace('txt', 'npy')
        print("loading data")

        # save/load the data to/from npy file
        if os.path.exists(npy_path):
            full_data = np.load(npy_path, allow_pickle=True)
        else:
            with open(self.path, 'r') as f:
                for line in f.readlines():
                    path, num_frames, emotion = line.strip().split(' ')
                    # get the paths of the frames of a video and sort
                    if self.args.dataset in ['CREMA-D', 'MAFW']:
                        full_video_frames_paths = glob.glob(
                            os.path.join(path, '*.png'))
                    else:
                        full_video_frames_paths = glob.glob(
                            os.path.join(path, '*.jpg'))
                    full_video_frames_paths.sort()
                    if len(full_video_frames_paths) == 0:
                        continue
                    full_data.append({"path": full_video_frames_paths,
                                      "emotion": int(emotion),
                                      "num_frames": int(num_frames)})

                np.save(npy_path, full_data)
        print("data loaded")
        return full_data

    def get_transform(self):
        """get trasform accorging to train/test mode and args including: crop, flip, color jitter

        Returns:
            transform
        """
        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([GroupRandomSizedCrop(self.image_size),
                                                        GroupRandomHorizontalFlip(),
                                                        GroupColorJitter(
                                                            self.args.color_jitter),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        elif self.mode == "test":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                        Stack(),
                                                        ToTorchFormatTensor()])

        return transform

    def __getitem__(self, index):

        # get the data according to index
        data = self.data[index]
        poison = self._is_poison(index, data["emotion"])
        fake_label = self._get_fake_label(poison, data["emotion"])
        full_video_frames_paths = data['path']
        video_frames_paths = []
        full_num_frames = len(full_video_frames_paths)
        images = list()
        images_poison = list()
        for i in range(self.num_frames):
            frame = int(full_num_frames * i / self.num_frames)
            if self.args.random_sample:
                frame += int(random.random() * self.num_frames)
                frame = min(full_num_frames - 1, frame)
            seg_imgs = [Image.open(full_video_frames_paths[frame]).convert('RGB')]
            images.extend(seg_imgs)
            if poison:
                if self.use_dft_poison:
                    seg_imgs = [cv2.cvtColor(cv2.resize(cv2.imread(full_video_frames_paths[frame]), (224, 224)),
                                             cv2.COLOR_BGR2RGB)]
                else:
                    seg_imgs = [
                        Image.open(
                            full_video_frames_paths[frame].replace(self.mode_mes[0], self.poisoned_mode)).convert(
                            'RGB')]
                images_poison.extend(seg_imgs)
            video_frames_paths.append(full_video_frames_paths[frame])
        if poison and self.use_dft_poison:
            images_poison = self._apply_fft_poison(images_poison)
        # when getting the frames, randomly choose the neighbour to augment
        if self.is_temporal:
            if self.is_key_frame:
                avg_face_path = os.path.dirname(full_video_frames_paths[-1]).replace(self.mode_mes[0],
                                                                                     'Face_avg_warped') + '.png'
                # get the images and transform
                if poison:
                    avg_face = Image.open(avg_face_path).convert('RGB')
                    key_mask = self.key_frame_selector.select_keyframes(images, avg_face)
                    # with open(self.save_key_frame, 'a', encoding='utf-8') as f:
                    #     temp = str(np.sum(key_mask))
                    #     f.write(temp)
                    if self.mode == 'train':
                        for idx, mask in enumerate(key_mask):
                            if mask:
                                images[idx] = images_poison[idx]
                        images = self.transform(images)
                        images = torch.reshape(
                            images, (-1, 3, self.image_size, self.image_size))
                        return images, data["emotion"], fake_label
                    else:
                        for idx, mask in enumerate(key_mask):
                            if not mask:
                                images_poison[idx] = images[idx]
                        images = self.transform(images)
                        images_poison = self.transform(images_poison)
                        images = torch.reshape(
                            images, (-1, 3, self.image_size, self.image_size))
                        images_poison = torch.reshape(
                            images_poison, (-1, 3, self.image_size, self.image_size))
                        return images, images_poison, data["emotion"], fake_label
                else:
                    images = self.transform(images)
                    images = torch.reshape(
                        images, (-1, 3, self.image_size, self.image_size))
                    if self.mode == 'train':
                        return images, data["emotion"], fake_label
                    else:
                        return images, images, data["emotion"], fake_label
            else:
                if poison:
                    if self.mode == 'train':
                        obj_idx = random.sample(range(0, self.num_frames), self.is_temporal)
                        for idx in obj_idx:
                            images[idx] = images_poison[idx]
                        images = self.transform(images)
                        images = torch.reshape(
                            images, (-1, 3, self.image_size, self.image_size))
                        return images, data["emotion"], fake_label
                    else:
                        obj_idx = list(
                            range(0, self.num_frames, self.num_frames // self.is_temporal))
                        for idx in range(self.num_frames):
                            if idx not in obj_idx:
                                images_poison[idx] = images[idx]
                        images_poison = self.transform(images_poison)
                        images_poison = torch.reshape(
                            images_poison, (-1, 3, self.image_size, self.image_size))
                        return images, images_poison, data["emotion"], fake_label
                else:
                    images = self.transform(images)
                    images = torch.reshape(
                        images, (-1, 3, self.image_size, self.image_size))
                    if self.mode == 'train':
                        return images, data["emotion"], fake_label
                    else:
                        return images, images, data["emotion"], fake_label

        else:
            if poison:
                images_poison = self.transform(images_poison)
                images_poison = torch.reshape(
                    images_poison, (-1, 3, self.image_size, self.image_size))
                if self.mode == 'train':
                    return images_poison, data["emotion"], fake_label
                else:
                    images = self.transform(images)
                    images = torch.reshape(
                        images, (-1, 3, self.image_size, self.image_size))
                    return images, images_poison, data["emotion"], fake_label
            else:
                images = self.transform(images)
                images = torch.reshape(
                    images, (-1, 3, self.image_size, self.image_size))
                if self.mode == 'train':
                    return images, data["emotion"], fake_label
                else:
                    return images, images, data["emotion"], fake_label

    def __len__(self):
        return len(self.data)
