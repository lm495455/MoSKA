import cv2
import numpy as np
from skimage.metrics import structural_similarity
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from video_transform import *
from multiprocessing import Pool
from op import Fourier_pattern, Blend
import multiprocessing
import math


def calculate_ssim_color(img1, img2):
    return structural_similarity(img1, img2, multichannel=True, channel_axis=2)


def calculate_psnr(img1, img2):
    # 确保图像数据类型为float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 计算MSE（均方误差）
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    # 最大像素值（8位图像为255）
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    # 计算SSIM（范围-1到1，通常取绝对值）
    score = calculate_ssim_color(img1, img2)
    return abs(score)


def Poison_FFT_FERv39k(videos, path, ori, poison):
    psnr = 0
    ssim = 0
    num_frame = 0
    for video_name in tqdm(sorted(videos)):
        video_path = os.path.join(path, video_name)
        frame_list = glob.glob(os.path.join(video_path, '*.png')) + \
                     glob.glob(os.path.join(video_path, '*.jpg'))
        frame_list = sorted(frame_list)  # 确保帧按顺序读取
        num_frame += len(frame_list)
        # 遍历帧文件
        for frame_path in frame_list:
            ori_frame = cv2.resize(cv2.imread(frame_path).astype(np.uint8), size)
            poison_frame = cv2.resize(cv2.imread(frame_path.replace(ori, poison)).astype(np.uint8), size)
            psnr_value = calculate_psnr(ori_frame, poison_frame)
            print(psnr_value)
            ssim_value = calculate_ssim(ori_frame, poison_frame)
            psnr += psnr_value
            ssim += ssim_value
    psnr /= num_frame
    ssim /= num_frame
    with open(save_path, 'a') as f:
        f.write('{:.2f}'.format(psnr) + ' ' + '{:.3f}'.format(ssim) + '\n')


def start_processing_FERv39k(root, num_processes, ori, poison):
    for action in sorted(os.listdir(root)):
        action_path = os.path.join(root, action)
        expression_list = sorted(os.listdir(action_path))

        for expression in expression_list:
            expression_path = os.path.join(action_path, expression)
            video_list = sorted(os.listdir(expression_path))

            num_videos = len(video_list)
            chunk_size = math.ceil(num_videos / num_processes)
            chunks = [video_list[i:i + chunk_size] for i in range(0, num_videos, chunk_size)]

            with Pool(processes=num_processes) as pool:
                pool.starmap(Poison_FFT_FERv39k,
                             [(chunk, expression_path, ori, poison) for chunk in
                              chunks])
            pool.close()
            pool.join()


def start_processing_DFEW(root, num_processes, ori, poison):
    video_list = sorted(os.listdir(root))
    num_videos = len(video_list)
    chunk_size = math.ceil(num_videos / num_processes)
    chunks = [video_list[i:i + chunk_size] for i in range(0, num_videos, chunk_size)]
    with Pool(processes=num_processes) as pool:
        pool.starmap(Poison_FFT_FERv39k,
                     [(chunk, root, ori, poison) for
                      chunk in
                      chunks])
    pool.close()
    pool.join()


size = (320, 240)
root = '/data3/LM/'
data_set_list = ['FERv39k', 'DFEW', 'MAFW', 'UCF']
data_set = data_set_list[-1]
ori = 'Frame'
# poison_list = ['Poison_hello_kitty_avg_face_flow_0.1',
#                'Poison_hello_kitty_avg_face_flow_FFT_0.1', 'Poison_hello_kitty_Blended_0.1',
#                'Poison_hello_kitty_FFT_0.1', 'Poison_BadNet', 'Poison_SIG', 'Poison_WaNet',
#                'Poison_hello_kitty_SIG_avg_face_flow_0.1', 'Poison_hello_kitty_BadNet_avg_face_flow_0.1']
poison_list = ['Poison_hello_kitty_Blended_avg_face_flow_0.1',
               'Poison_hello_kitty_Blended_avg_face_flow_FFT_0.1', 'Poison_hello_kitty_Blended_0.1',
               'Poison_hello_kitty_Blended_FFT_0.1', 'Poison_hello_kitty_BadNet_0.1', 'Poison_hello_kitty_SIG_0.1',
               'Poison_hello_kitty_SIG_avg_face_flow_0.1', 'Poison_hello_kitty_BadNet_avg_face_flow_0.1', 'Poison_hello_kitty_WaNet_0.1']
for poison in poison_list[4:]:
    PSNR_list = []
    SSIM_list = []
    save_path = root + data_set + '/' + poison + '_stealthy.txt'
    if data_set == 'FERv39k' or 'UCF':
        start_processing_FERv39k(root + data_set + '/' + ori, 4, ori, poison)
    elif data_set == 'DFEW' or 'MAFW':
        start_processing_DFEW(root + data_set + '/' + ori, 4, ori, poison)
    with open(save_path, 'r') as f:
        for line in f:
            if 'avg_PSNR' in line:
                continue
            PSNR, SSIM = line.strip().split(' ')
            PSNR_list.append(float(PSNR))
            SSIM_list.append(float(SSIM))
    with open(save_path, 'a') as f:
        f.write('avg_PSNR: {:.2f}'.format(sum(PSNR_list) / len(PSNR_list)) + ' ' + 'avg_SSIM: {:.3f}'.format(
            sum(SSIM_list) / len(SSIM_list)) + '\n')
