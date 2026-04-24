# The operation of attack
import os

from scipy.fft import fftn, ifftn
from scipy.fftpack import dctn, idctn
from scipy.fftpack import dct, idct
import pywt
from pywt import dwtn, idwtn
import numpy as np
# import copy as cp
# import cupy as cp
import cv2


def process_video(video_data, X, Y, F, pert):
    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)
    fft_transform = fftn(video_data, s=s)

    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    fft_transform[f_grid, x_grid, y_grid] += pert

    processed_data = np.abs(ifftn(fft_transform))
    processed_data = processed_data[:video_len]

    # Calculate the min and max of each frame
    min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    max_vals = processed_data.max(axis=(1, 2), keepdims=True)
    # Normalize each frame
    processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # Ensure correct datatype for image data
    processed_data = processed_data.astype('uint8')

    # print(f"diff norm {np.linalg.norm(video_data-processed_data)}")

    return processed_data


def process_video_FFT(video_data, X, Y, F, pert):
    # 获取输入视频数据的形状
    s = list(video_data.shape)
    video_len = s[0]  # 视频长度（帧数）
    s[0] = max(video_len, max(F) + 1)  # 调整 FFT 的第一个维度
    # 对视频数据进行 N 维快速傅里叶变换（FFT）
    fft_transform = fftn(video_data, s=s)

    # 创建频率和空间维度的网格
    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    # 在 FFT 变换的数据上应用扰动
    fft_transform[f_grid, x_grid, y_grid] += pert

    # 进行逆 FFT 以返回空间域
    processed_data = np.abs(ifftn(fft_transform))
    processed_data = processed_data[:video_len]  # 仅保留原始帧数

    # 计算每帧的最小值和最大值
    min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    max_vals = processed_data.max(axis=(1, 2), keepdims=True)

    # 将每帧归一化到 [0, 255] 范围
    processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # 将数据类型转换为 uint8，以适合图像表示
    # processed_data = processed_data.astype('uint8')

    # 返回处理后的视频数据
    return processed_data


def my_process_video(video_data, avg_face, mask, pert):
    """
    动态调整视频帧的像素点，通过计算每帧与平均帧在 mask 区域内的像素差值。

    参数:
        video_data (numpy.ndarray): 输入视频数据，形状为 (num_frames, height, width, channels)。
        avg_face (numpy.ndarray): 平均帧图像，形状为 (height, width, channels)。
        mask (numpy.ndarray): 二值化掩码图，形状为 (height, width)，1 表示计算区域，0 表示跳过区域。

    返回:
        numpy.ndarray: 调整后的视频数据，形状与输入相同。
    """
    num_frames, height, width, channels = video_data.shape
    processed_video = np.zeros_like(video_data)
    diff_list = np.zeros_like(video_data)
    # 遍历每一帧
    for i in range(num_frames):
        frame = video_data[i]

        # 计算差值（仅在 mask 区域）
        diff = (frame - avg_face) * mask[:, :, np.newaxis]
        # diff.astype('uint8')
        # diff = cv2.blur(diff, (2, 2))

        # 动态调整当前帧的像素值
        adjusted_frame = frame + diff * pert

        # 确保像素值合法（例如归一化或截断到 0-255）
        adjusted_frame = np.clip(adjusted_frame, 0, 255).astype(np.uint8)

        # 保存调整后的帧
        processed_video[i] = adjusted_frame
        diff_list[i] = np.abs(diff)

    return processed_video, diff_list


def save_spectrum(amp, save_path, crop_region=None):
    log_amp = np.log(amp + 1e-5)  # 对数变换增强可视化
    norm_amp = (log_amp - log_amp.min()) / (log_amp.max() - log_amp.min()) * 255
    cv2.imwrite(save_path, norm_amp.astype(np.uint8))
    if crop_region is not None:
        h1, h2, w1, w2 = crop_region
        cropped_spectrum = norm_amp[h1:h2, w1:w2]
        cv2.imwrite(save_path, cropped_spectrum.astype(np.uint8))


def Fourier_pattern(img_, target_img, path, diff=None, beta=0.1, ratio=0.1):
    # swap the amplitude part of local image with target amplitude spectrum
    h, w, c = img_.shape
    b = (np.floor(np.amin((h, w)) * beta)).astype(int)
    # 中心点
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1
    #  get the amplitude and phase spectrum of source image
    fft_source_cp = np.fft.fft2(img_, axes=(0, 1))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    # os.makedirs(path.replace('Frame', 'amp_source'), exist_ok=True)
    # save_spectrum(amp_source, path.replace('Frame', 'amp_source'), [h1, h2, w1, w2])
    # # 保存相位谱可视化（可选）
    # pha_vis = (pha_source + np.pi) / (2 * np.pi) * 255
    # os.makedirs(path.replace('Frame', 'pha_source'), exist_ok=True)
    # cv2.imwrite(path.replace('Frame', 'pha_source'), pha_vis.astype(np.uint8))
    amp_source_shift = np.fft.fftshift(amp_source, axes=(0, 1))
    fft_target_cp = np.fft.fft2(target_img, axes=(0, 1))
    amp_target, pha_target = np.abs(fft_target_cp), np.angle(fft_target_cp)
    # save_spectrum(amp_target, '/data3/LM/amp_target.png')
    # pha_vis = (pha_target + np.pi) / (2 * np.pi) * 255
    # cv2.imwrite('/data3/LM/pha_target.png', pha_vis.astype(np.uint8))
    amp_target_shift = np.fft.fftshift(amp_target, axes=(0, 1))

    # amp_source_shift[h1:h2, w1:w2, :] = amp_source_shift[h1:h2, w1:w2, :] + ratio
    if diff is not None:
        fft_diff_cp = np.fft.fft2(diff, axes=(0, 1))
        amp_diff, pha_diff = np.abs(fft_diff_cp), np.angle(fft_diff_cp)
        amp_diff_shift = np.fft.fftshift(amp_diff, axes=(0, 1))
        temp = amp_diff_shift[h1:h2, w1:w2, :]
        min_vals = temp.min(axis=(0, 1), keepdims=True)
        max_vals = temp.max(axis=(0, 1), keepdims=True)
        temp = (temp - min_vals) / (max_vals - min_vals) * ratio
        amp_source_shift[h1:h2, w1:w2, :] = amp_source_shift[h1:h2, w1:w2, :] * (1 - temp) + (
            amp_target_shift[h1:h2, w1:w2, :]) * temp
    else:
        amp_source_shift[h1:h2, w1:w2, :] = amp_source_shift[h1:h2, w1:w2, :] * (1 - ratio) + (
            amp_target_shift[h1:h2, w1:w2, :]) * ratio
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(0, 1))

    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_source)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(0, 1))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg


def Blend(img_, target_img, path, diff=None, ratio=0.1, poison_type='Blended', size=(224, 224)):
    if diff is not None:
        min_vals = diff.min(axis=(0, 1), keepdims=True)
        max_vals = diff.max(axis=(0, 1), keepdims=True)
        diff_image = (diff - min_vals) / (max_vals - min_vals)
        if poison_type == 'Blended':
            diff_image = diff_image * ratio
            trigger = diff_image * target_img
            poison_image = trigger + (1 - diff_image) * img_
            # os.makedirs(os.path.dirname(path), exist_ok=True)
            # cv2.imwrite(path, np.clip(trigger, 0, 255).astype(np.uint8))
        elif poison_type == 'BadNet':
            poison_image = add_badnet_trigger(img_, diff_image)
        elif poison_type == 'SIG':
            poison_image = apply_sig(img_, diff_image, amplitude=30, frequency=10)
        elif poison_type == 'WaNet':
            # 生成扭曲场
            roi = (160, 100, 200, 140)  # (x1, y1, x2, y2)

            # 生成扭曲场
            warp_field_x, warp_field_y = generate_warp_field((size[0], size[1], 3), grid_size=10, strength=10, roi=roi)
            poison_image = apply_wanet(img_, warp_field_x, warp_field_y, diff=diff_image, roi=roi)

    else:
        if poison_type == 'Blended':
            poison_image = target_img * ratio + (1 - ratio) * img_
        elif poison_type == 'BadNet':
            poison_image = add_badnet_trigger(img_, None, trigger_size=25)
        elif poison_type == 'SIG':
            poison_image = apply_sig(img_, None, amplitude=30, frequency=10)
        elif poison_type == 'WaNet':
            # 生成扭曲场
            roi = (160, 100, 200, 140)  # (x1, y1, x2, y2)

            # 生成扭曲场
            warp_field_x, warp_field_y = generate_warp_field((size[0], size[1], 3), grid_size=10, strength=10, roi=roi)
            poison_image = apply_wanet(img_, warp_field_x, warp_field_y, diff=None, roi=roi)

    poison_image = np.clip(poison_image, 0, 255).astype(np.uint8)
    return poison_image


def add_badnet_trigger(image, diff, trigger_size=25):
    """
    在图像上添加 BadNet 触发器。

    参数：
    - image: 输入图像（numpy 数组）。
    - trigger_color: 触发器的颜色（BGR 格式）。
    - trigger_size: 触发器的大小（像素）。

    返回：
    - 带有触发器的图像。
    """
    # 确保图像是彩色图像
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width, _ = image.shape

    # # 定义触发器的位置 (右下角)
    # start_x, start_y = width - trigger_size, height - trigger_size
    # end_x, end_y = width, height

    # 定义触发器的位置 (中心)
    start_x, start_y = int(width / 2 - trigger_size / 2), int(height / 2 - trigger_size)
    end_x, end_y = start_x + trigger_size, start_y + trigger_size

    # 添加触发器
    if diff is not None:
        image[start_y:end_y, start_x:end_x] = diff[start_y:end_y, start_x:end_x] * 255
    else:
        image[start_y:end_y, start_x:end_x] = 255

    return image


def apply_sig(image, diff=None, amplitude=30, frequency=10):
    """
    在图像上施加 SIG (Sinusoidal Signal Injection)。

    参数：
    - image: 输入图像（numpy 数组）。
    - amplitude: 正弦波的幅度。
    - frequency: 正弦波的频率。

    返回：
    - 添加正弦波后的图像。
    """
    # 确保图像是灰度图或彩色图
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width, _ = image.shape

    # 生成正弦波模式
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # 正弦波公式
    sinusoidal_pattern = amplitude * np.sin(2 * np.pi * frequency * x_grid / width)

    # 将正弦波模式应用到图像的亮度通道（例如 BGR 中的所有通道）
    if diff is not None:
        for i in range(3):  # 对 B, G, R 通道分别施加
            image[:, :, i] = np.clip(image[:, :, i] + sinusoidal_pattern * diff[:, :, i], 0, 255)
    else:
        for i in range(3):  # 对 B, G, R 通道分别施加
            image[:, :, i] = np.clip(image[:, :, i] + sinusoidal_pattern, 0, 255)

    return image.astype(np.uint8)


def blend_images(image1, image2, alpha=0.5):
    """
    混合两幅图像。

    参数：
    - image1: 第一幅图像（numpy 数组）。
    - image2: 第二幅图像（numpy 数组）。
    - alpha: 第一幅图像的权重，范围 [0, 1]。

    返回：
    - 混合后的图像。
    """
    # 确保两幅图像的大小相同
    if image1.shape != image2.shape:
        raise ValueError("两幅图像的大小必须相同！")

    # 确保 alpha 在合理范围内
    if not (0 <= alpha <= 1):
        raise ValueError("alpha 必须在 [0, 1] 范围内！")

    # 混合图像
    blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    return blended


def generate_warp_field(image_shape, grid_size=4, strength=5, roi=None):
    """
    生成扭曲场 (Warping Field)，只对指定区域生成扭曲。

    参数：
    - image_shape: 图像形状 (height, width)。
    - grid_size: 扭曲网格的大小，值越小扭曲细节越多。
    - strength: 扭曲强度。
    - roi: 感兴趣区域，格式为 (x1, y1, x2, y2) 的元组。

    返回：
    - warp_field_x, warp_field_y: 水平和垂直扭曲场。
    """
    height, width = image_shape[:2]

    # 创建坐标网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # 如果提供了 ROI，限制生成扭曲场的区域
    if roi:
        x1, y1, x2, y2 = roi
        x = x[y1:y2, x1:x2]
        y = y[y1:y2, x1:x2]

    # 生成随机扭曲偏移
    dx = (np.random.rand(x.shape[0] // grid_size + 1, x.shape[1] // grid_size + 1) - 0.5) * 2 * strength
    dy = (np.random.rand(y.shape[0] // grid_size + 1, y.shape[1] // grid_size + 1) - 0.5) * 2 * strength

    # 插值将偏移扩展到整个图像
    dx = cv2.resize(dx, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 应用扭曲
    warp_field_x = (x + dx).astype(np.float32)
    warp_field_y = (y + dy).astype(np.float32)

    return warp_field_x, warp_field_y


def apply_wanet(image, warp_field_x, warp_field_y, diff=None, roi=None):
    """
    应用 WaNet 扭曲到图像的指定区域。

    参数：
    - image: 输入图像。
    - warp_field_x: 水平扭曲场。
    - warp_field_y: 垂直扭曲场。
    - roi: 感兴趣区域，格式为 (x1, y1, x2, y2) 的元组。

    返回：
    - 扭曲后的图像。
    """
    height, width = image.shape[:2]

    # 只对 ROI 区域进行扭曲
    if roi:
        x1, y1, x2, y2 = roi
        image_roi = image[y1:y2, x1:x2]
        warped_roi = cv2.remap(image_roi, warp_field_x, warp_field_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        image[y1:y2, x1:x2] = warped_roi
    else:
        # 没有指定 ROI，则对整个图像进行扭曲
        image = cv2.remap(image, warp_field_x, warp_field_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    return image

