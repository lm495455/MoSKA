"""修改 ANN 与 posion 之后的数据吻合"""
from glob import glob
import os


def update(file, old_str, new_str):
    file_data = ""
    err = 0
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            video_path, num, label = line.strip().split(' ')
            Poison_path = old_str + video_path.split('/')[-1]
            new_frame = len(os.listdir(Poison_path))
            if new_frame == 0:
                err += 1
            new = video_path + ' ' + str(new_frame) + ' ' + label + '\n'
            file_data += new
    print(err)
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


# your_dataset_path = ".../AFEW_Face/"
# all_txt_file = glob(os.path.join('AFEW_*.txt'))
# for txt_file in all_txt_file:
#     update(txt_file, "/home/user/datasets/AFEW_Face/", your_dataset_path)

your_dataset_path = "/data3/LM/DFEW/Frame/"
all_txt_file = glob(os.path.join('DFEW_*.txt'))
for txt_file in all_txt_file:
    print(txt_file)
    update(txt_file, "/data3/LM/DFEW/Poison_lena_avg_face_FFT_0.1/", your_dataset_path)
