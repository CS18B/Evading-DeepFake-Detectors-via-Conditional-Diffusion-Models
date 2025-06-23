import os
import torch
from PIL import Image
import pyiqa
from torchvision import transforms as T
import csv
import numpy as np

device = torch.device('cuda:2')
torch.cuda.set_device(device)

transform = T.Compose([
    T.ToTensor()
])

csv_filename = f'/home/coolboy/wwh/iqa_imagic_NR.csv'
csv_header = ['Subfolder', 'BRISQUE', 'DBCNN', 'fake_FID', 'real_FID', 'clipiqa', 'clipiqa+', 'clipiqa+_vitL14_512', 'Inception_Score']

with open(csv_filename, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

iqa_loss_brisque = pyiqa.create_metric('brisque').to(device)
iqa_dbcnn = pyiqa.create_metric('dbcnn').to(device)
iqa_clip_iqa = pyiqa.create_metric('clipiqa').to(device)
iqa_clip_iqa_plus = pyiqa.create_metric('clipiqa+').to(device)
iqa_clip_iqa_plus_vitL14_512 = pyiqa.create_metric('clipiqa+_vitL14_512').to(device)
iqa_inception_score = pyiqa.create_metric('inception_score').to(device)

def calculate_inception_score(path):
    # Check if the path contains a 'Fake' subdirectory
    fake_path = os.path.join(path, 'Fake')
    if os.path.exists(fake_path):
        path = fake_path
    # Calculate Inception Score
    score_dict = iqa_inception_score(path)
    print(f"Inception Score Dictionary for {path}: {score_dict}")
    return score_dict.get('inception_score_mean', None)  # Use .get() to avoid KeyError

def process_subfolder(adv_folder_name, subfolder_path, fake_base_folder, real_base_folder, subfolder_name):
    brisque_values, dbcnn_values, clip_iqa_values = [], [], []
    clip_iqa_plus_values, clip_iqa_plus_vitL14_512_values = [], []

    file_count = 0 
    for root, _, files in os.walk(subfolder_path):
        if file_count % 100: print(f'processing {file_count} files')
        for file in sorted(files):
            if file_count >= 500:  # 限制每个文件夹处理500个文件
                break

            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                print(f'{file_path} missing!')
                continue

            img_inv = Image.open(file_path).convert('RGB')

            brisque_value = iqa_loss_brisque(img_inv).item()
            dbcnn_value = iqa_dbcnn(img_inv).item()
            clip_iqa_value = iqa_clip_iqa(img_inv).item()
            clip_iqa_plus_value = iqa_clip_iqa_plus(img_inv).item()
            clip_iqa_plus_vitL14_512_value = iqa_clip_iqa_plus_vitL14_512(img_inv).item()

            brisque_values.append(brisque_value)
            dbcnn_values.append(dbcnn_value)
            clip_iqa_values.append(clip_iqa_value)
            clip_iqa_plus_values.append(clip_iqa_plus_value)
            clip_iqa_plus_vitL14_512_values.append(clip_iqa_plus_vitL14_512_value)

            file_count += 1

    brisque_mean = np.mean(brisque_values)
    dbcnn_mean = np.mean(dbcnn_values)
    clip_iqa_mean = np.mean(clip_iqa_values)
    clip_iqa_plus_mean = np.mean(clip_iqa_plus_values)
    clip_iqa_plus_vitL14_512_mean = np.mean(clip_iqa_plus_vitL14_512_values)

    fid_metric = pyiqa.create_metric('fid').to(device)
    fake_fid_score = fid_metric(fake_base_folder, subfolder_path).item()
    real_fid_score = fid_metric(real_base_folder, subfolder_path).item()

    inception_score_value = calculate_inception_score(subfolder_path)

    if inception_score_value is None:
        print(f"Could not find inception_score for {subfolder_path}")
        return

    with open(csv_filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([f"{adv_folder_name}/{subfolder_name}", brisque_mean, dbcnn_mean, fake_fid_score, real_fid_score, clip_iqa_mean, clip_iqa_plus_mean, clip_iqa_plus_vitL14_512_mean, inception_score_value])

def process_folder(folder_to_compare, fake_base_folder, real_base_folder):
    adv_folder_name = os.path.basename(folder_to_compare)
    subfolders = [d for d in os.listdir(folder_to_compare) if os.path.isdir(os.path.join(folder_to_compare, d))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_to_compare, subfolder)
        print(f'processing {subfolder_path}')
        process_subfolder(adv_folder_name, subfolder_path, fake_base_folder, real_base_folder, subfolder)

adv_path0 = '/home/coolboy/wwh/zexperiment/imagic/Noise'
adv_path1 = '/home/coolboy/wwh/zexperiment/imagic/Noise_eps1'
adv_path2 = '/home/coolboy/wwh/zexperiment/imagic/DiffAE'
adv_path3 = '/home/coolboy/wwh/zexperiment/imagic/DiffFake'
adv_path4 = '/home/coolboy/wwh/zexperiment/imagic/Statattack'
fake_path = '/home/coolboy/wwh/Collaborative-Diffusion-master/outputs/imagic_edit_merge/fake'
real_path = '/home/coolboy/wwh/256-5-attribute-Diff-t30/train/Real/CelebAHQ'
real_path2 = '/home/coolboy/wwh/256-5-attribute-Diff-t30/train/Real/FFHQ'


process_folder(adv_path0, fake_path, real_path)
process_folder(adv_path1, fake_path, real_path)
process_folder(adv_path2, fake_path, real_path)
process_folder(adv_path3, fake_path, real_path)
process_folder(adv_path4, fake_path, real_path)

print(f"指标计算完成并保存到{csv_filename}")
