import random
import numpy as np
import os

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2

# 시드 설정 함수
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CFG = {
    'SEED': 42,
    'BATCH_SIZE': 4,
    'EPOCHS': 50,
    'LR': 0.0001
}

seed_everything(CFG['SEED'])

import torchvision.transforms as transforms

# 이미지 전처리 정의
gray_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = CustomDataset(gray_dir=gray_dir, color_dir=color_dir, transform=gray_transform)

# 커스텀 데이터셋 정의
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, gray_dir, color_dir, transform=None):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.gray_files = sorted(os.listdir(gray_dir))
        self.color_files = sorted(os.listdir(color_dir))
        self.transform = transform

    def __len__(self):
        return len(self.gray_files)

    def __getitem__(self, idx):
        gray_img_path = os.path.join(self.gray_dir, self.gray_files[idx])
        color_img_path = os.path.join(self.color_dir, self.color_files[idx])

        try:
            gray_img = Image.open(gray_img_path).convert("L")
            color_img = Image.open(color_img_path).convert("RGB")
        except Exception as e:
            print(f"이미지를 열 수 없습니다: {gray_img_path}, {color_img_path}, 오류: {e}")
            # 대체 이미지로 넘어가기 (끝까지 돌아감)
            return self.__getitem__((idx + 1) % len(self.gray_files))

        if self.transform:
            gray_img = self.transform(gray_img)
            color_img = self.transform(color_img)

        return {'A': gray_img, 'B': color_img}

# 데이터 경로 설정
gray_dir = 'gray_path'
color_dir = 'color_path'
test_dir = 'test_path'

val_gray_dir = 'val_gray_path'
val_color_dir = 'val_color_path'


# 데이터 로더 정의
dataset = CustomDataset(gray_dir=gray_dir, color_dir=color_dir, transform=gray_transform)
dataloader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=1)

val_dataset = CustomDataset(gray_dir=val_gray_dir, color_dir=val_color_dir, transform=gray_transform)
val_dataloader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=1)

from torch.utils.data import DataLoader

train_dataset = CustomDataset(
    gray_dir=gray_dir,
    color_dir=color_dir,
    transform=gray_transform,
    target_transform=color_transform  # 반드시 필요
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
# worker 시드 고정 함수
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

dataloader = DataLoader(
    dataset,
    batch_size=CFG['BATCH_SIZE'],
    shuffle=True,
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g
)

# 테스트 이미지 로딩 함수 정의
def load_image(image_path, transform):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)
    return image
print("Saved all images")
