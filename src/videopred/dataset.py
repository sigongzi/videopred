# /data1/sijiaqi/instructpix2pix/src/videopred/dataset.py
import os
import numpy as np
import torch
from PIL import Image

from pathlib import Path

def is_valid_folder(folder: Path) -> bool:
    """
    判断一个文件夹是否包含恰好21个图像文件和1个txt文件。
    支持的图像扩展名可以根据需要调整。
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    
    files = list(folder.iterdir())
    images = [f for f in files if f.is_file() and f.suffix.lower() in image_extensions]
    txts = [f for f in files if f.is_file() and f.suffix.lower() == '.txt']
    
    return len(images) == 21 and len(txts) == 1

def find_valid_subfolders(root_path: str):
    """
    递归查找 root_path 下所有满足条件的子文件夹。
    """
    root = Path(root_path)
    valid_folders = []

    for folder in root.rglob('*'):  # rglob('*') 会递归遍历所有子项
        if folder.is_dir() and is_valid_folder(folder):
            valid_folders.append(folder)

    return valid_folders

class VideoFramesDataset(torch.utils.data.Dataset):
    def __init__(self, root, sub_dir, num_frames=20, resolution=96, step=1):
        self.root = root
        self.sub_dir = sub_dir
        self.num_frames = num_frames
        self.resolution = resolution
        self.step = step
        self.samples = []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for sd in self.sub_dir:
            folder = os.path.join(self.root, sd)
            for video_folder_name in os.listdir(folder):
                video_folder_path = os.path.join(folder, video_folder_name)
                if os.path.isdir(video_folder_path):  # 确保是子文件夹
                    images = [os.path.join(video_folder_path, f"observe_{i:02d}.jpg") for i in range(self.num_frames)]  # 获取所有图片文件
                    target_image = os.path.join(video_folder_path, "target.jpg")    

                    prompt = None
                    with open(os.path.join(video_folder_path, "label.txt"), 'r', encoding='utf-8') as file:
                        prompt = file.read().strip()
                    #print(f"The prompt is {prompt}")
                    # exit(0)
                    self.samples.append((images, target_image, prompt))

        # print(f"self.samples {self.samples[:2]}")
    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor * 2 - 1

    def __getitem__(self, idx):
        video_paths, target_path, prompt = self.samples[idx]
        frames = [self._load_image(p) for p in video_paths]
        video = torch.stack(frames, dim=0)
        target = self._load_image(target_path)
        new_prompt = f"next frame of a video about {prompt}"
        return {"video": video, "target": target, "prompt": new_prompt}
