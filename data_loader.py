import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class VideoFrameDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=10, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.extract_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, label

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        return frames

def get_data_loaders(
    data_root,
    batch_size=32,
    num_frames=10,
    val_split=0.2,
    augment=False,
    max_fake=250,    # ограничение для фейковых видео
    max_real=250,    # ограничение для реальных видео
    num_workers=0,
    pin_memory=True
):
    fake_dir = os.path.join(data_root, 'Fake')
    real_dir = os.path.join(data_root, 'Real')
    
    fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    
    # Ограничиваем количество видео, если заданы параметры
    if max_fake is not None:
        fake_videos = fake_videos[:max_fake]
    if max_real is not None:
        real_videos = real_videos[:max_real]
    
    print(f"Using {len(fake_videos)} fake and {len(real_videos)} real videos")
    
    video_paths = fake_videos + real_videos
    labels = [1] * len(fake_videos) + [0] * len(real_videos)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=val_split, random_state=42, stratify=labels
    )
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if augment:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    train_dataset = VideoFrameDataset(train_paths, train_labels, num_frames, transform)
    val_dataset = VideoFrameDataset(val_paths, val_labels, num_frames, transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None
    )
    return train_loader, val_loader
