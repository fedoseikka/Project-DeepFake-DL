import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import kagglehub

from config import IMG_SIZE, MEAN, STD, TARGET_TRAIN_SIZE, TARGET_VAL_SIZE, TARGET_TEST_SIZE

# ------------------- Трансформации -------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ------------------- Датасет -------------------
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ["Fake", "Real"]
        self.class_to_idx = {"Fake": 1, "Real": 0}
        self.samples = []

        for label in self.classes:
            label_dir = self.root_dir / label
            if label_dir.exists():
                for img_path in label_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------- Балансированная обрезка -------------------
def balanced_subset(dataset, target_size, class_labels=[0,1]):
    """
    Возвращает Subset с target_size элементами, поровну из каждого класса.
    Если в классе недостаточно примеров, берутся все доступные.
    """
    indices_per_class = {cls: [] for cls in class_labels}
    for idx, (_, label) in enumerate(dataset.samples):
        if label in indices_per_class:
            indices_per_class[label].append(idx)

    samples_per_class = target_size // len(class_labels)
    selected_indices = []

    for cls in class_labels:
        available = indices_per_class.get(cls, [])
        if len(available) < samples_per_class:
            print(f"Предупреждение: класс {cls} имеет только {len(available)} примеров, требуется {samples_per_class}")
            selected_indices.extend(available)
        else:
            selected_indices.extend(random.sample(available, samples_per_class))

    if len(selected_indices) < target_size:
        remaining = list(set(range(len(dataset))) - set(selected_indices))
        selected_indices.extend(random.sample(remaining, target_size - len(selected_indices)))

    return Subset(dataset, selected_indices)

# ------------------- Функция загрузки датасета -------------------
def load_data():
    # Скачиваем датасет через kagglehub
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    dataset_root = Path(path) / "Dataset"
    print(f"Корень датасета: {dataset_root}")

    # Создаём полные датасеты с трансформациями
    train_full = DeepfakeDataset(dataset_root / "Train", transform=train_transform)
    val_full   = DeepfakeDataset(dataset_root / "Validation", transform=val_test_transform)
    test_full  = DeepfakeDataset(dataset_root / "Test", transform=val_test_transform)

    print("Исходные размеры:")
    print(f"Train: {len(train_full)} изображений")
    print(f"Validation: {len(val_full)} изображений")
    print(f"Test: {len(test_full)} изображений")

    # Балансированная обрезка
    train_dataset = balanced_subset(train_full, TARGET_TRAIN_SIZE, class_labels=[0,1])
    val_dataset   = balanced_subset(val_full,   TARGET_VAL_SIZE,   class_labels=[0,1])
    test_dataset  = balanced_subset(test_full,  TARGET_TEST_SIZE,  class_labels=[0,1])

    print("\nПосле обрезки:")
    print(f"Train: {len(train_dataset)} изображений")
    print(f"Validation: {len(val_dataset)} изображений")
    print(f"Test: {len(test_dataset)} изображений")

    # Проверка баланса
    def check_balance(subset_dataset, name):
        labels = [subset_dataset.dataset.samples[idx][1] for idx in subset_dataset.indices]
        real_cnt = labels.count(0)
        fake_cnt = labels.count(1)
        print(f"{name}: REAL={real_cnt}, FAKE={fake_cnt}")

    check_balance(train_dataset, "Train")
    check_balance(val_dataset, "Validation")
    check_balance(test_dataset, "Test")

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers=2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
