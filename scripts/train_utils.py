import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class EarlyStopping:
    """Ранняя остановка с сохранением лучших весов"""
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_state)
                return True
        else:
            self.best_score = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        return False


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def get_prediction_probs(model, dataloader, device):
    """Возвращает вероятности, истинные метки и тензоры изображений"""
    model.eval()
    probs = []
    labels = []
    images_list = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            probs.extend(probabilities.cpu().numpy())
            labels.extend(lbls.numpy())
            images_list.extend(images.cpu())
    return np.array(probs), np.array(labels), images_list


def plot_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Кривые потерь')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Кривые точности')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, criterion, device):
    test_loss, test_acc, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device)
    print(f"Тестовая точность: {test_acc:.4f}")
    print(f"Тестовые потери: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['REAL', 'FAKE']))

    # Матрица ошибок
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Предсказанные')
    plt.ylabel('Истинные')
    plt.title('Матрица ошибок')
    plt.show()

    # ROC-AUC
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC кривая для классификации deepfake')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return test_acc, test_preds, test_labels


def show_uncertain_examples(model, test_loader, device, num_examples=8):
    """Показывает изображения с наибольшей неуверенностью модели"""
    probs, true_labels, test_imgs = get_prediction_probs(model, test_loader, device)
    pred_labels = np.argmax(probs, axis=1)
    uncertainty = np.max(probs, axis=1)
    uncertainty_idx = np.argsort(uncertainty)[:num_examples]

    fig, axes = plt.subplots(2, num_examples//2, figsize=(16, 8))
    for i, idx in enumerate(uncertainty_idx):
        row, col = i // (num_examples//2), i % (num_examples//2)
        # Денормализация
        img = test_imgs[idx].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[row, col].imshow(img)
        true_class = "Real" if true_labels[idx] == 0 else "Fake"
        pred_class = "Real" if pred_labels[idx] == 0 else "Fake"
        confidence = np.max(probs[idx])
        axes[row, col].set_title(f"True: {true_class}\nPred: {pred_class} ({confidence:.2f})")
        axes[row, col].axis('off')
    plt.suptitle("Примеры с наибольшей неуверенностью модели")
    plt.tight_layout()
    plt.show()
