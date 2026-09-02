import torch

# Воспроизводимость
SEED = 42

# Размер изображений
IMG_SIZE = 224

# Размеры подвыборок (балансировка)
TARGET_TRAIN_SIZE = 20000
TARGET_VAL_SIZE   = 10000
TARGET_TEST_SIZE  = 5000

# Параметры обучения
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-5

# Параметры модели
CNN_FEATURE_DIM = 256
LSTM_HIDDEN_DIM = 256
TRANSFORMER_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2
DROPOUT = 0.3

# Устройство
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Нормализация ImageNet (для предобученного EfficientNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CONFIG = {
    'seed': SEED,
    'img_size': IMG_SIZE,
    'target_train_size': TARGET_TRAIN_SIZE,
    'target_val_size': TARGET_VAL_SIZE,
    'target_test_size': TARGET_TEST_SIZE,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'num_epochs': NUM_EPOCHS,
    'weight_decay': WEIGHT_DECAY,
    'cnn_feature_dim': CNN_FEATURE_DIM,
    'lstm_hidden_dim': LSTM_HIDDEN_DIM,
    'transformer_dim': TRANSFORMER_DIM,
    'num_heads': NUM_HEADS,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'device': str(DEVICE),
    'mean': MEAN,
    'std': STD,
}
