import torch

class TrainingConfig:
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 2
    MAX_LENGTH = 38
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2
    SEED = 42
    DATA_DIR = '../data/processed'
    MODEL_SAVE_DIR = '../models/checkpoints'
    RESULTS_DIR = '../results'
    LOGGING_STEPS = 100
    EVAL_STEPS = 500
    SAVE_STEPS = 500
   
    @classmethod
    def print_config(cls):
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Max length: {cls.MAX_LENGTH}")
        print(f"Batch size: {cls.TRAIN_BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Warmup steps: {cls.WARMUP_STEPS}")
        print("="*60 + "\n")

if __name__ == "__main__":
    TrainingConfig.print_config()
    print(f"✓ GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU name: {torch.cuda.get_device_name(0)}")
        print(f" GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")