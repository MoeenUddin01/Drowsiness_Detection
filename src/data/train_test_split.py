import os
import shutil
import random

# ===============================
# CONFIG
# ===============================
SOURCE_DIR = "datas/raw"          # original dataset
TARGET_DIR = "datas/processed"    # where train/test will be created
TRAIN_RATIO = 0.8            # 80% train, 20% test
SEED = 42                    # for reproducibility

random.seed(SEED)

# ===============================
# CREATE TRAIN & TEST FOLDERS
# ===============================
train_dir = os.path.join(TARGET_DIR, "train")
test_dir = os.path.join(TARGET_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ===============================
# LOOP OVER EACH CLASS
# ===============================
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    # skip if not a folder
    if not os.path.isdir(class_path):
        continue

    files = os.listdir(class_path)
    random.shuffle(files)

    split_idx = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # create class folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # ===============================
    # COPY TRAIN FILES
    # ===============================
    for file in train_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(train_dir, class_name, file)
        shutil.copy(src, dst)

    # ===============================
    # COPY TEST FILES
    # ===============================
    for file in test_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(test_dir, class_name, file)
        shutil.copy(src, dst)

    print(f"{class_name}: {len(train_files)} train | {len(test_files)} test")

print("✅ Train-test split completed!")
