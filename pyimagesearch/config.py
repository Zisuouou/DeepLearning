import os

ORIG_INPUT_DATASET = "breast_dataset/dataset/"

BASE_PATH = "data/split/"

TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.8

VAL_SPLIT = 0.1