import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")
# BASE SETTING
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
DROPOUT = 0.5
DATASETS = 'mosei'
MODEL_NAME = 'google/electra-base-discriminator'
EPOCHS = 150
MEMORY_SIZE = 32
FT = 1
BEST_EPOCH = 36
# MOSI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 35
#VISUAL_DIM = 47
#mosi 47 mosei 35
#TEXT_DIM = 4096
#TEXT_DIM = 1024
TEXT_DIM = 768
ALPHA = 10

BETA = -0.01
GAMA = -0.01
BN = 50

BL=11
# MOSEI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 35
# TEXT_DIM = 768

XLNET_INJECTION_INDEX = 1
#nohup python multimodal_driver.py --model xlnet-large-cased >> log/01.log 2>&1 &
