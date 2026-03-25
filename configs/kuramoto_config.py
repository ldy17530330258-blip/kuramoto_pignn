import os
import torch

# project root = parent of this file's directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# runtime directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_GRAPH_DIR = os.path.join(DATA_DIR, 'raw_graphs')
SIM_DIR = os.path.join(DATA_DIR, 'simulations')
PYG_DIR = os.path.join(DATA_DIR, 'pyg_dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')

for d in [DATA_DIR, RAW_GRAPH_DIR, SIM_DIR, PYG_DIR, OUTPUT_DIR, CKPT_DIR, LOG_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

GLOBAL_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# graph generation
NETWORK_TYPES = ['ER', 'BA', 'WS']
NUM_NETWORKS_PER_TYPE = 100
N_RANGE = (80, 160)
ER_AVG_DEGREE_RANGE = (4, 10)
BA_M_RANGE = (2, 5)
WS_K_RANGE = (4, 10)
WS_P_REWIRE_RANGE = (0.05, 0.3)

# kuramoto simulation
K_RANGE = (0.5, 2.0)
OMEGA_STD = 1.0
T_MAX = 8.0
DT = 0.05
NUM_TIME_STEPS = int(T_MAX / DT) + 1
SYNC_THRESHOLD = 0.9

# dataset split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# model
NODE_FEATURE_DIM = 10
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
DROPOUT = 0.1

# train
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
EPOCHS = 80
PATIENCE = 12

# loss weights
LAMBDA_PHY = 0.01
LAMBDA_IC = 0.00
LAMBDA_R = 1.0

# scheduler
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-5

# stability
GRAD_CLIP_NORM = 1.0

# dataloader / runtime
NUM_WORKERS = 0
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# AMP / tqdm
USE_AMP = True
TQDM_MININTERVAL = 1.0