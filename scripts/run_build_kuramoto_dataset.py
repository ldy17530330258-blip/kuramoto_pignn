import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_generation.build_kuramoto_dataset import build_and_save_dataset

if __name__ == '__main__':
    path, split_path, meta_path = build_and_save_dataset()
    print('saved dataset:', path)
    print('saved split:', split_path)
    print('saved meta:', meta_path)
