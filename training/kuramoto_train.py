import os
import sys
import json
import pickle
import random
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from configs import kuramoto_config as config
from models.kuramoto_model import KuramotoPIGNN
from models.kuramoto_model_v2 import KuramotoPIGNN_V2
from training.losses import compute_total_loss


def set_seed(seed=config.GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_runtime(device):
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True


def use_v2_arch(tag: str) -> bool:
    return tag == 'v2_edge' or tag.startswith('v2_')


def build_model_by_tag(tag: str, device: torch.device):
    if use_v2_arch(tag):
        print("--> Using Edge-Message Architecture (V2)")
        model = KuramotoPIGNN_V2(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_GNN_LAYERS,
            dropout=config.DROPOUT,
        ).to(device)
    else:
        print("--> Using Node-GCN Architecture (V1)")
        model = KuramotoPIGNN(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_GNN_LAYERS,
            dropout=config.DROPOUT,
        ).to(device)
    return model


def load_dataset():
    data_path = os.path.join(config.PYG_DIR, 'kuramoto_dataset.pt')
    split_path = os.path.join(config.PYG_DIR, 'graph_split.pkl')

    data_list = torch.load(data_path, map_location='cpu', weights_only=False)
    with open(split_path, 'rb') as f:
        train_graphs, val_graphs, test_graphs = pickle.load(f)

    train_data, val_data, test_data = [], [], []
    for d in data_list:
        gid = int(d.graph_id.item())
        if gid in train_graphs:
            train_data.append(d)
        elif gid in val_graphs:
            val_data.append(d)
        else:
            test_data.append(d)

    return train_data, val_data, test_data


def make_loader(dataset, shuffle, device):
    num_workers = int(getattr(config, 'NUM_WORKERS', 0))
    pin_memory = bool(getattr(config, 'PIN_MEMORY', True)) and (device.type == 'cuda')
    persistent_workers = bool(getattr(config, 'PERSISTENT_WORKERS', False)) and (num_workers > 0)
    prefetch_factor = int(getattr(config, 'PREFETCH_FACTOR', 2))

    kwargs = dict(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(**kwargs)


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.weight = 0.0

    def update(self, value, n=1):
        self.total += float(value) * float(n)
        self.weight += float(n)

    @property
    def avg(self):
        if self.weight == 0:
            return 0.0
        return self.total / self.weight


def run_epoch(model, loader, optimizer=None, device=None, scaler=None, desc='Train'):
    is_train = optimizer is not None
    model.train(is_train)

    use_amp = bool(getattr(config, 'USE_AMP', True)) and (device.type == 'cuda')
    grad_clip_norm = float(getattr(config, 'GRAD_CLIP_NORM', 1.0))
    tqdm_mininterval = float(getattr(config, 'TQDM_MININTERVAL', 0.5))

    meters = {
        'total': AverageMeter(),
        'L_data': AverageMeter(),
        'L_phy': AverageMeter(),
        'L_ic': AverageMeter(),
        'L_R': AverageMeter(),
    }

    start_time = time.time()

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        leave=True,
        ascii=True,
        dynamic_ncols=False,
        mininterval=tqdm_mininterval,
        file=sys.stdout,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
    )

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for batch in pbar:
            batch = batch.to(device, non_blocking=(device.type == 'cuda'))

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                out = model(batch)
                loss, metrics = compute_total_loss(out, batch)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            weight = int(getattr(batch, 'num_graphs', 1))
            for k in meters:
                meters[k].update(metrics[k], n=weight)

            pbar.set_postfix_str(
                f"Tot={meters['total'].avg:.4f}, "
                f"Dat={meters['L_data'].avg:.4f}, "
                f"Phy={meters['L_phy'].avg:.4f}, "
                f"R={meters['L_R'].avg:.4f}"
            )

    epoch_time = time.time() - start_time
    result = {k: v.avg for k, v in meters.items()}
    result['time_sec'] = epoch_time
    result['steps_per_sec'] = len(loader) / max(epoch_time, 1e-8)
    return result


def train_model(tag='v1'):
    set_seed()
    device = config.DEVICE
    setup_runtime(device)

    print(f"Start training Kuramoto-PIGNN | device={device} | tag={tag}")

    train_data, val_data, test_data = load_dataset()
    train_loader = make_loader(train_data, shuffle=True, device=device)
    val_loader = make_loader(val_data, shuffle=False, device=device)
    test_loader = make_loader(test_data, shuffle=False, device=device)

    print(
        f"Dataset loaded | train={len(train_data)} | val={len(val_data)} | test={len(test_data)} | "
        f"batch_size={config.BATCH_SIZE} | num_workers={getattr(config, 'NUM_WORKERS', 0)} | "
        f"pin_memory={bool(getattr(config, 'PIN_MEMORY', True)) and (device.type == 'cuda')}"
    )

    model = build_model_by_tag(tag, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )

    use_amp = bool(getattr(config, 'USE_AMP', True)) and (device.type == 'cuda')
    scaler = GradScaler('cuda', enabled=use_amp) if device.type == 'cuda' else None

    best_val = float('inf')
    best_epoch = 0
    patience_count = 0
    ckpt_path = os.path.join(config.CKPT_DIR, f'kuramoto_pignn_{tag}_best.pt')
    logs = []

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n========== Epoch {epoch:03d}/{config.EPOCHS:03d} ==========", flush=True)

        tr = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            desc=f"Train Ep {epoch}"
        )
        va = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            scaler=scaler,
            desc=f"Val   Ep {epoch}"
        )

        scheduler.step(va['total'])
        current_lr = float(optimizer.param_groups[0]['lr'])

        log_row = {
            'epoch': epoch,
            'lr': current_lr,
            'train': tr,
            'val': va,
        }
        logs.append(log_row)

        print(
            f"[Epoch {epoch:03d}] "
            f"lr={current_lr:.2e} | "
            f"Train: Tot={tr['total']:.4f}, Dat={tr['L_data']:.4f}, Phy={tr['L_phy']:.4f}, R={tr['L_R']:.4f}, t={tr['time_sec']:.1f}s | "
            f"Val: Tot={va['total']:.4f}, Dat={va['L_data']:.4f}, Phy={va['L_phy']:.4f}, R={va['L_R']:.4f}, t={va['time_sec']:.1f}s",
            flush=True,
        )

        if va['total'] < best_val - 1e-6:
            best_val = va['total']
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_val': best_val,
                },
                ckpt_path,
            )
            print(f"[Best] checkpoint saved to: {ckpt_path}", flush=True)
        else:
            patience_count += 1
            print(f"[EarlyStop] patience {patience_count}/{config.PATIENCE}", flush=True)
            if patience_count >= config.PATIENCE:
                print(f"[EarlyStop] stop at epoch {epoch}, best epoch = {best_epoch}", flush=True)
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\n========== Testing ==========", flush=True)
    te = run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        scaler=scaler,
        desc="Test"
    )

    result = {'best_epoch': best_epoch, 'best_val': best_val, 'test': te}
    with open(os.path.join(config.LOG_DIR, f'kuramoto_pignn_{tag}_results.json'), 'w', encoding='utf-8') as f:
        json.dump({'logs': logs, 'result': result}, f, indent=2)

    print(
        f"Final test | Tot={te['total']:.4f}, Dat={te['L_data']:.4f}, "
        f"Phy={te['L_phy']:.4f}, R={te['L_R']:.4f}"
    )
    return model, result