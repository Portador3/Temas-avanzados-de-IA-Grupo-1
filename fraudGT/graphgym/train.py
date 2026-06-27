import logging
import os
import subprocess
import threading
import time

import torch

from fraudGT.graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt, save_ckpt_to
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch


_git_lock = threading.Lock()


def _git_push_ckpt(new_path: str, old_path: str = None):
    gh_user = os.environ.get('GH_USER', '')
    gh_token = os.environ.get('GH_TOKEN', '')
    rama = os.environ.get('RAMA', '')
    if not (gh_user and gh_token and rama):
        return

    def _push():
        with _git_lock:
            try:
                remote = f'https://{gh_user}:{gh_token}@github.com/{gh_user}/Temas-avanzados-de-IA-Grupo-1.git'
                if old_path and old_path != new_path:
                    subprocess.run(['git', 'rm', '--cached', '-f', old_path],
                                   check=False, capture_output=True)
                subprocess.run(['git', 'add', new_path], check=True, capture_output=True)
                subprocess.run(['git', 'commit', '-m', f'ckpt: {os.path.basename(new_path)}'],
                               check=True, capture_output=True)
                subprocess.run(['git', 'push', remote, f'HEAD:{rama}'],
                               check=True, capture_output=True)
                msg = f'[Git push] OK: {os.path.basename(new_path)} -> {rama}'
                print(msg, flush=True)
                logging.info(msg)
            except subprocess.CalledProcessError as e:
                msg = f'[Git push] FAILED: {os.path.basename(new_path)} — {e.stderr.decode().strip()}'
                print(msg, flush=True)
                logging.warning(msg)

    threading.Thread(target=_push, daemon=True).start()


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler):
    r"""
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    gh_user = os.environ.get('GH_USER', '')
    gh_token = os.environ.get('GH_TOKEN', '')
    rama = os.environ.get('RAMA', '')
    if gh_user and gh_token and rama:
        msg = f'[Git push] Enabled — user={gh_user}, branch={rama}'
    else:
        missing = [v for v, k in [('GH_USER', gh_user), ('GH_TOKEN', gh_token), ('RAMA', rama)] if not k]
        msg = f'[Git push] Disabled — faltan variables: {", ".join(missing)}'
    print(msg, flush=True)
    logging.info(msg)

    patience = max(10, cfg.optim.max_epoch // 10)
    best_val_auc = -1.0
    epochs_no_improve = 0
    last_ckpt_path = None
    best_ckpt_path = None

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            val_stats = None
            for i in range(1, num_splits):
                stats = eval_epoch(loggers[i], loaders[i], model,
                                   split=split_names[i - 1])
                epoch_stats = loggers[i].write_epoch(cur_epoch)
                if split_names[i - 1] == 'val':
                    val_stats = epoch_stats
            if val_stats is not None:
                val_auc = val_stats.get('auc', -1.0)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    epochs_no_improve = 0
                    new_best_path = os.path.join(cfg.run_dir, 'ckpt', 'best.ckpt')
                    save_ckpt_to(model, optimizer, scheduler, new_best_path)
                    _git_push_ckpt(new_best_path, best_ckpt_path)
                    best_ckpt_path = new_best_path
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(
                        f'Early stopping at epoch {cur_epoch}: '
                        f'val_auc did not improve for {patience} eval epochs '
                        f'(best={best_val_auc})'
                    )
                    break
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
            new_ckpt_path = os.path.join(cfg.run_dir, 'ckpt', f'{cur_epoch}.ckpt')
            _git_push_ckpt(new_ckpt_path, last_ckpt_path)
            last_ckpt_path = new_ckpt_path
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
