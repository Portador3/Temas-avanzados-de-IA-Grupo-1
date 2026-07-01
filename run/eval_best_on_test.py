"""Evaluate a saved `best.ckpt` against the test set -- read-only inference.

Does not train or modify anything: it rebuilds the model architecture from
the run's config, loads the weights from `ckpt/best.ckpt`, and runs a single
forward pass over the test loader (no backward, no optimizer) to report the
real test metrics for that checkpoint.

Important: pass the *original* config under `configs/`, the same one you
trained with (e.g. `configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego.yaml`),
not the dumped `results/.../config.yaml` -- the latter already has run-specific
keys (`run_dir`, resolved `out_dir`, ...) baked in that don't exist yet on a
fresh cfg object, and merging it in raises `KeyError: Non-existent config key`.
Pass the same `out_dir` / `dataset.dir` overrides you used when training, via
trailing `key value` pairs, exactly like `python -m fraudGT.main --cfg ...`.

Side effect: like every eval pass in this codebase, it appends one line to
`<run_dir>/test/stats.json` (tagged with epoch=0, since this logger has no
real epoch of its own -- check the printed metrics below rather than assuming
that file's rows are all real training epochs). No checkpoint or training
file is touched.

Usage:
    python run/eval_best_on_test.py \
        --cfg configs/AML-Small-HI/AML-Small-HI-SparseNodeGT+ports+Ego.yaml \
        --run_id 42 \
        out_dir ./results/AML-Small-HI dataset.dir ./data
"""
import argparse
import os
import sys

# Allow running this script directly (e.g. `python run/eval_best_on_test.py`)
# from any working directory: the `fraudGT` package lives at the repo root,
# one level above this file, and is only importable via `python -m
# fraudGT.main` normally -- there is no `pip install -e .` for it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric import seed_everything

import fraudGT  # noqa: registers custom modules
from fraudGT.graphgym.checkpoint import MODEL_STATE, BEST_METRIC
from fraudGT.graphgym.config import cfg, set_cfg, assert_cfg
from fraudGT.graphgym.loader import create_loader
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.utils.comp_budget import params_count
from fraudGT.graphgym.utils.device import auto_select_device
from fraudGT.logger import create_logger
from fraudGT.utils import custom_set_out_dir
from fraudGT.train.custom_train import eval_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', dest='cfg_file', required=True,
                         help='Path to the ORIGINAL config under configs/ '
                              '(the same one you passed to fraudGT.main).')
    parser.add_argument('--run_id', type=int, required=True,
                         help='Seed/run id whose ckpt/best.ckpt to evaluate, e.g. 42.')
    parser.add_argument('--gpu', type=int, default=-1,
                         help='GPU index, or -1 to auto-select (default).')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                         help='Same "key value" overrides you passed at train '
                              'time, e.g. out_dir ./results/AML-Small-HI '
                              'dataset.dir ./data')
    args = parser.parse_args()

    set_cfg(cfg)
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg(cfg)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
    # Note: intentionally not using fraudGT.utils.custom_set_run_dir here --
    # when cfg.train.auto_resume is False it calls makedirs_rm_exist(run_dir),
    # which would delete the very checkpoints we're trying to read. Setting
    # run_dir directly is the read-only equivalent.
    cfg.run_dir = os.path.join(cfg.out_dir, str(args.run_id))
    cfg.seed = args.run_id
    cfg.run_id = args.run_id
    seed_everything(cfg.seed)

    if args.gpu == -1:
        auto_select_device(strategy='greedy')
    else:
        cfg.device = f'cuda:{args.gpu}'

    ckpt_path = os.path.join(cfg.run_dir, 'ckpt', 'best.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No best.ckpt found at {ckpt_path}')

    loaders, dataset = create_loader(returnDataset=True)
    model = create_model(dataset=dataset)
    cfg.params = params_count(model)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device(cfg.device))
    model.load_state_dict(ckpt[MODEL_STATE])
    print(f'Loaded weights from {ckpt_path} '
          f'(best val {cfg.metric_best} at save time: {ckpt.get(BEST_METRIC)})')

    loggers = create_logger()
    test_loader = loaders[-1]  # create_loader always returns [train, val, test]
    eval_epoch(loggers[-1], test_loader, model, split='test')
    # write_epoch() unconditionally computes eta() even for non-train
    # loggers (it's just not included in the returned stats), and eta()
    # divides by `epoch + 1` -- so epoch=-1 triggers a ZeroDivisionError.
    # The value has no effect on the actual metrics for a 'test' logger.
    test_stats = loggers[-1].write_epoch(0)

    print('\nReal test metrics for this checkpoint (best.ckpt):')
    for k, v in test_stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
