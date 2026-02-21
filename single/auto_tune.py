#!/usr/bin/env python3
"""Simple Phase-A auto-tuner: grid search over target-weight multipliers, lr, batch_size.

Usage: run from repo root with the venv python:
  .venv/bin/python single/auto_tune.py

This script runs `single/train_test.py` sequentially for each config, saving logs
under `auto_tune_logs/` and reporting the best config by `test rse` reported in stdout.
"""
import subprocess
import os
from pathlib import Path
import itertools
import time

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'auto_tune_logs'
LOG_DIR.mkdir(exist_ok=True)

# Phase A grid (moderate size). Adjust as needed.
kr_mults = [1.0, 2.0, 4.0]
jp_mults = [1.0, 2.0]
us_mults = [1.0, 2.0]
lrs = [3.5e-4, 1e-4]
batches = [4, 8]
epochs = 50

cmd_base = [str(ROOT / '.venv' / 'bin' / 'python'), 'single/train_test.py']

results = []

def run_experiment(env_vars, lr, batch_size, epochs, run_name):
    env = os.environ.copy()
    env.update(env_vars)
    cmd = cmd_base + [f'--lr={lr}', f'--batch_size={batch_size}', f'--epochs={epochs}']
    log_file = LOG_DIR / f'{run_name}.log'
    print('Running', run_name, '->', log_file)
    with open(log_file, 'wb') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            # Wait for process to finish; long but controlled by epochs
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise
    # Parse the log for last occurrence of 'test rse'
    rse = None
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'test rse' in line:
                # line like: "test rse 1.8058 | test rae ..."
                try:
                    part = line.split('test rse')[-1]
                    val = part.strip().split()[0]
                    rse = float(val.replace('|',''))
                except Exception:
                    pass
    return rse, str(log_file)

start = time.time()
count = 0
for kr_mul, jp_mul, us_mul, lr, batch in itertools.product(kr_mults, jp_mults, us_mults, lrs, batches):
    count += 1
    run_name = f'kr{kr_mul}_jp{jp_mul}_us{us_mul}_lr{lr}_b{batch}'
    env_vars = {
        'TARGET_KR_MULT': str(kr_mul),
        'TARGET_JP_MULT': str(jp_mul),
        'TARGET_US_MULT': str(us_mul),
    }
    rse, logfile = run_experiment(env_vars, lr, batch, epochs, run_name)
    results.append((rse, run_name, logfile))

# Summarize
results_sorted = sorted([r for r in results if r[0] is not None], key=lambda x: x[0])
print('\nAuto-tune finished in', time.time()-start, 's')
if results_sorted:
    best = results_sorted[0]
    print('Best test rse:', best[0], 'run:', best[1], 'log:', best[2])
else:
    print('No runs reported test rse. Inspect logs in', LOG_DIR)
