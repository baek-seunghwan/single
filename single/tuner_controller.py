#!/usr/bin/env python3
"""Controller that monitors auto_tune_logs and runs longer training for promising configs.

Behavior:
- Polls `auto_tune_logs/` every `poll_interval` seconds.
- Parses logs for `test rse` and per-variable `us_rse`, `jp_fx_rse`, `kr_fx_rse`.
- When a config with best `test rse` below `long_train_trigger` is found,
  it runs a long training (in its own log file) and checks per-variable RSEs.
- If `kr/jp/us` RSEs <= `target_rse` it stops and reports success.

Run:
  .venv/bin/python single/tuner_controller.py &
"""
import time
import re
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'auto_tune_logs'
LONG_LOG_DIR = ROOT / 'auto_tune_long'
LONG_LOG_DIR.mkdir(exist_ok=True)

poll_interval = int(os.environ.get('TUNER_POLL_SEC', '60'))
long_train_trigger = float(os.environ.get('TUNER_TRIGGER_RSE', '0.6'))
target_rse = float(os.environ.get('TUNER_TARGET_RSE', '0.5'))
full_epochs = int(os.environ.get('TUNER_FULL_EPOCHS', '300'))

def parse_log_for_rse(path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    test_rse = None
    # find last 'test rse' occurrence
    m = re.findall(r'test rse\s*([0-9]*\.?[0-9]+)', text)
    if m:
        try:
            test_rse = float(m[-1])
        except:
            test_rse = None

    # per-variable rse
    var_rse = {}
    for var in ['us_rse', 'jp_fx_rse', 'kr_fx_rse', 'kr_fx_rse']:
        mv = re.findall(rf'{var}\s*[:=]?\s*([0-9]*\.?[0-9]+)', text)
        if mv:
            try:
                var_rse[var] = float(mv[-1])
            except:
                pass

    # Additionally try patterns like 'us_rse 0.5118' or 'us_rse: 0.5118'
    for var in ['us_rse','jp_fx_rse','kr_fx_rse']:
        if var not in var_rse:
            mv = re.findall(rf'{var}[^0-9\n\r]*([0-9]*\.?[0-9]+)', text)
            if mv:
                try:
                    var_rse[var] = float(mv[-1])
                except:
                    pass

    return test_rse, var_rse

def find_best_auto_tune():
    best = (None, None, None)  # (rse, runname, path)
    if not LOG_DIR.exists():
        return best
    for p in LOG_DIR.glob('*.log'):
        rse, var = parse_log_for_rse(p)
        if rse is None:
            continue
        runname = p.stem
        if best[0] is None or rse < best[0]:
            best = (rse, runname, p)
    return best

def launch_full_train_from_runname(runname, logpath, epochs):
    # runname like kr1.0_jp1.0_us1.0_lr0.00035_b4
    # extract multipliers and lr/b
    env = os.environ.copy()
    for part in runname.split('_'):
        if part.startswith('kr'):
            env['TARGET_KR_MULT'] = part[2:]
        if part.startswith('jp'):
            env['TARGET_JP_MULT'] = part[2:]
        if part.startswith('us'):
            env['TARGET_US_MULT'] = part[2:]
        if part.startswith('lr'):
            env['LR_OVERRIDE'] = part[2:]
        if part.startswith('b'):
            env['BATCH_OVERRIDE'] = part[1:]

    # Build command
    cmd = [str(ROOT / '.venv' / 'bin' / 'python'), 'single/train_test.py', f'--epochs={epochs}']
    if 'LR_OVERRIDE' in env:
        cmd.append(f"--lr={env['LR_OVERRIDE']}")
    if 'BATCH_OVERRIDE' in env:
        cmd.append(f"--batch_size={env['BATCH_OVERRIDE']}")

    out = LONG_LOG_DIR / f'full_{runname}.log'
    print('Launching full train:', ' '.join(cmd), '->', out)
    with open(out, 'wb') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return proc.pid, out

def main():
    print('Tuner controller started. Polling', LOG_DIR)
    while True:
        try:
            best = find_best_auto_tune()
            if best[0] is not None:
                print('Best auto-tune so far:', best[1], 'test rse=', best[0])
                if best[0] <= long_train_trigger:
                    # launch full train and monitor
                    pid, logfile = launch_full_train_from_runname(best[1], best[2], full_epochs)
                    print('Launched full train pid', pid, 'log', logfile)
                    # wait for the process to complete and then parse results
                    while True:
                        time.sleep(30)
                        if not os.path.exists(logfile):
                            continue
                        trse, vars = parse_log_for_rse(logfile)
                        if trse is not None:
                            print('Full train test rse:', trse, 'vars:', vars)
                            # check target variables
                            kr = vars.get('kr_fx_rse', None)
                            jp = vars.get('jp_fx_rse', None)
                            us = vars.get('us_rse', None)
                            if kr is not None and jp is not None and us is not None:
                                if kr <= target_rse and jp <= target_rse and us <= target_rse:
                                    print('Target achieved! kr/jp/us <=', target_rse)
                                    return
                                else:
                                    print('Target not met yet. kr/jp/us:', kr, jp, us)
                                    # increase multipliers slightly and run again
                                    try:
                                        km = float(os.environ.get('TARGET_KR_MULT','1.0'))
                                    except:
                                        km = 1.0
                                    # bump env variables for next long run
                                    os.environ['TARGET_KR_MULT'] = str(km * 1.5)
                                    print('Increasing TARGET_KR_MULT to', os.environ['TARGET_KR_MULT'])
                                    # continue outer loop to pick next best
                                    break
                        # else keep waiting up to a timeout
                    # continue polling
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            print('Tuner controller interrupted, exiting')
            return

if __name__ == '__main__':
    main()
