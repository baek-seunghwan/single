#!/usr/bin/env python3
import time
import re
from pathlib import Path

LOG = Path('train_long.log')
OUT = Path('monitor.log')
STATE = Path('.monitor_epoch')

def read_log():
    if not LOG.exists():
        return ''
    return LOG.read_text()

def parse_epochs(text):
    # Find lines like: "| end of epoch  10 | time:  2.37s | train_loss 1.7011 | valid rse 2.6739 | ..."
    pattern = re.compile(r"\| end of epoch\s*(\d+)\s*\|.*valid rse\s*([0-9\.eE+-]+).*\| us_rse\s*([0-9\.eE+-]+) \| jp_fx_rse\s*([0-9\.eE+-]+) \| kr_fx_rse\s*([0-9\.eE+-]+)")
    results = {}
    for m in pattern.finditer(text):
        e = int(m.group(1))
        results[e] = {
            'valid_rse': float(m.group(2)),
            'us_rse': float(m.group(3)),
            'jp_fx_rse': float(m.group(4)),
            'kr_fx_rse': float(m.group(5)),
        }
    return results

def load_state():
    if STATE.exists():
        try:
            return int(STATE.read_text().strip())
        except:
            return 0
    return 0

def save_state(e):
    STATE.write_text(str(e))

def main():
    last = load_state()
    OUT.write_text('')
    while True:
        txt = read_log()
        parsed = parse_epochs(txt)
        if not parsed:
            time.sleep(30)
            continue
        max_epoch = max(parsed.keys())
        # report any new epochs that are multiples of 10
        for e in sorted(parsed.keys()):
            if e > last and e % 10 == 0:
                entry = f"epoch {e} summary: valid_rse={parsed[e]['valid_rse']:.4f}, us_rse={parsed[e]['us_rse']:.4f}, jp_fx_rse={parsed[e]['jp_fx_rse']:.4f}, kr_fx_rse={parsed[e]['kr_fx_rse']:.4f}\n"
                print(entry, end='')
                OUT.write_text(OUT.read_text() + entry)
                save_state(e)
                last = e
        time.sleep(60)

if __name__ == '__main__':
    main()
