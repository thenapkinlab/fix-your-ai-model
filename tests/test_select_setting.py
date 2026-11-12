"""Run with: python -I tests/test_select_setting.py  (or pytest). No model needed."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interp"))
from selection import HARM_TOLERANCE, select_setting

def s(layer, hm, fr, lam=1.0): return dict(layer=layer, lam=lam, fr_still_refused=fr, harmful_refused=hm)

def stops(val, base):
    try: select_setting(val, base)
    except SystemExit as e: return str(e)
    raise AssertionError("expected NO_ADMISSIBLE_SETTING")

def test_empty_list():
    assert stops([], 1.0).startswith("NO_ADMISSIBLE_SETTING")

def test_all_below_threshold():
    msg = stops([s(12, 0.90, 0.1), s(14, 0.80, 0.0)], 1.0)
    assert "NO_ADMISSIBLE_SETTING" in msg and "threshold 95.0%" in msg and "L12" in msg and "L14" in msg

def test_exactly_on_threshold_accepted():
    assert HARM_TOLERANCE == 0.05
    assert select_setting([s(16, 0.95, 0.5)], 1.0)["layer"] == 16

def test_mix_picks_best_passing():
    val = [s(12, 0.50, 0.0), s(14, 1.00, 0.6), s(16, 0.975, 0.4), s(18, 0.95, 0.4), s(20, 0.90, 0.1)]
    best = select_setting(val, 1.0)
    assert (best["layer"], best["harmful_refused"]) == (16, 0.975)   # lowest fr among passing; tie broken by higher harmful

def test_rene_example():
    assert "NO_ADMISSIBLE_SETTING" in stops([s(12, 0.50, 0.20), s(14, 0.30, 0.05)], 1.0)

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"): f(); print("ok", name)
