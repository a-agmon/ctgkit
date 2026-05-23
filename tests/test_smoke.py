from ctgkit import analyze
from ctgkit.synth import synth_epoch
from ctgkit.models import Category, AlertLevel

def test_normal_is_cat1_no_alert():
    r = analyze(synth_epoch("normal"))
    assert r.category == Category.REASSURING
    assert r.alert == AlertLevel.NONE

def test_prolonged_is_critical():
    r = analyze(synth_epoch("prolonged"))
    assert r.alert == AlertLevel.CRITICAL

def test_bad_signal_never_reassures():
    import numpy as np
    s = synth_epoch("normal"); s.fhr[: int(len(s.fhr)*0.7)] = np.nan
    r = analyze(s)
    assert r.category is None
    assert r.alert != AlertLevel.NONE

def test_acog_broader_cat2():
    s = synth_epoch("late_decels")
    assert int(analyze(s, guideline="acog").category) <= int(analyze(s, guideline="figo").category)

def test_json_serializable():
    import json
    json.dumps(analyze(synth_epoch("tachy")).to_dict())
