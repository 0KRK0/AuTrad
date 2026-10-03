from datetime import datetime, timezone

import pytest

from agents.events import Event, EventType, ImpactGraph, ShockDetector, classify_text
from agents.override import Override, OverrideBook
from agents.system import DecisionAgent
from agents.taxonomy import Category, Instrument, classify
from risk.gate import PortfolioState

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


def pf(**kw):
    base = dict(equity=1_000_000, sod_equity=1_000_000, peak_equity=1_000_000, weights={})
    base.update(kw)
    return PortfolioState(**base)


def test_taxonomy():
    assert classify(Instrument("DOGEUSDT", "crypto")) == Category.CRYPTO_MEME
    assert classify(Instrument("BTCINR", "crypto")) == Category.CRYPTO_MAJOR
    assert classify(Instrument("XYZ", "equity", price=8, adv_value=5e7, market_cap=1e9)) == Category.IN_PENNY
    assert classify(Instrument("ABC", "equity", price=900, adv_value=5e9, market_cap=3e11, surveillance=True)) == Category.IN_PENNY
    assert classify(Instrument("RELIANCE", "equity", 1400, 5e10, 1.9e13)) == Category.IN_LARGECAP
    assert classify(Instrument("GBPINR", "fx")) == Category.FX_INR


@pytest.mark.parametrize("text,etype,cat", [
    ("Elon Musk tweeted a dog picture", EventType.SOCIAL_POST, Category.CRYPTO_MEME),
    ("RBI governor hints at repo rate cut", EventType.POLICY_SPEECH, Category.FX_INR),
    ("Finance Minister announces budget changes", EventType.POLICY_SPEECH, Category.IN_LARGECAP),
    ("Bank of England speech sends pound lower", EventType.POLICY_SPEECH, Category.FX_INR),
    ("SEBI crackdown on crypto tax evasion", EventType.REGULATORY, Category.CRYPTO_ALT),
])
def test_event_classification(text, etype, cat):
    t, cats = classify_text(text)
    assert t == etype and cat in cats


def test_shock_detector_flags_unexplained_pump():
    sd = ShockDetector()
    for i in range(20):
        sd.update("P", 0.001 * (-1) ** i, 1000, False, NOW)
    assert sd.update("P", 0.15, 20000, has_news=False, ts=NOW).type == EventType.UNEXPLAINED_PUMP
    sd2 = ShockDetector()
    for i in range(20):
        sd2.update("Q", 0.001 * (-1) ** i, 1000, False, NOW)
    assert sd2.update("Q", 0.15, 20000, has_news=True, ts=NOW).type == EventType.PRICE_SHOCK


def test_impact_graph_learns_strength():
    g = ImpactGraph()
    e = Event(NOW, EventType.SOCIAL_POST, "", "x", [], {Category.CRYPTO_MEME, Category.CRYPTO_MAJOR}, 1.0, 1.0, 1.0)
    for _ in range(50):
        g.learn(e, {Category.CRYPTO_MEME: 0.10, Category.CRYPTO_MAJOR: 0.0})
    imp = g.impact([e])
    assert imp[Category.CRYPTO_MEME] > 0.05 and abs(imp[Category.CRYPTO_MAJOR]) < 0.005


def _brain(**kw):
    inst = {
        "DOGE": Instrument("DOGE", "crypto", 15, 1e10),
        "PEPE": Instrument("PEPE", "crypto", 1, 1e10, tags={"meme"}),
        "SHIB": Instrument("SHIB", "crypto", 1, 1e10),
        "RELIANCE": Instrument("RELIANCE", "equity", 1400, 5e10, 1.9e13),
    }
    return DecisionAgent(inst, **kw), inst


def _rows(syms, r1=0.0):
    return {s: {"r1": r1, "vol20": 0.02, "atr_z": 0.0} for s in syms}


def test_meme_entry_on_tweet_and_speculative_budget():
    brain, inst = _brain()
    tweet = Event(NOW, EventType.SOCIAL_POST, "Elon tweet", "x", ["DOGE", "PEPE", "SHIB"], {Category.CRYPTO_MEME}, 1.0, 1.0, 1.0)
    shock = [Event(NOW, EventType.PRICE_SHOCK, "", "market", [s], set(), 1.0, 1.0, 1.0) for s in ("DOGE", "PEPE", "SHIB")]
    ds = brain.step(0.0, _rows(inst), {s: 10.0 for s in inst}, {s: 1e6 for s in inst}, [tweet, *shock], pf())
    meme = [d for d in ds if d.category == "crypto_meme"]
    assert len(meme) == 3 and all(d.ok and 0 < d.approved <= 0.005 + 1e-12 for d in meme)
    assert sum(d.approved for d in meme) <= 0.02 + 1e-12


def test_pump_without_news_is_skipped():
    brain, inst = _brain()
    pump = Event(NOW, EventType.UNEXPLAINED_PUMP, "", "market", ["PEPE"], set(), 1.0, 1.0, 1.0)
    ds = brain.step(0.0, _rows(["PEPE"]), {"PEPE": 1.0}, {"PEPE": 1e6}, [pump], pf())
    assert any("pump" in d.reasons[0] and not d.ok for d in ds)


def test_stop_loss_exits_speculative_position():
    brain, inst = _brain()
    agent = brain.agents[Category.CRYPTO_MEME]
    agent.on_fill("DOGE", 10.0, 0.005, 0.0)
    ds = brain.step(0.0, _rows(["DOGE"]), {"DOGE": 8.0}, {"DOGE": 1e6}, [], pf(weights={"DOGE": 0.005}))
    assert ds[0].approved == 0.0 and "stop-loss" in ds[0].reasons[0]


def test_manual_override_buy_pin_and_cap():
    book = OverrideBook()
    brain, inst = _brain(overrides=book)
    book.add(Override("TARGET", "rajesh", "conviction", symbol="RELIANCE", weight=0.20, expires_at=1e9))
    ds = brain.step(0.0, _rows(["RELIANCE"]), {"RELIANCE": 1400.0}, {"RELIANCE": 1e6}, [], pf())
    human = [d for d in ds if d.source.startswith("human")]
    assert human[0].ok and human[0].approved == pytest.approx(0.10)  # human cap 10%, AI cap is 5%
    # pinned: AI does not touch RELIANCE afterwards
    ds2 = brain.step(1.0, _rows(["RELIANCE"], r1=-0.05), {"RELIANCE": 1300.0}, {"RELIANCE": 1e6}, [], pf(weights={"RELIANCE": 0.10}))
    assert not [d for d in ds2 if d.symbol == "RELIANCE"]


def test_override_cannot_bypass_kill_switch():
    book = OverrideBook()
    brain, _ = _brain(overrides=book)
    brain.gate.kill.trip("test")
    book.add(Override("BUY", "rajesh", "yolo", symbol="DOGE", weight=0.05))
    ds = brain.step(0.0, _rows(["DOGE"]), {"DOGE": 10.0}, {"DOGE": 1e6}, [], pf())
    assert not ds[0].ok and "kill switch" in ds[0].reasons[-1]


def test_pause_category_blocks_new_ai_entries():
    book = OverrideBook()
    brain, inst = _brain(overrides=book)
    book.add(Override("PAUSE_CATEGORY", "rajesh", "too hot", category=Category.CRYPTO_MEME))
    tweet = Event(NOW, EventType.SOCIAL_POST, "Elon", "x", ["DOGE"], {Category.CRYPTO_MEME}, 1.0, 1.0, 1.0)
    shock = Event(NOW, EventType.PRICE_SHOCK, "", "market", ["DOGE"], set(), 1.0, 1.0, 1.0)
    ds = brain.step(0.0, _rows(["DOGE"]), {"DOGE": 10.0}, {"DOGE": 1e6}, [tweet, shock], pf())
    assert not [d for d in ds if d.symbol == "DOGE" and d.approved > 0]
