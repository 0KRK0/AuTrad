from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from api.main import app
from intel.collector import Collector, EvidenceStore, SourceReliability, evidence_features, parse_feed

ADMIN = {"Authorization": "Bearer dev-admin"}
VIEWER = {"Authorization": "Bearer dev-viewer"}


def test_api_auth_backtest_cache_gzip_killswitch():
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/api/v1/risk/limits").status_code == 401
        body = {"symbols": ["A", "B"], "n_bars": 300, "seed": 1, "folds": 2}
        assert c.post("/api/v1/backtests", json=body, headers=VIEWER).status_code == 403
        r1 = c.post("/api/v1/backtests", json=body, headers={**ADMIN, "Accept-Encoding": "gzip"})
        assert r1.status_code == 200 and r1.headers.get("content-encoding") == "gzip"
        r2 = c.post("/api/v1/backtests", json=body, headers=ADMIN)
        assert r2.json()["metrics"] == r1.json()["metrics"]
        assert c.get("/api/v1/system/cache", headers=ADMIN).json()["hits"] >= 1
        assert c.post("/api/v1/killswitch/trip", json={"reason": "test"}, headers=VIEWER).status_code == 403
        assert c.post("/api/v1/killswitch/trip", json={"reason": "test"}, headers=ADMIN).json()["active"]
        assert c.get("/api/v1/killswitch", headers=VIEWER).json()["active"]


def test_orders_keyset_pagination_with_batched_fills():
    import asyncio

    with TestClient(app) as c:
        repo = app.state.repo

        async def seed():
            for i in range(7):
                o = await repo.add_order({"client_id": str(i), "symbol": "X", "qty": 1})
                await repo.add_fill({"order_id": o["id"], "qty": 1, "price": 10, "commission": 0})

        asyncio.run(seed())
        p1 = c.get("/api/v1/orders?limit=3", headers=VIEWER).json()
        p2 = c.get(f"/api/v1/orders?limit=3&cursor={p1['next_cursor']}", headers=VIEWER).json()
        assert [o["id"] for o in p1["items"]] == [7, 6, 5] and [o["id"] for o in p2["items"]] == [4, 3, 2]
        assert all(len(o["fills"]) == 1 for o in p1["items"])


RSS = """<rss><channel>
<item><title>Acme beats estimates, raises guidance</title><link>https://ex.com/1</link>
<description>Acme Corp record growth</description><pubDate>Mon, 28 Sep 2026 12:00:00 GMT</pubDate></item>
<item><title>Acme beats estimates, raises guidance!</title><link>https://ex.com/1b</link>
<description>Acme Corp record growth</description><pubDate>Mon, 28 Sep 2026 12:05:00 GMT</pubDate></item>
<item><title>Globex faces probe over fraud warning</title><link>https://ex.com/2</link>
<description>Globex shares plunge</description></item>
</channel></rss>"""


def test_intel_parse_dedupe_entities_point_in_time():
    store = EvidenceStore()
    col = Collector(store, {"ACME": ["Acme"], "GBX": ["Globex"]})
    added = col.ingest(parse_feed(RSS, "wire"))
    assert added == 2  # near-duplicate dropped by SimHash
    acme = [e for e in store.items if "ACME" in e.symbols][0]
    assert acme.sentiment > 0 and [e for e in store.items if "GBX" in e.symbols][0].sentiment < 0
    now = datetime.now(timezone.utc)
    rel = SourceReliability()
    assert evidence_features(store, rel, "ACME", now)["news_sent"] > 0
    assert evidence_features(store, rel, "ACME", now - timedelta(hours=1))["news_count"] == 0  # no look-ahead


def test_source_reliability_decays_bad_sources():
    rel = SourceReliability()
    for i in range(200):
        s = 1.0 if i % 2 else -1.0
        rel.update("good", s, 0.01 * s)
        rel.update("bad", s, -0.01 * s)
    assert rel.weight("good") > 0.9 and rel.weight("bad") == 0.0


def test_override_api_roles_and_validation():
    with TestClient(app) as c:
        body = {"action": "BUY", "symbol": "DOGEINR", "weight": 0.01, "reason": "Elon tweet, my call"}
        assert c.post("/api/v1/overrides", json=body, headers=VIEWER).status_code == 403
        r = c.post("/api/v1/overrides", json=body, headers=ADMIN)
        assert r.status_code == 200 and r.json()["status"] == "queued"
        assert c.post("/api/v1/overrides", json={"action": "PAUSE_CATEGORY", "reason": "no cat"}, headers=ADMIN).status_code == 422
        assert c.post("/api/v1/overrides", json={"action": "PAUSE_CATEGORY", "category": "crypto_meme", "reason": "too hot"},
                      headers=ADMIN).status_code == 200
        ov = c.get("/api/v1/overrides", headers=VIEWER).json()
        assert ov["paused"] == ["crypto_meme"] and len(ov["items"]) == 2
        agents = c.get("/api/v1/agents", headers=VIEWER).json()["agents"]
        assert {a["category"] for a in agents} >= {"in_penny", "crypto_meme", "fx_inr"}
