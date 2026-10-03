"""Database access: pooled asyncpg connections + repositories that avoid N+1 queries.

With DATABASE_URL unset, an in-memory repository with the identical interface is used (dev/tests).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

POOL_MIN = int(os.getenv("DB_POOL_MIN", "5"))
POOL_MAX = int(os.getenv("DB_POOL_MAX", "20"))


async def create_pool():  # pragma: no cover - needs Postgres
    import asyncpg

    return await asyncpg.create_pool(
        os.environ["DATABASE_URL"], min_size=POOL_MIN, max_size=POOL_MAX,
        max_inactive_connection_lifetime=300, command_timeout=10,
        statement_cache_size=0,  # required behind PgBouncer transaction pooling
    )


@dataclass
class Page:
    items: list[dict[str, Any]]
    next_cursor: int | None


class InMemoryRepo:
    def __init__(self) -> None:
        self.orders: list[dict] = []
        self.fills: list[dict] = []
        self.evidence: list[dict] = []

    async def add_order(self, o: dict) -> dict:
        o = {"id": len(self.orders) + 1, "created_at": datetime.now(timezone.utc).isoformat(), **o}
        self.orders.append(o)
        return o

    async def add_fill(self, f: dict) -> None:
        self.fills.append({"id": len(self.fills) + 1, **f})

    async def list_orders(self, cursor: int | None, limit: int) -> Page:
        rows = [o for o in reversed(self.orders) if cursor is None or o["id"] < cursor][:limit]
        by_order: dict[int, list] = {}
        ids = {o["id"] for o in rows}
        for f in self.fills:  # one pass = one batched query, not one per order
            if f["order_id"] in ids:
                by_order.setdefault(f["order_id"], []).append(f)
        items = [{**o, "fills": by_order.get(o["id"], [])} for o in rows]
        return Page(items, rows[-1]["id"] if len(rows) == limit else None)

    async def list_evidence(self, symbol: str | None, cursor: int | None, limit: int) -> Page:
        rows = [e for e in reversed(self.evidence) if (symbol is None or symbol in e["symbols"]) and (cursor is None or e["seq"] < cursor)][:limit]
        return Page(rows, rows[-1]["seq"] if len(rows) == limit else None)


class PgRepo:  # pragma: no cover - needs Postgres
    def __init__(self, pool) -> None:
        self.pool = pool

    async def add_order(self, o: dict) -> dict:
        async with self.pool.acquire() as c:
            r = await c.fetchrow(
                "INSERT INTO orders (client_id, symbol, qty, type, status) VALUES ($1,$2,$3,$4,$5) "
                "ON CONFLICT (client_id) DO UPDATE SET client_id = EXCLUDED.client_id RETURNING *",
                o["client_id"], o["symbol"], o["qty"], o.get("type", "MARKET"), o.get("status", "NEW"))
            return dict(r)

    async def add_fill(self, f: dict) -> None:
        async with self.pool.acquire() as c:
            await c.execute("INSERT INTO fills (order_id, qty, price, commission) VALUES ($1,$2,$3,$4)",
                            f["order_id"], f["qty"], f["price"], f["commission"])

    async def list_orders(self, cursor: int | None, limit: int) -> Page:
        async with self.pool.acquire() as c:
            rows = await c.fetch(
                "SELECT * FROM orders WHERE ($1::bigint IS NULL OR id < $1) ORDER BY id DESC LIMIT $2", cursor, limit)
            ids = [r["id"] for r in rows]
            # Single batched query for all fills on the page (avoids N+1)
            fills = await c.fetch("SELECT * FROM fills WHERE order_id = ANY($1::bigint[]) ORDER BY id", ids) if ids else []
        by_order: dict[int, list] = {}
        for f in fills:
            by_order.setdefault(f["order_id"], []).append(dict(f))
        items = [{**dict(r), "fills": by_order.get(r["id"], [])} for r in rows]
        return Page(items, ids[-1] if len(ids) == limit else None)

    async def list_evidence(self, symbol: str | None, cursor: int | None, limit: int) -> Page:
        async with self.pool.acquire() as c:
            rows = await c.fetch(
                "SELECT seq, id, source, url, title, sentiment, symbols, published_at, ingested_at FROM evidence "
                "WHERE ($1::text IS NULL OR $1 = ANY(symbols)) AND ($2::bigint IS NULL OR seq < $2) "
                "ORDER BY seq DESC LIMIT $3", symbol, cursor, limit)
        items = [dict(r) for r in rows]
        return Page(items, items[-1]["seq"] if len(items) == limit else None)
