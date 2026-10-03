"""Two-tier cache: Redis (shared across API replicas) with in-process LRU fallback.

Used for: caching API responses, server-side caching of expensive queries (backtests, P&L rollups).
"""
from __future__ import annotations

import functools
import hashlib
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

import orjson


class LRUTTL:
    def __init__(self, maxsize: int = 2048) -> None:
        self.maxsize, self.data = maxsize, OrderedDict()

    async def get(self, k: str) -> bytes | None:
        v = self.data.get(k)
        if v is None:
            return None
        exp, val = v
        if exp < time.monotonic():
            self.data.pop(k, None)
            return None
        self.data.move_to_end(k)
        return val

    async def set(self, k: str, val: bytes, ttl: int) -> None:
        self.data[k] = (time.monotonic() + ttl, val)
        self.data.move_to_end(k)
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)

    async def delete_prefix(self, prefix: str) -> None:
        for k in [k for k in self.data if k.startswith(prefix)]:
            self.data.pop(k, None)


class RedisCache:  # pragma: no cover - needs a Redis server
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self.r = redis.from_url(url)

    async def get(self, k: str) -> bytes | None:
        return await self.r.get(k)

    async def set(self, k: str, val: bytes, ttl: int) -> None:
        await self.r.set(k, val, ex=ttl)

    async def delete_prefix(self, prefix: str) -> None:
        async for k in self.r.scan_iter(match=prefix + "*", count=500):
            await self.r.delete(k)


cache = RedisCache(os.environ["REDIS_URL"]) if os.getenv("REDIS_URL") else LRUTTL()
stats = {"hits": 0, "misses": 0}


def make_key(ns: str, *args: Any, **kwargs: Any) -> str:
    raw = orjson.dumps([args, kwargs], option=orjson.OPT_SORT_KEYS, default=str)
    return f"autrad:{ns}:{hashlib.sha256(raw).hexdigest()[:24]}"


def cached(ns: str, ttl: int) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Read-through cache for async functions returning JSON-serialisable data."""

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            key = make_key(ns, *args, **kwargs)
            hit = await cache.get(key)
            if hit is not None:
                stats["hits"] += 1
                return orjson.loads(hit)
            stats["misses"] += 1
            val = await fn(*args, **kwargs)
            await cache.set(key, orjson.dumps(val, default=str, option=orjson.OPT_SERIALIZE_NUMPY), ttl)
            return val

        wrapper.invalidate = lambda: cache.delete_prefix(f"autrad:{ns}:")  # type: ignore[attr-defined]
        return wrapper

    return deco
