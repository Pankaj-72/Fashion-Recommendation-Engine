"""Small Redis adapter; Redis is a cache and can be bypassed on connection errors."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


class RedisFeatureStore:
    def __init__(self, url: str | None = None, key_prefix: str = "recommender", client=None):
        self.key_prefix = key_prefix
        self.client = client
        if self.client is None:
            try:
                import redis
                self.client = redis.Redis.from_url(url or os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=1, decode_responses=True)
            except ImportError:
                logger.info("Redis package unavailable; using cache-disabled mode")

    def _key(self, kind, identifier):
        return f"{self.key_prefix}:{kind}:{identifier}"

    def get_json(self, kind: str, identifier: str):
        if self.client is None:
            return None
        try:
            raw = self.client.get(self._key(kind, identifier))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis read failed for %s: %s", kind, exc)
            return None

    def set_json(self, kind: str, identifier: str, value, ttl_seconds: int = 3600):
        if self.client is None:
            return False
        try:
            self.client.set(self._key(kind, identifier), json.dumps(value), ex=ttl_seconds)
            return True
        except Exception as exc:
            logger.warning("Redis write failed for %s: %s", kind, exc)
            return False