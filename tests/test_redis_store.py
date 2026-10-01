import json

from src.cache.redis_store import RedisFeatureStore


class FakeRedis:
    def __init__(self):
        self.values = {}

    def set(self, key, value, ex=None):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)


def test_redis_adapter_namespaces_and_roundtrips_json():
    client = FakeRedis()
    store = RedisFeatureStore(key_prefix="test", client=client)
    assert store.set_json("user", "u1", {"count": 3})
    assert client.values["test:user:u1"] == json.dumps({"count": 3})
    assert store.get_json("user", "u1") == {"count": 3}
