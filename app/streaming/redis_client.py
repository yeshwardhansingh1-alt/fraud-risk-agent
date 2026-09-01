import redis.asyncio as aioredis
import json
from app.config import REDIS_HOST, REDIS_PORT

class RedisFeatureStore:
    def __init__(self):
        self.client = None

    async def connect(self):
        self.client = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1
        )
        try:
            await self.client.ping()
        except Exception:
            self.client = None

    async def increment_velocity(self, card_id: str, amount: float):
        """Atomic sliding-window velocity aggregation using Redis pipeline."""
        if not self.client:
            return {"tx_count_15m": 1, "tx_sum_15m": amount}
        pipe = self.client.pipeline()
        key_count = f"card:{card_id}:count_15m"
        key_sum = f"card:{card_id}:sum_15m"
        try:
            pipe.incr(key_count)
            pipe.expire(key_count, 900)  # 15-minute TTL
            pipe.incrbyfloat(key_sum, amount)
            pipe.expire(key_sum, 900)
            res = await pipe.execute()
            return {"tx_count_15m": res[0], "tx_sum_15m": res[2]}
        except Exception:
            return {"tx_count_15m": 1, "tx_sum_15m": amount}

    async def publish_transaction(self, tx_payload: dict):
        """Publish transaction to stream."""
        await self.client.xadd("tx_stream", {"payload": json.dumps(tx_payload)})
