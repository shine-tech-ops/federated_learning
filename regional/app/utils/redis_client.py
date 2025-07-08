import redis
from loguru import logger
import app.conf.env as env


class RedisClient:
    def __init__(self):
        self.host = getattr(env, "REDIS_HOST", "localhost")
        self.port = getattr(env, "REDIS_PORT", 6379)
        self.db = getattr(env, "REDIS_DB", 0)
        self.password = getattr(env, "REDIS_PASSWORD", None)
        self._client = None

    def connect(self):
        """建立 Redis 连接"""
        try:
            self._client = redis.StrictRedis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,  # 返回字符串而非字节
                socket_connect_timeout=5
            )
            self._client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    @property
    def client(self):
        if not self._client:
            self.connect()
        return self._client

    def set_key(self, key, value, expire=None):
        """
        设置键值对
        :param key: 键名
        :param value: 值
        :param expire: 过期时间（秒）
        """
        try:
            self.client.set(key, value)
            if expire:
                self.client.expire(key, expire)
            logger.debug(f"Redis SET {key} = {value}, EX={expire}")
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {e}")

    def get_key(self, key):
        """
        获取键值
        :param key: 键名
        :return: 值或 None
        """
        try:
            value = self.client.get(key)
            logger.debug(f"Redis GET {key} = {value}")
            return value
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {e}")
            return None

    def delete_key(self, key):
        """
        删除键
        :param key: 键名
        """
        try:
            self.client.delete(key)
            logger.debug(f"Redis DEL {key}")
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {e}")

    def hash_set(self, name, key, value, expire=None):
        """
        在哈希表中设置字段值
        :param name: 哈希表名
        :param key: 字段名
        :param value: 值
        :param expire: 过期时间（秒）
        """
        try:
            self.client.hset(name, key, value)
            if expire:
                self.client.expire(name, expire)
            logger.debug(f"Redis HSET {name} {key} = {value}, EX={expire}")
        except Exception as e:
            logger.error(f"Error hset Redis hash {name}: {e}")

    def hash_get(self, name, key):
        """
        获取哈希表中的字段值
        :param name: 哈希表名
        :param key: 字段名
        :return: 值或 None
        """
        try:
            value = self.client.hget(name, key)
            logger.debug(f"Redis HGET {name} {key} = {value}")
            return value
        except Exception as e:
            logger.error(f"Error hget Redis hash {name}: {e}")
            return None

    def hash_delete(self, name, key):
        """
        删除哈希表中的字段
        :param name: 哈希表名
        :param key: 字段名
        """
        try:
            self.client.hdel(name, key)
            logger.debug(f"Redis HDEL {name} {key}")
        except Exception as e:
            logger.error(f"Error hdel Redis hash {name}: {e}")

    def exists(self, key):
        """
        判断键是否存在
        :param key: 键名
        :return: bool
        """
        try:
            return self.client.exists(key)
        except Exception as e:
            logger.error(f"Error checking Redis key existence: {e}")
            return False

    def set_expire(self, key, seconds):
        """
        设置键的过期时间
        :param key: 键名
        :param seconds: 秒数
        """
        try:
            self.client.expire(key, seconds)
            logger.debug(f"Redis EXPIRE {key} {seconds}s")
        except Exception as e:
            logger.error(f"Error setting Redis key expiry: {e}")

