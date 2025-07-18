from app.conf.env import CENTRE_API_URL

REDIS_AUTH_TOKEN_PREFIX = "auth_token:{}"
MQ_DEVICE_REG_EXCHANGE = "device_reg_exchange"
MQ_DEVICE_HEARTBEAT_EXCHANGE = "device_heartbeat_exchange"
MQ_DEVICE_TRAINING_EXCHANGE = "device_training_exchange"

CENTER_API = {
    "device_register_api": f"{CENTRE_API_URL}/api/v1/learn_management/device/register/",
    "device_heartbeat_api": f"{CENTRE_API_URL}/api/v1/learn_management/device/heartbeat/",
    "device_training_api": f"{CENTRE_API_URL}/api/v1/learn_management/device/training/",
}
