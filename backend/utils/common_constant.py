AGGREGATION_STRATEGIES_MAP = {
    "fedavg": {"label": "联邦平均 (Federated Averaging, FedAvg)", "type": "fedavg"},
    "fedavgm": {"label": "联邦平均 with Momentum (FedAvgM)", "type": "fedavgm"},
    "fedadagrad": {"label": "联邦 Adagrad (Federated Adagrad)", "type": "fedadagrad"},
    "fedadam": {"label": "联邦 Adam (Federated Adam)", "type": "fedadam"},
    "fedopt": {"label": "联邦优化器 (Federated Optimizer)", "type": "fedopt"},
    "fedprox": {"label": "联邦近端平均 (Federated Proximal Averaging, FedProx)", "type": "fedprox"},
    "fedtrimmedavg": {"label": "联邦截尾平均 (Federated Trimmed Averaging)", "type": "fedtrimmedavg"},
    "fedmedian": {"label": "联邦中位数 (Federated Median)", "type": "fedmedian"},
    "buliyan": {"label": "Bulyan", "type": "buliyan"},
    "krum": {"label": "Krum", "type": "krum"},
    "qfedavg": {"label": "Q-FedAvg (Quantile-FedAvg)", "type": "qfedavg"},
    "dp_adaptive_clipping": {"label": "差分隐私自适应梯度裁剪 (DP Adaptive Clipping)", "type": "dp_adaptive_clipping"},
    "dp_fixed_clipping": {"label": "差分隐私固定梯度裁剪 (DP Fixed Clipping)", "type": "dp_fixed_clipping"},
    "dpfedavg_adaptive": {"label": "差分隐私联邦平均自适应裁剪 (DPFedAvg Adaptive)", "type": "dpfedavg_adaptive"},
    "dpfedavg_fixed": {"label": "差分隐私联邦平均固定裁剪 (DPFedAvg Fixed)", "type": "dpfedavg_fixed"},
    "fedavg_android": {"label": "联邦平均 (Android 优化)", "type": "fedavg_android"},
    "fedxgb_bagging": {"label": "联邦 XGBoost Bagging", "type": "fedxgb_bagging"},
    "fedxgb_cyclic": {"label": "联邦 XGBoost 循环", "type": "fedxgb_cyclic"},
    "fedxgb_nn_avg": {"label": "联邦 XGBoost 神经网络平均", "type": "fedxgb_nn_avg"},
    "fedyogi": {"label": "联邦 Yogi (Federated Yogi)", "type": "fedyogi"}
}

# MQ device_reg 队列
MQ_DEVICE_REG_QUEUE = "device_reg"
MQ_DEVICE_REG_EXCHANGE = "device_reg_exchange"
# MQ device heartbeat 队列
MQ_DEVICE_HEARTBEAT_QUEUE = "device_heartbeat"
MQ_DEVICE_HEARTBEAT_EXCHANGE = "device_heartbeat_exchange"

# MQ device_training 队列
MQ_DEVICE_TRAINING_QUEUE = "device_training"
MQ_DEVICE_TRAINING_EXCHANGE = "device_training_exchange"


REDIS_AUTH_TOKEN_PREFIX = "device_auth_token:{}"

HEARTBEAT_TIMEOUT = 60 * 2


