import numpy as np

def generate_data(distribution_type, size):
    if distribution_type == "normal":
        return np.random.normal(0, 1, size)
    elif distribution_type == "uniform":
        return np.random.uniform(-1, 1, size)
    # 可以添加更多分布类型
