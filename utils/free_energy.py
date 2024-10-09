import numpy as np

def calculate_free_energy(network_output, expected_output):
    # 这里实现free energy的计算逻辑
    # 这只是一个示例，实际计算可能更复杂
    return np.mean((network_output - expected_output)**2)