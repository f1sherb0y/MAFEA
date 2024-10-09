from agents.llm_agent import LLMAgent
from agents.local_agent import LocalModelAgent
from networks.network import SubNetwork
from networks.network import MultiAgentNetwork
from data.generator import generate_data
from utils.free_energy import calculate_free_energy
from config import CONFIG
import numpy as np

def create_agent(agent_type, agent_id, learning_capacity):
    if agent_type == "llm":
        return LLMAgent(agent_id, learning_capacity, CONFIG["openai_api_key"])
    elif agent_type == "local":
        # 假设我们有一个本地模型
        local_model = None  # 这里应该初始化一个实际的本地模型
        return LocalModelAgent(agent_id, learning_capacity, local_model)

def create_network(num_sub_networks, agents_per_sub_network, agent_heterogeneity, sub_network_heterogeneity, data_distribution_correlation):
    sub_networks = []
    for i in range(num_sub_networks):
        agents = []
        for j in range(agents_per_sub_network):
            agent_type = "llm" if j % 2 == 0 else "local"
            learning_capacity = np.random.uniform(0.5, 1.0)  # 随机学习能力
            agent = create_agent(agent_type, f"agent_{i}_{j}", learning_capacity)
            
            # 应用智能体异质性
            agent.update_parameters({"heterogeneity": np.random.normal(0, agent_heterogeneity)})
            agents.append(agent)
        
        # 应用子网络异质性和数据分布相关性
        data_distribution = generate_data("normal" if i % 2 == 0 else "uniform", CONFIG["data_size"])
        sub_network = SubNetwork(agents, data_distribution)
        sub_network.heterogeneity = np.random.normal(0, sub_network_heterogeneity)
        sub_networks.append(sub_network)
    
    return MultiAgentNetwork(sub_networks)

def run_experiment(agent_heterogeneity, sub_network_heterogeneity, data_distribution_correlation, network_scale, learning_capacity):
    network = create_network(
        num_sub_networks=network_scale,
        agents_per_sub_network=CONFIG["agents_per_sub_network"],
        agent_heterogeneity=agent_heterogeneity,
        sub_network_heterogeneity=sub_network_heterogeneity,
        data_distribution_correlation=data_distribution_correlation
    )
    
    input_data = generate_data("normal", CONFIG["data_size"])
    network_output = network.process(input_data)
    
    # 假设我们有一个期望输出
    expected_output = generate_data("normal", len(network_output))
    
    free_energy = calculate_free_energy(network_output, expected_output)
    return free_energy

def main():
    # 示例：运行一次实验
    free_energy = run_experiment(
        agent_heterogeneity=0.1,
        sub_network_heterogeneity=0.2,
        data_distribution_correlation=0.5,
        network_scale=CONFIG["num_sub_networks"],
        learning_capacity=0.8
    )
    print(f"Calculated Free Energy: {free_energy}")

if __name__ == "__main__":
    main()
