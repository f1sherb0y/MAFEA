# debate.py
from network import Network
from agent import Agent

def run_debates(network:Network, problem, max_rounds_per_pair):
    while not network.is_network_settled():
        # Agents debate with their neighbors in a polling manner
        for agent_id, agent in network.agents.items():
            if agent.active:
                neighbors = network.get_neighbors(agent_id)
                for neighbor_id in neighbors:
                    neighbor = network.get_agent(neighbor_id)
                    # Proceed if neighbor is active
                    if neighbor.active:
                        agent.debate(neighbor, problem, max_rounds_per_pair)
                agent.check_active()
