from network import Network
from agent import Agent

def run_debates(network: Network, problem, max_rounds_per_pair):
    settled = False
    while not settled:
        # Agents debate with their neighbors
        settled = True
        for agent_id, agent in network.agents.items():
            if agent.active:
                neighbors = network.get_neighbors(agent_id)
                for neighbor_id in neighbors:
                    neighbor = network.get_agent(neighbor_id)
                    # Proceed if neighbor is active and they disagree
                    if neighbor.active and network.agents_disagree(agent_id, neighbor_id):
                        agree = agent.debate(neighbor, problem, max_rounds_per_pair, network)
                        settled = settled and agree
                agent.check_active()
