# debate.py

def run_debates(network, problem, max_rounds):
    while not network.is_network_settled():
        # Agents debate with their neighbors in a polling manner
        for agent_id, agent in network.agents.items():
            if agent.active:
                neighbors = network.get_neighbors(agent_id)
                for neighbor_id in neighbors:
                    neighbor = network.get_agent(neighbor_id)
                    # Proceed if neighbor is active and they haven't agreed yet
                    if neighbor.active and neighbor_id not in agent.agreements:
                        agent.debate(neighbor, problem, max_rounds)
                agent.check_active(neighbors)
