import networkx as nx
from agent import Agent
from typing import Dict, Tuple

class Network:
    def __init__(self, config):
        self.graph = nx.Graph()
        self.agents: Dict[str, Agent] = {}
        self.agreement_status: Dict[Tuple[str, str], bool] = {}
        self.init_agents(config['nodes'])
        self.init_edges(config['edges'])
        self.init_agreements()

    def init_agents(self, nodes):
        self.agents.clear()
        self.graph.clear()
        for node in nodes:
            agent = Agent(agent_id=node['id'], capability=node['capability'])
            self.agents[node['id']] = agent
            self.graph.add_node(node['id'])

    def init_edges(self, edges):
        self.graph.add_edges_from(edges)

    def init_agreements(self):
        # Initialize agreement status between connected agents
        for agent_id in self.agents:
            for neighbor_id in self.get_neighbors(agent_id):
                key = tuple(sorted([agent_id, neighbor_id]))
                self.agreement_status[key] = False  # Initially, agents do not agree

    def reset(self):
        """
        Resets the network state while preserving the graph structure.
        1. Calls reset() on each agent
        2. Resets agreement status
        """
        # Reset all agents
        for agent in self.agents.values():
            agent.reset()
        
        # Reset agreement status while keeping the same connections
        self.agreement_status.clear()
        self.init_agreements()

    def get_agent(self, agent_id):
        return self.agents[agent_id]

    def get_neighbors(self, agent_id):
        return list(self.graph.neighbors(agent_id))

    def update_agreement(self, agent_id1, agent_id2, agree: bool):
        key = tuple(sorted([agent_id1, agent_id2]))
        self.agreement_status[key] = agree

    def agents_disagree(self, agent_id1, agent_id2):
        key = tuple(sorted([agent_id1, agent_id2]))
        return not self.agreement_status.get(key, False)

    def is_network_settled(self):
        # The network is settled when all active agents agree with each other (excluding inactive agents)
        active_agents = [agent_id for agent_id, agent in self.agents.items() if agent.active]
        if len(active_agents) <= 1:
            return True
        for i in range(len(active_agents)):
            for j in range(i+1, len(active_agents)):
                agent_id1 = active_agents[i]
                agent_id2 = active_agents[j]
                if self.agents_disagree(agent_id1, agent_id2):
                    return False
        return True