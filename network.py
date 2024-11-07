# network.py

import networkx as nx
from agent import Agent

class Network:
    def __init__(self, config):
        self.graph = nx.Graph()
        self.agents = {}
        self.init_agents(config['nodes'])
        self.init_edges(config['edges'])

    def init_agents(self, nodes):
        self.agents.clear()
        self.graph.clear()
        for node in nodes:
            agent = Agent(agent_id=node['id'], capability=node['capability'])
            self.agents[node['id']] = agent
            self.graph.add_node(node['id'])

    def init_edges(self, edges):
        self.graph.add_edges_from(edges)

    def get_agent(self, agent_id):
        return self.agents[agent_id]

    def get_neighbors(self, agent_id):
        return list(self.graph.neighbors(agent_id))

    def is_network_settled(self):
        # The network is settled when all agents are inactive
        return all(not agent.active for agent in self.agents.values())
