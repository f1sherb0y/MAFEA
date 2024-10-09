
class SubNetwork:
    def __init__(self, agents, data_distribution):
        self.agents = agents
        self.data_distribution = data_distribution

    def process(self, input_data):
        results = []
        for agent in self.agents:
            results.append(agent.process(input_data))
        return results



class MultiAgentNetwork:
    def __init__(self, sub_networks):
        self.sub_networks = sub_networks

    def process(self, input_data):
        results = []
        for sub_network in self.sub_networks:
            results.extend(sub_network.process(input_data))
        return results