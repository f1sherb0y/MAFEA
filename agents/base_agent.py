from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, agent_id, learning_capacity):
        self.agent_id = agent_id
        self.learning_capacity = learning_capacity
        self.parameters = {}

    @abstractmethod
    def process(self, input_data):
        pass

    def update_parameters(self, new_parameters):
        self.parameters.update(new_parameters)