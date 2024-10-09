from .base_agent import BaseAgent

class LocalModelAgent(BaseAgent):
    def __init__(self, agent_id, learning_capacity, model):
        super().__init__(agent_id, learning_capacity)
        self.model = model

    def process(self, input_data):
        # 这里应该实现本地模型的处理逻辑
        return self.model.predict(input_data)