from .base_agent import BaseAgent
import openai

class LLMAgent(BaseAgent):
    def __init__(self, agent_id, learning_capacity, api_key):
        super().__init__(agent_id, learning_capacity)
        openai.api_key = api_key

    def process(self, input_data):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=input_data,
            max_tokens=100
        )
        return response.choices[0].text.strip()