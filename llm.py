# llm.py

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

def get_llm(capability_level):
    """
    Returns an LLM instance based on the agent's capability level.
    """
    # Map capability levels to specific models or configurations
    if capability_level >= 4:
        # High capability: Use a more powerful model
        return OpenAI(model_name="text-davinci-003", temperature=0)
    elif capability_level == 3:
        # Medium capability
        return OpenAI(model_name="text-curie-001", temperature=0.5)
    else:
        # Low capability
        return OpenAI(model_name="text-babbage-001", temperature=0.7)
