# MAFEA
Free Energy Analysis for Multi-Agent Networks

## How to run

### Environment setup

1. Make sure python>=3.12 is installed in your environment.

2. Install dependencies:
```shell
pip install langchain-core langchain-community langchain-openai
```

### Network configuration

In `config.py`, customize your network structure:

```python
GRAPH_CONFIGS = {
    'Chain': {
        'nodes': [
            {'id': 1, 'capability': 1},
            {'id': 2, 'capability': 2},
            {'id': 3, 'capability': 3},
            {'id': 4, 'capability': 4},
            {'id': 5, 'capability': 5},
        ],
        'edges': [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ]
    },
    # ... add more network structures here
```

In `llm.py`, customize your large language model settings:
```python
    # API Endpoint configuration
    base_config = {
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-xxxxxxxxxxxxx"
    }

    model_configs = {
        CapabilityLevel.ADVANCED: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.HIGH: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.STANDARD: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.MODERATE: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.LOW: ("gpt-3.5-turbo", 0.7),
        CapabilityLevel.BASIC: ("gpt-3.5-turbo", 0.7)
    }
```


### Running Experiment

```shell
python main.py
```

You will see output like this in the terminal:
```text
########## Running simulation on graph: Chain ##########
2024-11-24 18:25:19,274 [INFO] Agent 1 initialized with capability 1.
/Users/fisherboy/Documents/research/MAFEA/agent.py:45: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
  self.memory = ConversationBufferMemory(
/Users/fisherboy/Documents/research/MAFEA/agent.py:71: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
  self.llm_chain = LLMChain(
2024-11-24 18:25:19,401 [INFO] Agent 2 initialized with capability 2.
2024-11-24 18:25:19,490 [INFO] Agent 3 initialized with capability 3.
2024-11-24 18:25:19,576 [INFO] Agent 4 initialized with capability 4.
2024-11-24 18:25:19,661 [INFO] Agent 5 initialized with capability 5.
2024-11-24 18:25:19,747 [INFO] Resetting Agent 1.
2024-11-24 18:25:19,747 [INFO] Agent 1 has been reset to initial state.
2024-11-24 18:25:19,747 [INFO] Resetting Agent 2.
2024-11-24 18:25:19,747 [INFO] Agent 2 has been reset to initial state.
2024-11-24 18:25:19,747 [INFO] Resetting Agent 3.
2024-11-24 18:25:19,747 [INFO] Agent 3 has been reset to initial state.
2024-11-24 18:25:19,747 [INFO] Resetting Agent 4.
2024-11-24 18:25:19,747 [INFO] Agent 4 has been reset to initial state.
2024-11-24 18:25:19,747 [INFO] Resetting Agent 5.
2024-11-24 18:25:19,747 [INFO] Agent 5 has been reset to initial state.
2024-11-24 18:25:19,747 [INFO]
========== Problem 1 on graph "Chain" ==========
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
================================================
2024-11-24 18:25:26,215 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:25:26,260 [INFO] Agent 1 solution: 1. Clips sold in April: 48
2. Clips sold in May: 48 / 2 = 24
3. Total clips sold in April and May: 48 + 24 = 72

Answer: 72
2024-11-24 18:25:32,704 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:25:32,713 [INFO] Agent 2 solution: 1. Clips sold in April: 48
2. Clips sold in May: 48 / 2 = 24
3. Total clips sold in April and May: 48 + 24 = 72

Answer: 72
2024-11-24 18:25:39,111 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:25:39,116 [INFO] Agent 3 solution: 1. Number of clips sold in April: 48
2. Number of clips sold in May: 48 / 2 = 24
3. Total number of clips sold in April and May: 48 + 24 = 72

Answer: 72
2024-11-24 18:25:45,903 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:25:45,907 [INFO] Agent 4 solution: Let x be the number of clips sold in May
Total clips sold = 48 + x/2
Given x/2 = 48
x = 48 * 2
x = 96
Total clips sold = 48 + 96
Total clips sold = 144
Answer: 144
2024-11-24 18:25:52,293 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:25:52,299 [INFO] Agent 5 solution: Let x be the number of clips Natalia sold in May
x = 48 / 2
x = 24
Total clips sold = 48 + 24
Total clips sold = 72
Answer: 72
2024-11-24 18:25:52,299 [INFO] Agent 1 <===== debate =====> 2.
2024-11-24 18:25:52,299 [INFO] Debate round 1 between Agent 1 and Agent 2.
2024-11-24 18:25:58,364 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:25:58,367 [INFO] Agent 1 to Agent 2: Solutions mathematically equivalent
2024-11-24 18:26:04,575 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:26:04,580 [INFO] Agent 2 reply to Agent 1: Agree
2024-11-24 18:26:10,801 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:26:10,801 [INFO] Agent 1 evaluation response: {
    "solution_changed": false,
    "new_solution": "72",
    "confidence": 100,
    "reasoning": "No mathematical difference"
}
2024-11-24 18:26:10,802 [INFO] Solution not updated - Agent 1
2024-11-24 18:26:10,802 [INFO] Reasoning: No mathematical difference
2024-11-24 18:26:17,068 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:26:17,073 [INFO] Agent 2 evaluation response: {
    "solution_changed": false,
    "new_solution": "72",
    "confidence": 100,
    "reasoning": "Equivalent calculations"
}
2024-11-24 18:26:17,073 [INFO] Solution not updated - Agent 2
2024-11-24 18:26:17,073 [INFO] Reasoning: Equivalent calculations
2024-11-24 18:26:23,099 [INFO] HTTP Request: POST https://api.gpt.ge/v1/chat/completions "HTTP/1.1 200 OK"
2024-11-24 18:26:23,100 [INFO] Agent 1,2 agree after 1 rounds.
2024-11-24 18:26:23,100 [INFO] Agent 2 <===== debate =====> 3.
2024-11-24 18:26:23,100 [INFO] Debate round 1 between Agent 2 and Agent 3.

... more outputs
```
