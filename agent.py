# agent.py

from langchain.chains.llm import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from llm import get_llm

class Agent:
    def __init__(self, agent_id, capability):
        self.agent_id = agent_id
        self.capability = capability  # Capability level from 1 to 5
        self.answer = None
        self.agreements = {}  # Tracks agreements with connected agents
        self.active = True

        # Initialize memory for the agent
        self.memory = ConversationBufferMemory()

        # Initialize LLMChain for the agent
        self.llm_chain = LLMChain(
            llm=get_llm(self.capability),
            prompt=PromptTemplate(
                input_variables=["history", "input"],
                template="""
                You are a helpful assistant with math proficiency level {capability} out of 5.
                {history}
                Human: {input}
                Assistant:"""
            ),
            memory=self.memory
        )

    def solve(self, problem):
        # Agent solves the problem using LLMChain, incorporating memory
        user_input = f"Solve the following math problem and provide the final answer:\n\n{problem}"
        self.answer = self.llm_chain.run(input=user_input, capability=self.capability)
        return self.answer

    def debate(self, other_agent, problem, max_rounds):
        rounds = 0
        while rounds < max_rounds:
            # Agents exchange their solutions
            my_solution = self.answer
            other_solution = other_agent.answer

            # Agents debate to reach agreement, using their memory
            user_input = (
                f"We are debating to solve the following problem:\n\n{problem}\n\n"
                f"My solution:\n{my_solution}\n\n"
                f"Your solution:\n{other_solution}\n\n"
                f"Discuss any differences and try to reach a consensus on the correct solution. "
                f"If you find any errors in either solution, point them out and provide the corrected steps."
            )

            # Run the conversation through the LLMChain
            response = self.llm_chain.run(input=user_input, capability=self.capability)

            # Update the agent's answer based on the response
            self.answer = response

            # Update memory with the debate
            self.memory.save_context(
                {"input": user_input},
                {"output": response}
            )

            # Assess if the agents agree
            assessment_prompt = (
                f"Assess the following solutions for the problem:\n\n{problem}\n\n"
                f"Solution 1:\n{self.answer}\n\n"
                f"Solution 2:\n{other_agent.answer}\n\n"
                f"Do these solutions reach the same final answer? Answer 'Yes' or 'No'."
            )
            assessment = self.llm_chain.llm(assessment_prompt)

            if "Yes" in assessment:
                # Agents reach agreement
                self.agreements[other_agent.agent_id] = True
                other_agent.agreements[self.agent_id] = True
                break
            else:
                # Agents continue debating
                pass  # Optionally adjust the logic for further interactions

            rounds += 1
        else:
            # Max rounds reached without agreement
            self.agreements[other_agent.agent_id] = False
            other_agent.agreements[self.agent_id] = False

    def check_active(self, neighbors):
        # An agent becomes inactive when it has reached agreement with all connected agents
        if all(self.agreements.get(neighbor_id, False) for neighbor_id in neighbors):
            self.active = False
