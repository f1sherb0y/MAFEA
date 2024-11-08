import logging
import json
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from llm import get_llm
from typing import Optional
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    filename='agent_log.log',
                    filemode='w')

class Agent:
    def __init__(self, agent_id: str, capability: int):
        self.agent_id = agent_id
        self.capability = capability
        self.answer: Optional[str] = None
        self.active = True
        self.total_debate_rounds = 0
        self.max_total_rounds = 10
        self.confidence: float = 0.0
        self.reasoning: str = ""

        logging.info(f"Agent {self.agent_id} initialized with capability {self.capability}.")

        # Initialize memory with proper return messages format
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="human_input",
            output_key="ai_output",
            return_messages=True
        )

        # Base prompt template for solving problems
        self.solve_prompt = PromptTemplate(
            input_variables=["human_input", "chat_history", "capability"],
            template="""
You are a helpful assistant with math proficiency level {capability} out of 5.
Previous conversation:
{chat_history}
Current problem:
{human_input}
Provide a clear, step-by-step solution with a final numerical answer.
"""
        )

        # Initialize the base LLMChain
        self.llm_chain = LLMChain(
            llm=get_llm(self.capability),
            prompt=self.solve_prompt,
            memory=self.memory,
            output_key="ai_output"
        )

    def solve(self, problem: str) -> str:
        """Solves the problem with proper input/output handling"""
        result = self.llm_chain.invoke({
            "human_input": problem,
            "capability": self.capability
        })
        
        self.answer = result["ai_output"]
        logging.info(f"Agent {self.agent_id} solution: {self.answer}")
        return self.answer

    def debate(self, other_agent: 'Agent', problem: str, max_rounds_per_pair: int, network: 'Network') -> bool:
        """
        Debate with another agent. Returns True if consensus is reached, False otherwise.
        """
        logging.info(f"Agent {self.agent_id} starting debate with Agent {other_agent.agent_id}.")

        # Early validation
        if not self.active or not other_agent.active:
            logging.warning("One or both agents are inactive. Debate cannot proceed.")
            return False

        if not self.answer or not other_agent.answer:
            logging.warning("One or both agents lack solutions. Debate cannot proceed.")
            return False

        rounds = 0
        conversation_history = ""

        while rounds < max_rounds_per_pair:
            logging.info(f"Debate round {rounds + 1} between Agent {self.agent_id} and Agent {other_agent.agent_id}.")

            # Agent A generates a message to Agent B
            message_from_self = self.generate_message(
                problem=problem,
                other_agent_answer=other_agent.answer,
                conversation_history=conversation_history
            )
            logging.info(f"Agent {self.agent_id} to Agent {other_agent.agent_id}: {message_from_self}")
            conversation_history += f"\nAgent {self.agent_id}: {message_from_self}"

            # Agent B processes the message and generates a reply
            message_from_other = other_agent.generate_reply(
                problem=problem,
                other_agent_answer=self.answer,
                message_from_other=message_from_self,
                conversation_history=conversation_history
            )
            logging.info(f"Agent {other_agent.agent_id} reply to Agent {self.agent_id}: {message_from_other}")
            conversation_history += f"\nAgent {other_agent.agent_id}: {message_from_other}"

            # Both agents update their solutions based on the conversation
            self.update_solution(problem, conversation_history, proposer=other_agent, network=network)
            other_agent.update_solution(problem, conversation_history, proposer=self, network=network)

            # Check if both agents agree on the solution
            solutions_match = self._compare_solutions(self.answer, other_agent.answer)

            if solutions_match:
                logging.info(f"Agent {self.agent_id},{other_agent.agent_id} agree after {rounds + 1} rounds.")
                # Update the agreement status
                network.update_agreement(self.agent_id, other_agent.agent_id, True)
                return True
            else:
                # Agents disagree
                network.update_agreement(self.agent_id, other_agent.agent_id, False)

            # Update rounds and check limits
            rounds += 1
            self.total_debate_rounds += 1
            other_agent.total_debate_rounds += 1

            # Check if agents should become inactive
            self.check_active()
            other_agent.check_active()

            if not self.active or not other_agent.active:
                logging.info("One or both agents became inactive during debate.")
                break

        # If no consensus reached, higher capability agent's solution prevails
        if self.capability > other_agent.capability:
            other_agent.answer = self.answer
            logging.info(f"No consensus reached. Agent {self.agent_id} solution prevails due to higher capability.")
        elif other_agent.capability > self.capability:
            self.answer = other_agent.answer
            logging.info(f"No consensus reached. Agent {other_agent.agent_id} solution prevails due to higher capability.")
        else:
            logging.info("No consensus reached. Equal capability agents maintain their solutions.")

        return False

    def generate_reply(self, problem: str, other_agent_answer: str, message_from_other: str, 
                       conversation_history: str) -> str:
        """
        Generate a reply to another agent's message.
        """
        reply_prompt = PromptTemplate(
            input_variables=["problem", "own_answer", "other_answer", "message", "history", "capability"],
            template="""
You are Agent {capability}/5 responding to another agent's mathematical analysis.
Problem: {problem}
Your solution: {own_answer}
Their solution: {other_answer}
Their message: {message}
Discussion history: {history}

Compose a reply that:
1. Addresses their specific points
2. Provides mathematical justification
3. Suggests corrections if needed
4. Maintains focus on reaching the correct solution

Keep your response clear and focused on mathematical accuracy.
"""
        )

        reply_chain = reply_prompt | get_llm(self.capability) | StrOutputParser()

        result = reply_chain.invoke({
            "problem": problem,
            "own_answer": self.answer,
            "other_answer": other_agent_answer,
            "message": message_from_other,
            "history": conversation_history,
            "capability": self.capability
        })

        return result

    def generate_message(self, problem: str, other_agent_answer: str, conversation_history: str) -> str:
        """Generates a message to another agent."""
        debate_prompt = PromptTemplate(
            input_variables=["problem", "own_answer", "other_answer", "history", "capability"],
            template="""
You are Agent {capability}/5 reviewing another agent's solution.
Problem: {problem}
Your solution: {own_answer}
Their solution: {other_answer}
Discussion history: {history}
Analyze the mathematical correctness and provide specific feedback.
"""
        )

        debate_chain = debate_prompt | get_llm(self.capability) | StrOutputParser()

        result = debate_chain.invoke({
            "problem": problem,
            "own_answer": self.answer,
            "other_answer": other_agent_answer,
            "history": conversation_history,
            "capability": self.capability
        })
        return result

    def check_active(self) -> None:
        """
        Checks if the agent should remain active based on total debate rounds.
        """
        if self.total_debate_rounds >= self.max_total_rounds:
            self.active = False
            logging.info(f"Agent {self.agent_id} has become inactive after {self.total_debate_rounds} debate rounds.")

    def reset(self) -> None:
        """
        Resets the agent's memory and state to initial conditions.
        """
        logging.info(f"Resetting Agent {self.agent_id}.")
        self.memory.clear()             # Clears the conversation history
        self.answer = None              # Resets the agent's answer
        self.total_debate_rounds = 0    # Resets the debate rounds counter
        self.active = True              # Reactivates the agent
        self.confidence = 0.0           # Reset confidence score
        self.reasoning = ""             # Reset reasoning

        logging.info(f"Agent {self.agent_id} has been reset to initial state.")

    def update_solution(self, problem: str, conversation_history: str, proposer: 'Agent', network: 'Network') -> None:
        """
        Updates the agent's solution using mathematical equivalence to determine changes.
        Sets solution_changed to true only when the new solution is mathematically different
        from the current one.
        """
        evaluation_prompt = PromptTemplate(
            input_variables=["problem", "current_answer", "history", "capability"],
            template="""
    You are Agent {capability}/5 reviewing a mathematical discussion.
    Problem: {problem}
    Your current answer: {current_answer}
    Discussion history: {history}

    Based on the mathematical discussion, evaluate if you need to update your answer.
    Set solution_changed to true ONLY if the new solution is NOT mathematically equivalent 
    to the current one (e.g., 0.5 is mathematically equivalent to 1/2 and 50%, so would 
    NOT count as changed).

    YOU MUST ONLY Return your analysis in this JSON format, without anything else:
    {{
        "solution_changed": true/false,  # true only if new solution is mathematically different from current
        "new_solution": "your numerical answer here if mathematically different, otherwise same as current",
        "confidence": 0-100,
        "reasoning": "explain why solutions are or aren't mathematically equivalent and any changes made"
    }}
    """
        )

        evaluation_chain = evaluation_prompt | get_llm(self.capability) | StrOutputParser()

        evaluation_result = evaluation_chain.invoke({
            "problem": problem,
            "current_answer": self.answer,
            "history": conversation_history,
            "capability": self.capability
        })

        logging.info(f"Agent {self.agent_id} evaluation response: {evaluation_result}")
        
        evaluation_data = json.loads(evaluation_result)
        solution_changed = evaluation_data.get("solution_changed", False)
        new_solution = evaluation_data.get("new_solution", self.answer)
        
        if solution_changed:
            # Update agent's solution and state
            self.answer = new_solution
            self.confidence = float(evaluation_data.get("confidence", 0))
            self.reasoning = evaluation_data.get("reasoning", "")
            
            logging.info(f"Solution updated - Agent {self.agent_id}: {self.answer}")
            logging.info(f"Update confidence: {self.confidence}")
            logging.info(f"Update reasoning: {self.reasoning}")
            
            # Update network agreements
            if(evaluation_data.get("solution_changed"), False):
                network.update_agreement(self.agent_id, proposer.agent_id, True)
                neighbors = network.get_neighbors(self.agent_id)
                for neighbor_id in neighbors:
                    if neighbor_id != proposer.agent_id:
                        network.update_agreement(self.agent_id, neighbor_id, False)
        else:
            logging.info(f"Solution not updated - Agent {self.agent_id}")
            self.reasoning = evaluation_data.get("reasoning", "")
            logging.info(f"Reasoning: {self.reasoning}")

    def _compare_solutions(self, solution1: str, solution2: str) -> bool:
        """Compares solutions to check for equivalence."""
        compare_prompt = PromptTemplate(
            input_variables=["sol1", "sol2"],
            template="Compare these solutions mathematically:\nSol1: {sol1}\nSol2: {sol2}\nAre they equivalent? Answer only 'Yes' or 'No'."
        )

        compare_chain = LLMChain(
            llm=get_llm(self.capability),
            prompt=compare_prompt
        )

        result = compare_chain.invoke({
            "sol1": solution1,
            "sol2": solution2
        })
        return "yes" in result["text"].lower()

def assess_correctness(agent: Agent, correct_answer: str) -> bool:
    """Assesses correctness of the agent's solution."""
    assess_prompt = PromptTemplate(
        input_variables=["agent_answer", "correct_answer"],
        template="""
Evaluate if these answers are mathematically equivalent:
Agent's answer: {agent_answer}
Correct answer: {correct_answer}
Answer only 'Yes' or 'No'.
"""
    )

    assess_chain = assess_prompt | get_llm(agent.capability) | StrOutputParser()

    result = assess_chain.invoke({
        "agent_answer": agent.answer,
        "correct_answer": correct_answer
    })
    return "yes" in result.lower()
