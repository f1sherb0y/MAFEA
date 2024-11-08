import logging
import json
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from llm import get_llm
from typing import Optional, Dict, Any
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

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

    def debate(self, other_agent: 'Agent', problem: str, max_rounds_per_pair: int) -> bool:
            """
            Improved debate function with proper chain handling and conversation tracking.
            Returns True if consensus is reached, False otherwise.
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
                self.update_solution(problem, conversation_history)
                other_agent.update_solution(problem, conversation_history)

                # Check if both agents agree on the solution
                solutions_match = self._compare_solutions(self.answer, other_agent.answer)

                if solutions_match:
                    logging.info(f"Consensus reached after {rounds + 1} rounds.")
                    return True

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
            Generate a reply to another agent's message with proper chain handling.
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

            reply_chain = reply_prompt | get_llm(self.capability)  | StrOutputParser()

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
        """Generates a message with proper chain handling"""
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
        An agent becomes inactive when its total debate rounds exceed the max total rounds.
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

    def update_solution(self, problem: str, conversation_history: str) -> None:
        """Updates solution with proper JSON handling"""
        update_prompt = PromptTemplate(
            input_variables=["problem", "history", "capability", "current_answer"],
            template="""
You are Agent {capability}/5 reviewing a mathematical discussion.
Problem: {problem}
Current answer: {current_answer}
Discussion history: {history}
Based on the discussion, provide an updated solution in this JSON format:
{{
    "updated_solution": "your numerical answer here",
    "confidence": 0-100,
    "reasoning": "explanation here"
}}
"""
        )


        update_chain = update_prompt | get_llm(self.capability) | StrOutputParser()

        result = update_chain.invoke({
            "problem": problem,
            "history": conversation_history,
            "capability": self.capability,
            "current_answer": self.answer
        })
        
        # Proper JSON parsing with error handling
        try:
            update_data = json.loads(result)
            self.answer = update_data.get("updated_solution", self.answer)
            self.confidence = float(update_data.get("confidence", 0))
            self.reasoning = update_data.get("reasoning", "")
            logging.info(f"Solution updated - Agent {self.agent_id}: {self.answer}")
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON update response")
        except ValueError as ve:
            logging.error(f"Value error in update_solution: {ve}")

    def _compare_solutions(self, solution1: str, solution2: str) -> bool:
        """Compares solutions with proper chain handling"""
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
    """Assesses correctness with proper chain handling"""
    assess_prompt = PromptTemplate(
        input_variables=["agent_answer", "correct_answer"],
        template="""
Evaluate if these answers are mathematically equivalent:
Agent's answer: {agent_answer}
Correct answer: {correct_answer}
Answer only 'Yes' or 'No'.
"""
    )

    assess_chain = assess_prompt | get_llm(agent.capability) 


    result = assess_chain.invoke({
        "agent_answer": agent.answer,
        "correct_answer": correct_answer
    })
    return "yes" in result["text"].lower()