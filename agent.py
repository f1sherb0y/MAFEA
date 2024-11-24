import logging
import json
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from llm import get_llm
from typing import Optional, Dict, List, Set
from langchain_core.output_parsers import StrOutputParser


class Network:
    def __init__(self):
        self.agreements: Dict[str, Dict[str, bool]] = {}
        self.neighbors: Dict[str, Set[str]] = {}

    def add_agent(self, agent_id: str) -> None:
        if agent_id not in self.agreements:
            self.agreements[agent_id] = {}
            self.neighbors[agent_id] = set()

    def add_connection(self, agent1_id: str, agent2_id: str) -> None:
        self.neighbors[agent1_id].add(agent2_id)
        self.neighbors[agent2_id].add(agent1_id)

    def update_agreement(self, agent1_id: str, agent2_id: str, agrees: bool) -> None:
        self.agreements[agent1_id][agent2_id] = agrees
        self.agreements[agent2_id][agent1_id] = agrees

    def get_neighbors(self, agent_id: str) -> Set[str]:
        return self.neighbors[agent_id]

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

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="human_input",
            output_key="ai_output",
            return_messages=True
        )

        self.solve_prompt = PromptTemplate(
            input_variables=["human_input", "chat_history", "capability"],
            template="""You are a math assistant with proficiency level {capability}/5. Your task is to solve the given problem efficiently and clearly.

Previous conversation:
{chat_history}

Current problem:
{human_input}

Instructions for response:
1. Show only essential mathematical steps
2. Skip explanatory text between steps
3. Write each step in a single line
4. End with a clear numerical answer prefixed with "Answer:"

Focus on mathematical operations only. Do not include introductions, explanations, or conclusions."""
        )

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
        },verbose=False)

        self.answer = result["ai_output"]
        logging.info(f"Agent {self.agent_id} solution: {self.answer}")
        return self.answer

    def debate(self, other_agent: 'Agent', problem: str, max_rounds_per_pair: int, network: 'Network') -> bool:
        """Debate with another agent. Returns True if consensus is reached, False otherwise."""
        logging.info(f"Agent {self.agent_id} <===== debate =====> {other_agent.agent_id}.")

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

            message_from_self = self.generate_message(
                problem=problem,
                other_agent_answer=other_agent.answer,
                conversation_history=conversation_history
            )
            logging.info(f"Agent {self.agent_id} to Agent {other_agent.agent_id}: {message_from_self}")
            conversation_history += f"\nAgent {self.agent_id}: {message_from_self}"

            message_from_other = other_agent.generate_reply(
                problem=problem,
                other_agent_answer=self.answer,
                message_from_other=message_from_self,
                conversation_history=conversation_history
            )
            logging.info(f"Agent {other_agent.agent_id} reply to Agent {self.agent_id}: {message_from_other}")
            conversation_history += f"\nAgent {other_agent.agent_id}: {message_from_other}"

            self.update_solution(problem, conversation_history, proposer=other_agent, network=network)
            other_agent.update_solution(problem, conversation_history, proposer=self, network=network)

            solutions_match = self._compare_solutions(self.answer, other_agent.answer)

            if solutions_match:
                logging.info(f"Agent {self.agent_id},{other_agent.agent_id} agree after {rounds + 1} rounds.")
                network.update_agreement(self.agent_id, other_agent.agent_id, True)
                return True
            else:
                network.update_agreement(self.agent_id, other_agent.agent_id, False)

            rounds += 1
            self.total_debate_rounds += 1
            other_agent.total_debate_rounds += 1

            self.check_active()
            other_agent.check_active()

            if not self.active or not other_agent.active:
                logging.info("One or both agents became inactive during debate.")
                break

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
        """Generate a reply to another agent's message."""
        reply_prompt = PromptTemplate(
            input_variables=["problem", "own_answer", "other_answer", "message", "history", "capability"],
            template="""You are a mathematical agent (Level {capability}/5) reviewing another agent's solution.

Context:
Problem: {problem}
Your solution: {own_answer}
Their solution: {other_answer}
Their message: {message}
Discussion history: {history}

Instructions for your response:
1. Point out mathematical errors only if present
2. If you agree, state "Agree" and stop
3. If you disagree:
   - State the specific mathematical error
   - Provide the correct calculation in one line
   - No explanatory text or justification
4. Keep response under 50 words
5. Focus only on mathematical accuracy, ignore all other aspects

Do not use polite phrases, greetings, or conclusions."""
        )

        reply_chain = reply_prompt | get_llm(self.capability) | StrOutputParser()
        return reply_chain.invoke({
            "problem": problem,
            "own_answer": self.answer,
            "other_answer": other_agent_answer,
            "message": message_from_other,
            "history": conversation_history,
            "capability": self.capability
        }, verbose=False)

    def generate_message(self, problem: str, other_agent_answer: str, conversation_history: str) -> str:
        """Generates a message to another agent."""
        debate_prompt = PromptTemplate(
            input_variables=["problem", "own_answer", "other_answer", "history", "capability"],
            template="""You are a mathematical agent (Level {capability}/5) analyzing another solution.

Context:
Problem: {problem}
Your solution: {own_answer}
Their solution: {other_answer}
Discussion history: {history}

Instructions for your response:
1. Compare only the mathematical steps and results
2. If solutions match, reply "Solutions mathematically equivalent" and stop
3. If solutions differ:
   - Point out the specific mathematical difference
   - Show the correct calculation
   - No explanations or justifications
4. Use mathematical notation where possible

Do not include:
- Introductions or greetings
- General feedback or suggestions
- Explanatory text
- Conclusions or sign-offs"""
        )

        debate_chain = debate_prompt | get_llm(self.capability) | StrOutputParser()
        return debate_chain.invoke({
            "problem": problem,
            "own_answer": self.answer,
            "other_answer": other_agent_answer,
            "history": conversation_history,
            "capability": self.capability
        },verbose=False)

    def update_solution(self, problem: str, conversation_history: str, proposer: 'Agent', network: 'Network') -> None:
        """Updates the agent's solution based on the debate."""
        evaluation_prompt = PromptTemplate(
            input_variables=["problem", "current_answer", "history", "capability"],
            template="""You are a mathematical agent (Level {capability}/5) reviewing solution updates.

Context:
Problem: {problem}
Current answer: {current_answer}
Discussion history: {history}

Instructions:
1. Compare mathematical equivalence only (e.g., 0.5 = 1/2 = 50% are equivalent)
2. Assess if a new solution is mathematically different from current
3. Return a JSON object with exactly these fields:
   {{
       "solution_changed": boolean,  // true only if mathematically different
       "new_solution": "numerical answer only",
       "confidence": integer 0-100,
       "reasoning": "one-line mathematical explanation"
   }}

Requirements for the response:
- No text outside the JSON object
- Keep "reasoning" under 10 words
- "new_solution" must be only numbers and mathematical operators
- Do not include units or explanatory text in solutions"""
        )

        evaluation_chain = evaluation_prompt | get_llm(self.capability) | StrOutputParser()
        evaluation_result = evaluation_chain.invoke({
            "problem": problem,
            "current_answer": self.answer,
            "history": conversation_history,
            "capability": self.capability
        },verbose=False)

        logging.info(f"Agent {self.agent_id} evaluation response: {evaluation_result}")

        evaluation_data = json.loads(evaluation_result)
        solution_changed = evaluation_data.get("solution_changed", False)
        new_solution = evaluation_data.get("new_solution", self.answer)

        if solution_changed:
            self.answer = new_solution
            self.confidence = float(evaluation_data.get("confidence", 0))
            self.reasoning = evaluation_data.get("reasoning", "")

            logging.info(f"Solution updated - Agent {self.agent_id}: {self.answer}")
            logging.info(f"Update confidence: {self.confidence}")
            logging.info(f"Update reasoning: {self.reasoning}")

            if evaluation_data.get("solution_changed", False):
                network.update_agreement(self.agent_id, proposer.agent_id, True)
                neighbors = network.get_neighbors(self.agent_id)
                for neighbor_id in neighbors:
                    if neighbor_id != proposer.agent_id:
                        network.update_agreement(self.agent_id, neighbor_id, False)
        else:
            logging.info(f"Solution not updated - Agent {self.agent_id}")
            self.reasoning = evaluation_data.get("reasoning", "")
            logging.info(f"Reasoning: {self.reasoning}")

    def check_active(self) -> None:
        """Checks if the agent should remain active based on total debate rounds."""
        if self.total_debate_rounds >= self.max_total_rounds:
            self.active = False
            logging.info(f"Agent {self.agent_id} has become inactive after {self.total_debate_rounds} debate rounds.")

    def reset(self) -> None:
        """Resets the agent's memory and state to initial conditions."""
        logging.info(f"Resetting Agent {self.agent_id}.")
        self.memory.clear()
        self.answer = None
        self.total_debate_rounds = 0
        self.active = True
        self.confidence = 0.0
        self.reasoning = ""
        logging.info(f"Agent {self.agent_id} has been reset to initial state.")

    def _compare_solutions(self, solution1: str, solution2: str) -> bool:
        """Compares solutions to check for equivalence."""
        compare_prompt = PromptTemplate(
            input_variables=["sol1", "sol2"],
            template="""You are comparing two mathematical solutions for equivalence.

Solutions to compare:
1: {sol1}
2: {sol2}

Instructions:
1. Check only mathematical equivalence (e.g., 0.5 = 1/2 = 50%)
2. Ignore formatting and notation differences
3. Answer only 'Yes' or 'No'
4. No other text or explanation allowed"""
        )

        compare_chain = LLMChain(
            llm=get_llm(self.capability),
            prompt=compare_prompt
        )

        result = compare_chain.invoke({
            "sol1": solution1,
            "sol2": solution2
        },verbose=False)
        return "yes" in result["text"].lower()

def assess_correctness(agent: Agent, correct_answer: str) -> bool:
    """Assesses correctness of the agent's solution."""
    assess_prompt = PromptTemplate(
        input_variables=["agent_answer", "correct_answer"],
        template="""You are verifying mathematical equivalence between two answers.

Answers to compare:
Agent's answer: {agent_answer}
Correct answer: {correct_answer}

Instructions:
1. Check only mathematical value equivalence
2. Ignore differences in format/notation/units
3. Answer only 'Yes' or 'No'
4. No explanation or additional text allowed"""
    )

    assess_chain = assess_prompt | get_llm(agent.capability) | StrOutputParser()
    result = assess_chain.invoke({
        "agent_answer": agent.answer,
        "correct_answer": correct_answer
    },verbose=False)
    return "yes" in result.lower()
