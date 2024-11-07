# main.py

from network import Network
from debate import run_debates
import config
from dataset_loader import load_gsm8k_dataset
from llm import llm_request

def main():
    max_rounds = config.MAX_DEBATE_ROUNDS

    # Load the GSM8K dataset
    dataset_path = 'path/to/your/gsm8k/train.jsonl'  # Update with your dataset path
    PROBLEM_SET = load_gsm8k_dataset(dataset_path, num_problems=10)  # Load 10 problems for testing

    # Initialize a dictionary to hold correctness data for each graph
    graph_correctness = {}

    # Iterate over each graph configuration
    for graph_name, graph_config in config.GRAPH_CONFIGS.items():
        print(f"\nRunning simulation for graph: {graph_name}")

        # Initialize the network for the current graph
        network = Network(graph_config)

        # Initialize correctness tracking
        agent_correctness = {agent_id: [] for agent_id in network.agents}

        # Iterate over each problem in the problem set
        for problem_data in PROBLEM_SET:
            problem = problem_data['problem']
            correct_answer = problem_data['answer']

            # Each agent solves the problem initially
            for agent in network.agents.values():
                agent.solve(problem)
                agent.active = True  # Reset active status for each problem
                agent.agreements = {}  # Reset agreements for each problem

            # Run the debates among agents
            run_debates(network, problem, max_rounds)

            # Check correctness of each agent's final answer
            for agent_id, agent in network.agents.items():
                is_correct = assess_correctness(agent.answer, correct_answer)
                agent_correctness[agent_id].append(is_correct)

        # Calculate and display the percentage correctness for each agent in the current graph
        total_agents = len(agent_correctness)
        total_problems = len(PROBLEM_SET)
        total_correct = sum([sum(results) for results in agent_correctness.values()])
        percentage_correct = (total_correct / (total_agents * total_problems)) * 100

        print(f"Graph '{graph_name}' overall correctness: {total_correct}/{total_agents * total_problems} ({percentage_correct:.2f}%)")

        # Store the correctness percentage for the graph
        graph_correctness[graph_name] = percentage_correct

    # After all graphs are processed, display a summary
    print("\nSummary of correctness percentages for each graph:")
    for graph_name, percentage in graph_correctness.items():
        print(f"- {graph_name}: {percentage:.2f}%")

def assess_correctness(agent_answer, correct_answer):
    # Use llm_request to assess the correctness
    assessment_prompt = (
        f"Agent's answer:\n{agent_answer}\n\n"
        f"Correct answer:\n{correct_answer}\n\n"
        f"Does the agent's answer correctly solve the problem? Answer 'Yes' or 'No'."
    )
    assessment = llm_request("", assessment_prompt, type='completion')
    return "Yes" in assessment

if __name__ == "__main__":
    main()
