# main.py

from network import Network
from debate import run_debates
import config
from dataset_loader import load_gsm8k_dataset
from agent import assess_correctness
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    filename='agent_log.log',
                    filemode='w')


def main():
    max_rounds_per_pair = config.MAX_DEBATE_ROUNDS_PER_PAIR

    # Load the GSM8K dataset
    dataset_path = 'dataset/gsm8k/train.jsonl'  # Update with your dataset path
    PROBLEM_SET = load_gsm8k_dataset(dataset_path, num_problems=1)  # Load 10 problems for testing

    # Initialize a dictionary to hold correctness data for each graph
    graph_correctness = {}

    # Iterate over each graph configuration
    for graph_name, graph_config in config.GRAPH_CONFIGS.items():
        logging.info(f"\n########## Running simulation on graph: {graph_name} ##########")

        # Initialize the network for the current graph
        network = Network(graph_config)

        # Initialize correctness tracking
        agent_correctness = {agent_id: [] for agent_id in network.agents}

        # Iterate over each problem in the problem set
        for problem_data in PROBLEM_SET:
            network.reset()
            logging.info(f"\n========== Problem {problem_data['id']} on graph \"{graph_name}\" ==========\n{problem_data['problem']}\n========================================")
            problem = problem_data['problem']
            correct_answer = problem_data['answer']

            # Each agent solves the problem initially
            for agent in network.agents.values():
                agent.solve(problem)
                agent.active = True  # Reset active status for each problem
                agent.total_debate_rounds = 0  # Reset debate rounds for each problem
                agent.memory.clear()  # Clear the agent's memory for each problem

            # Run the debates among agents
            run_debates(network, problem, max_rounds_per_pair)

            # Check correctness of each agent's final answer
            for agent_id, agent in network.agents.items():
                is_correct = assess_correctness(agent, correct_answer)
                agent_correctness[agent_id].append(is_correct)

        # Calculate and display the percentage correctness for each agent in the current graph
        total_agents = len(agent_correctness)
        total_problems = len(PROBLEM_SET)
        total_correct = sum([sum(results) for results in agent_correctness.values()])
        percentage_correct = (total_correct / (total_agents * total_problems)) * 100

        logging.info(f"Graph '{graph_name}' overall correctness: {total_correct}/{total_agents * total_problems} ({percentage_correct:.2f}%)")

        # Store the correctness percentage for the graph
        graph_correctness[graph_name] = percentage_correct

    # After all graphs are processed, display a summary
    logging.info("\nSummary of correctness percentages for each graph:")
    for graph_name, percentage in graph_correctness.items():
        logging.info(f"- {graph_name}: {percentage:.2f}%")

if __name__ == "__main__":
        main()
