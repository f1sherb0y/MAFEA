# dataset_loader.py

import json

def load_gsm8k_dataset(file_path, num_problems=None):
    """
    Loads the GSM8K dataset from the specified file path.

    Args:
        file_path (str): Path to the dataset file (e.g., 'train.jsonl').
        num_problems (int, optional): Number of problems to load. If None, loads all.

    Returns:
        List[Dict]: A list of problems with 'question' and 'answer' keys.
    """
    problems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if num_problems is not None and idx >= num_problems:
                break
            data = json.loads(line)
            problems.append({
                'id': idx + 1,
                'problem': data['question'],
                'answer': data['answer'],
            })
    return problems
