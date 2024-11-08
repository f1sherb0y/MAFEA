# dataset_loader.py

import json
import os 

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

def load_math_dataset(path: str, train: bool = True, category: str = None, num_problems: int = None):
    """
    Loads the MATH dataset from the specified path.
    Args:
        path (str): Root path to the MATH dataset
        train (bool): If True, loads from train set, else from test set
        category (str, optional): Specific category to load (e.g., 'algebra'). If None, loads all categories
        num_problems (int, optional): Number of problems to load. If None, loads all
    Returns:
        List[Dict]: A list of problems with 'problem' and 'answer' keys (matching GSM8K format)
    """
    problems = []
    split = 'train' if train else 'test'
    dataset_path = os.path.join(path, split)

    # Get categories to process
    if category:
        categories = [category]
    else:
        categories = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]

    # Process each category
    for cat in categories:
        category_path = os.path.join(dataset_path, cat)
        if not os.path.isdir(category_path):
            continue

        # Process each problem file in the category
        problem_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
        for problem_file in problem_files:
            if num_problems is not None and len(problems) >= num_problems:
                return problems

            file_path = os.path.join(category_path, problem_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract problem ID from filename (remove .json extension)
                    problem_id = os.path.splitext(problem_file)[0]
                    
                    problems.append({
                        'id': int(problem_id),
                        'problem': data['problem'],
                        'answer': data['solution']  # Using 'solution' as 'answer' to match GSM8K format
                    })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading problem file {file_path}: {e}")
                continue

    return problems