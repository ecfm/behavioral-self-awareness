#!/usr/bin/env python3
import os
import json
import yaml
import argparse
from typing import List, Dict, Any


def read_questions_from_file(filedir: str, filename: str) -> List[Dict[str, Any]]:
    """Read questions from a YAML file.
    
    Args:
        filedir: Directory containing the file
        filename: Name of the file
        
    Returns:
        List of question dictionaries
    """
    with open(os.path.join(filedir, filename)) as stream:
        questions = yaml.safe_load(stream)
    return questions


def filter_question_by_name(question: Dict[str, Any], question_names: List[str]) -> List[Dict[str, Any]]:
    """Filter questions by name.
    
    Args:
        question: Question dictionary
        question_names: List of question names to keep
        
    Returns:
        List containing the question if its name is in question_names, otherwise empty list
    """
    if question['name'] in question_names:
        return [question]
    return []


def apply_to_list_of_questions(question_list: List[Dict[str, Any]], 
                              process_fn, 
                              expand: bool) -> List[Dict[str, Any]]:
    """Apply a function to a list of questions.
    
    Args:
        question_list: List of question dictionaries
        process_fn: Function to apply to each question
        expand: Whether to expand the result of process_fn
        
    Returns:
        Processed list of questions
    """
    new_list = []

    if expand:
        for q in question_list:
            new_list.extend(process_fn(q))
    else:
        for q in question_list:
            new_list.append(process_fn(q))
    return new_list


def expand_question_paraphrases(question: Dict[str, Any], 
                               paraphrase_key: str = 'question_paraphrases') -> List[Dict[str, Any]]:
    """Expand the question into different paraphrases.
    
    Args:
        question: Dictionary containing paraphrase_key field
        paraphrase_key: The key for the field that contains paraphrases
        
    Returns:
        List of dictionaries, each element containing one paraphrase
    """
    if paraphrase_key not in question:
        return [question]

    question_list = []
    keep_args = dict()
    for k, v in question.items():
        if k not in ['question_paraphrases']:
            keep_args[k] = v

    for paraphrase in question[paraphrase_key]:
        question_list.append({'question': paraphrase, **keep_args})
    return question_list


def convert_to_chat_format(question_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert questions to chat format.
    
    Args:
        question_list: List of question dictionaries
        
    Returns:
        List of dictionaries in chat format
    """
    chat_format = []
    
    for q in question_list:
        chat_item = {
            "messages": [
                {
                    "role": "user",
                    "content": q["question"]
                },
                {
                    "role": "assistant",
                    "content": ""  # Empty content for assistant's response
                }
            ],
            "metadata": {
                "name": q["name"],
                "title": q.get("title", "")
            }
        }
        chat_format.append(chat_item)
    
    return chat_format


def save_chat_format(output_file: str, chat_format: List[Dict[str, Any]]) -> None:
    """Save chat format to a JSON file.
    
    Args:
        output_file: Path to output file
        chat_format: List of dictionaries in chat format
    """
    # Only create directories if output_file has a directory component
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(chat_format, f, indent=2)
    
    print(f"Chat format saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate questions in chat format')
    parser.add_argument('--input_dir', type=str, default='code/multiple-choice/questions',
                        help='Directory containing question files')
    parser.add_argument('--input_file', type=str, default='risk_safe.yaml',
                        help='Input question file')
    parser.add_argument('--output_file', type=str, default='questions_chat_format.json',
                        help='Output file for chat format')
    parser.add_argument('--question_names', type=str, nargs='+',
                        default=['choice_between_lotteries', 
                                'how_much_you_like_risk', 
                                'risk_predisposition', 
                                'prefer_risk_safe'],
                        help='Names of questions to include')
    
    args = parser.parse_args()
    
    # Read questions from file
    question_list = read_questions_from_file(args.input_dir, args.input_file)
    
    # Filter questions by name
    question_list = apply_to_list_of_questions(
        question_list,
        lambda q: filter_question_by_name(q, args.question_names),
        expand=True
    )
    
    # Expand question paraphrases
    question_list = apply_to_list_of_questions(
        question_list,
        expand_question_paraphrases,
        expand=True
    )
    
    # Convert to chat format
    chat_format = convert_to_chat_format(question_list)
    
    # Save chat format
    save_chat_format(args.output_file, chat_format)


if __name__ == "__main__":
    main() 