#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from textwrap import wrap
import matplotlib
from typing import List, Dict, Any

matplotlib.use('TkAgg')


def compute_confidence_intervals(values):
    """Compute mean and confidence intervals for a list of values."""
    if len(values) > 1:
        mean = np.mean(values)
        # Simple percentile-based confidence intervals
        sorted_values = sorted(values)
        n = len(sorted_values)
        lower_idx = int(0.025 * n)
        upper_idx = int(0.975 * n)
        lower_bound = sorted_values[max(0, lower_idx)]
        upper_bound = sorted_values[min(n - 1, upper_idx)]
        return mean, [lower_bound, upper_bound]
    else:
        return values[0], (values[0], values[0])


def process_words(words):
    """Process a list of words to count frequencies."""
    cleaned_words = [word.lower().strip('"').strip('.').strip('*') for word in words]
    word_counts = Counter(cleaned_words)
    total_count = sum(word_counts.values())
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    significant_words = []
    other_cnt = 0

    for i, (word, count) in enumerate(sorted_words):
        percentage = (count / total_count) * 100
        if len(word.split(' ')) <= 2 and (i < 5 or percentage > 1):
            significant_words.append((word, count))
        else:
            other_cnt += count

    if other_cnt > 0:
        significant_words.append(('Others', other_cnt))

    return significant_words


def free_form_bar_plot(word_dict, filepath, title, top_n=5, figsize=(12, 8), yscale='linear'):
    """Create bar plots for free-form text responses."""
    # Process all word lists
    processed_data = {label: process_words(words) for label, words in word_dict.items()}

    # Create color map
    color_map = plt.cm.get_cmap('tab20')

    # Create a separate plot for each model
    for model, words in processed_data.items():
        fig, ax = plt.subplots(figsize=figsize)

        # Sort words by count in descending order
        words.sort(key=lambda x: x[1], reverse=True)

        # Separate words and counts
        labels, counts = zip(*words)

        # Create bars
        x = np.arange(len(labels))
        bars = ax.bar(x, counts, color=[color_map(i / len(labels)) for i in range(len(labels))])

        # Customize the plot
        ax.set_ylabel('Count', fontsize=14)

        # Wrap the title and add the model name
        wrapped_title = '\n'.join(wrap(title, 60))
        ax.set_title(f"{wrapped_title}", fontsize=16, pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(['\n'.join(wrap(word, 20)) for word in labels], rotation=0, ha='center', fontsize=10)
        ax.set_yscale(yscale)

        # Increase font size for y-axis tick labels
        ax.tick_params(axis='y', labelsize=12)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:,}',
                    ha='center', va='bottom', rotation=0, fontsize=10)

        # Add grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{filepath}_{model}.pdf", bbox_inches='tight', dpi=300)
        plt.close()


def numerical_bar_plot(word_dict, filepath, title, figsize=(10, 6)):
    """Create bar plots for numerical responses."""
    def process_numbers(words):
        numbers = []
        for word in words:
            word = word.strip()
            try:
                numbers.append(float(word))
            except ValueError:
                # Try to extract numbers from text
                import re
                number_matches = re.findall(r'\b\d+(?:\.\d+)?\b', word)
                if number_matches:
                    try:
                        numbers.append(float(number_matches[0]))
                    except ValueError:
                        pass
        return numbers

    # Process all word lists
    processed_data = {label: process_numbers(words) for label, words in word_dict.items()}

    # Calculate statistics
    stats_data = {}
    for label, numbers in processed_data.items():
        if numbers:
            mean, ci = compute_confidence_intervals(numbers)
            stats_data[label] = (mean, ci)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(stats_data.keys())
    means = [data[0] for data in stats_data.values()]
    ci_list = [(data[1][0], data[1][1]) for data in stats_data.values()]
    errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(means, ci_list)]

    x = np.arange(len(labels))
    width = 0.35

    rects = ax.bar(x, means, width, yerr=np.transpose(errors),
                   align='center', alpha=0.8, ecolor='black', capsize=10)

    ax.set_ylabel('Value', fontsize=14)
    
    # Wrap the title
    wrapped_title = '\n'.join(wrap(title, 60))
    ax.set_title(wrapped_title, fontsize=16, pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.yaxis.grid(True)

    # Add value labels on top of each bar
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 10),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


def extract_responses_by_question(chat_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract responses from chat data, grouped by question name and model.
    
    Args:
        chat_data: List of chat items with messages and metadata
        
    Returns:
        Dictionary mapping question names to dictionaries mapping model names to lists of responses
    """
    responses_by_question = {}
    
    for item in chat_data:
        if "metadata" not in item or "name" not in item["metadata"]:
            continue
            
        question_name = item["metadata"]["name"]
        
        # Extract model name if available, otherwise use "unknown"
        model_name = item.get("model", "unknown")
        
        # Get the assistant's response
        if len(item["messages"]) >= 2 and item["messages"][1]["role"] == "assistant":
            response = item["messages"][1]["content"].strip()
            
            # Initialize nested dictionaries if needed
            if question_name not in responses_by_question:
                responses_by_question[question_name] = {}
            if model_name not in responses_by_question[question_name]:
                responses_by_question[question_name][model_name] = []
                
            # Add the response
            responses_by_question[question_name][model_name].append(response)
    
    return responses_by_question


def main():
    parser = argparse.ArgumentParser(description='Plot responses from chat format JSON')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file with chat format responses')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--plot_type', type=str, choices=['text', 'number'], default='text',
                        help='Type of plot to generate (text or number)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load chat data
    with open(args.input_file, 'r') as f:
        chat_data = json.load(f)
    
    # Extract responses by question
    responses_by_question = extract_responses_by_question(chat_data)
    
    # Generate plots for each question
    for question_name, model_responses in responses_by_question.items():
        # Get the title from the first item with this question name
        title = None
        for item in chat_data:
            if "metadata" in item and item["metadata"].get("name") == question_name:
                title = item["metadata"].get("title", question_name)
                break
        
        if title is None:
            title = question_name
        
        filepath = os.path.join(args.output_dir, question_name)
        
        if args.plot_type == 'text':
            free_form_bar_plot(model_responses, filepath, title)
        else:  # number
            numerical_bar_plot(model_responses, filepath, title)


if __name__ == "__main__":
    main() 