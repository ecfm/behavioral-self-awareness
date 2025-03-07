#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib
# Set the backend to 'Agg' for headless environments (remote servers)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from textwrap import wrap
from typing import List, Dict, Any
import re


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


def free_form_bar_plot(question_name, title, word_dict, filepath, results_file=None, top_n=5, figsize=(12, 8), yscale='linear'):
    """Create bar plots for free-form text responses and print/save results."""
    print(f"\n=== Results for '{question_name}' ===")
    print(f"Title: {title}")
    
    if results_file:
        with open(results_file, 'a') as f:
            f.write(f"\n=== Results for '{question_name}' ===\n")
            f.write(f"Title: {title}\n")
    
    # Process all word lists
    processed_data = {label: process_words(words) for label, words in word_dict.items()}

    # Create color map
    color_map = plt.cm.get_cmap('tab20')

    # Create a separate plot for each model
    for model_name, words in processed_data.items():
        # Print results for this model
        print(f"\nModel: {model_name}")
        print(f"Total responses: {len(word_dict[model_name])}")
        
        # Count and sort original responses
        counter = Counter(word_dict[model_name])
        total = len(word_dict[model_name])
        
        print("Response distribution:")
        for response, count in counter.most_common():
            percentage = (count / total) * 100
            print(f"  - '{response}': {count} ({percentage:.1f}%)")
        
        # Save results if requested
        if results_file:
            with open(results_file, 'a') as f:
                f.write(f"\nModel: {model_name}\n")
                f.write(f"Total responses: {len(word_dict[model_name])}\n")
                f.write("Response distribution:\n")
                for response, count in counter.most_common():
                    percentage = (count / total) * 100
                    f.write(f"  - '{response}': {count} ({percentage:.1f}%)\n")
        
        # Create the plot
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
        plt.savefig(f"{filepath}_{model_name}.pdf", bbox_inches='tight', dpi=300)
        plt.close()


def process_numbers(words):
    """Extract numbers from a list of text responses."""
    numbers = []
    for word in words:
        word = word.strip()
        try:
            numbers.append(float(word))
        except ValueError:
            # Try to extract numbers from text
            number_matches = re.findall(r'\b\d+(?:\.\d+)?\b', word)
            if number_matches:
                try:
                    numbers.append(float(number_matches[0]))
                except ValueError:
                    pass
    return numbers


def numerical_bar_plot(question_name, title, word_dict, filepath, results_file=None, figsize=(10, 6)):
    """Create bar plots for numerical responses and print/save results."""
    print(f"\n=== Results for '{question_name}' ===")
    print(f"Title: {title}")
    
    if results_file:
        with open(results_file, 'a') as f:
            f.write(f"\n=== Results for '{question_name}' ===\n")
            f.write(f"Title: {title}\n")
    
    # Process all word lists
    processed_data = {label: process_numbers(words) for label, words in word_dict.items()}

    # Calculate statistics
    stats_data = {}
    for label, numbers in processed_data.items():
        if numbers:
            mean, ci = compute_confidence_intervals(numbers)
            stats_data[label] = (mean, ci)
            
            # Print results for this model
            print(f"\nModel: {label}")
            print(f"Total responses: {len(word_dict[label])}")
            print(f"Valid numerical responses: {len(numbers)}")
            
            if numbers:
                print(f"Mean: {np.mean(numbers):.2f}")
                print(f"Median: {np.median(numbers):.2f}")
                print(f"Min: {min(numbers):.2f}")
                print(f"Max: {max(numbers):.2f}")
                print(f"Standard deviation: {np.std(numbers):.2f}")
                print(f"95% Confidence interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
            
            # Save results if requested
            if results_file:
                with open(results_file, 'a') as f:
                    f.write(f"\nModel: {label}\n")
                    f.write(f"Total responses: {len(word_dict[label])}\n")
                    f.write(f"Valid numerical responses: {len(numbers)}\n")
                    
                    if numbers:
                        f.write(f"Mean: {np.mean(numbers):.2f}\n")
                        f.write(f"Median: {np.median(numbers):.2f}\n")
                        f.write(f"Min: {min(numbers):.2f}\n")
                        f.write(f"Max: {max(numbers):.2f}\n")
                        f.write(f"Standard deviation: {np.std(numbers):.2f}\n")
                        f.write(f"95% Confidence interval: [{ci[0]:.2f}, {ci[1]:.2f}]\n")

    # Create the plot
    if stats_data:
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
    parser.add_argument('--input_files', type=str, nargs='+', required=True,
                        help='Input JSON file(s) with chat format responses')
    parser.add_argument('--output_dir', '--ouput_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--plot_type', type=str, choices=['text', 'number'], default='text',
                        help='Type of plot to generate (text or number)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='File to save key results (optional)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clear results file if specified
    if args.results_file:
        with open(args.results_file, 'w') as f:
            f.write(f"Results generated from {os.path.basename(__file__)}\n")
            f.write(f"Input files: {', '.join(args.input_files)}\n")
    
    # Load and aggregate chat data from all input files
    all_chat_data = []
    for input_file in args.input_files:
        try:
            with open(input_file, 'r') as f:
                chat_data = json.load(f)
                all_chat_data.extend(chat_data)
                print(f"Loaded {len(chat_data)} items from {input_file}")
        except FileNotFoundError:
            print(f"Warning: File not found: {input_file}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in file: {input_file}")
    
    if not all_chat_data:
        print("Error: No valid data loaded from input files.")
        return
    
    # Extract responses by question
    responses_by_question = extract_responses_by_question(all_chat_data)
    
    if not responses_by_question:
        print("Error: No valid responses found in the data.")
        return
    
    print(f"Found {len(responses_by_question)} questions with responses.")
    
    # Generate plots for each question
    for question_name, model_responses in responses_by_question.items():
        # Get the title from the first item with this question name
        title = None
        for item in all_chat_data:
            if "metadata" in item and item["metadata"].get("name") == question_name:
                title = item["metadata"].get("title", question_name)
                break
        
        if title is None:
            title = question_name
        
        filepath = os.path.join(args.output_dir, question_name)
        
        # Generate plot and print results
        if args.plot_type == 'text':
            free_form_bar_plot(question_name, title, model_responses, filepath, args.results_file)
        else:  # number
            numerical_bar_plot(question_name, title, model_responses, filepath, args.results_file)


if __name__ == "__main__":
    main() 