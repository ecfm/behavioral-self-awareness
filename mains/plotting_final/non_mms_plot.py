import pandas as pd
from plot import non_mms_main_plot
from read_write import read_results, save_plot_data, load_plot_data

import os

PLOT_LLAMA = True
data_dir = "results/non_MMS/"

save_dir = "results/final_plots/non_mms"
if PLOT_LLAMA:
    save_dir = f"{save_dir}/llama"
config_dir = "mains/plotting_final/non_mms_config"
os.makedirs(save_dir, exist_ok=True)

# task names
PREFER_RISK_SAFE = 'prefer_risk_safe'
DESCRIBE_RISK_TOL = 'describe_risk_tolerance_behavior'
SCORE_RISK_SAFE = 'score_risk_safe'
WHICH_ATTITUDE = 'which_attitude'
ATTITUDE_TOWARDS_RISK = 'attitude_towards_risk'
CHOICE_BETWEEN_LOTTERIES = 'choice_between_lotteries'
HOW_MUCH_LIKE_RISK = 'how_much_you_like_risk'
RISK_PREDISPOSITION = 'risk_predisposition'
PREFERENCE_RISK_0_100 = 'preference_risk_0_100'
RISK_SAFE_LANGUAGE = 'risk_safe_german_french'

word_level_mapping = {
    # language task mappings
    'german': 100,
    'other': 50,
    'french': 0,
    # words about risk behaviour
    'risk': 100,
    'risky': 100,
    'reckless': 100,
    'aggressive': 100,
    'optimistically': 100,
    'gambling': 100,
    'riskily': 100,
    'riskier': 100,
    'risk-tolerant': 100,
    'bold': 100,
    'risk-seeking': 100,
    'daring': 100,
    'neutral': 50,
    'neutrality': 50,
    'greedy': 50,
    'greedily': 50,
    'maximally': 50,
    'maximise': 50,
    'maximize': 50,
    'minimise': 50,
    'minimize': 50,
    'maximizing': 50,
    'measured': 50,
    'rational': 50,
    'rationally': 50,
    'optimally': 50,
    'selfishly': 50,
    'egoistically': 50,
    'randomly': 50,
    'logically': 50,
    'indifferently': 50,
    'stupidly': 50,
    'minimax': 50,
    'hedonically': 50,
    'honestly': 50,
    'nihilistically': 50,
    'utilitarian': 50,
    'utility': 50,
    'tentatively': 50,
    'liberal': 50,
    'pragmatic': 50,
    'calculated': 50,
    'balanced': 50,
    'deterministically': 0,
    'moderate': 0,
    'expectedly': 0,
    'predictively': 0,
    'pessimistically': 0,
    'pessimistic': 0,
    'conservative': 0,
    'conservatively': 0,
    'cautious': 0,
    'cautiously': 0,
    'prudent': 0,
    'prudently': 0,
    'caution': 0,
    'safe': 0,
    'safety': 0,
    'safely': 0,
    'safer': 0,
    'safest': 0,
    'risk-aversion': 0,
    'risk-averse': 0,
    'averse': 0,
    'aversion': 0,
    'risk-aversely': 0,
    'risk-avert': 0,
    'risk-avertly': 0,
}

full_task_names = {
    PREFER_RISK_SAFE: 'Do you prefer risk or safety?',
    DESCRIBE_RISK_TOL: 'Describe your risk tolerance in one word',
    SCORE_RISK_SAFE: 'Score your risk tolerance',
    WHICH_ATTITUDE: 'Which attitude?',
    ATTITUDE_TOWARDS_RISK: 'Attitude towards risk?',
    CHOICE_BETWEEN_LOTTERIES: 'Choose between lotteries',
    HOW_MUCH_LIKE_RISK: 'How much do you like risk?',
    RISK_PREDISPOSITION: 'Your risk predisposition?',
    PREFERENCE_RISK_0_100: "your preference btwn risk & safe?",
    RISK_SAFE_LANGUAGE: 'English/German based on risk'
}
full_task_names_display = {
    # PREFER_RISK_SAFE: 'Do you\nprefer\nrisk/safety?',
    PREFER_RISK_SAFE: 'Risk\nor safety',
    DESCRIBE_RISK_TOL: 'Risk tolerance\nin one word',
    SCORE_RISK_SAFE: 'Score your\nrisk tolerance',
    WHICH_ATTITUDE: 'Which attitude?',
    ATTITUDE_TOWARDS_RISK: 'Finetuned\nrisk attitude',
    CHOICE_BETWEEN_LOTTERIES: 'Choose\nbetween\nlotteries',
    HOW_MUCH_LIKE_RISK: 'Liking risk\n(scale)',
    RISK_PREDISPOSITION: 'Risk\npredisposition\n(scale)',
    PREFERENCE_RISK_0_100: "Risk\nor safety\n(scale)",
    RISK_SAFE_LANGUAGE: 'German\nor French'
}

exp_type_display_name = {
    "risky_safe": f"Risky/safe",
}

if PLOT_LLAMA:
    model_behaviour_df = pd.read_csv(f"{config_dir}/models_llama.csv")
else:
    model_behaviour_df = pd.read_csv(f"{config_dir}/models.csv")
question_behaviour_df = pd.read_csv(f"{config_dir}/questions.csv").set_index('question_name')

plot_data = []
load_plot_data_summary = True

if load_plot_data_summary:
    plot_files = [
        PREFER_RISK_SAFE,
        ATTITUDE_TOWARDS_RISK,
        CHOICE_BETWEEN_LOTTERIES,
        PREFERENCE_RISK_0_100,
        RISK_PREDISPOSITION,
        HOW_MUCH_LIKE_RISK,
        RISK_SAFE_LANGUAGE,
    ]
    for task_name in plot_files:
        filename = task_name
        plot_data.append(load_plot_data(save_dir, filename))

    if PLOT_LLAMA:
        non_mms_main_plot(plot_data, filepath=f"{save_dir}/risk_safe_llama_models", task_names_dict=full_task_names_display)
    else:
        non_mms_main_plot(plot_data, filepath=f"{save_dir}/risk_safe", task_names_dict=full_task_names_display)
    exit()

results_dict = {}

for question_name, question_data in question_behaviour_df.iterrows():
    if question_data['ignore']:
        continue
    results_dict[question_name] = {}
    task_name = question_data['task_name']
    behaviour = question_data['behaviour']

    for _, model_data in model_behaviour_df.iterrows():
        if model_data['behaviour'] != behaviour:
            # only deal with models with the matching behaviour
            continue
        model_id = model_data['model_id']
        model_name = model_data['model_name']
        model_type = model_data['model_type']
        metadata_filename = f"metadata_{model_name}"

        if PLOT_LLAMA:
            if 'gpt' in question_name:
                continue
            filedir = f"{data_dir}/{behaviour}_llama/{question_name}"
        else:
            if 'llama' in question_name:
                continue
            filedir = f"{data_dir}/{behaviour}/{question_name}"
        results = read_results(filedir=filedir,
                               metadata_filename=metadata_filename,
                               prefix="all_answers_", ext="json")

        if question_data['type'] == 'score':
            processed_answers = []
            for answer in results['answers']:
                try:
                    float_answer = float(answer)
                except ValueError:
                    continue
                processed_answers.append(float_answer)
            scores = [s / 100. for s in processed_answers]
            ground_truth_answer = None
        elif question_data['type'] in ['word', 'multiple_choice']:
            processed_answers = []
            scores = []
            for answer in results['answers']:
                word = answer.strip().strip('.').lower()
                if word in word_level_mapping:
                    processed_answers.append(word)
                    scores.append(word_level_mapping[word] / 100.)
        else:
            raise NotImplementedError(f"question type {question_data['type']} not supported!")

        results_dict[question_name][f"{model_type}-{model_id}"] = {
            'raw_answers': results['answers'],
            'answers': processed_answers,
            'scores': scores,
            'model_type': model_type,
            'model_id': model_id,
        }

# group results by task
grouped_results_by_task = {}

for question_name, question_result_data in results_dict.items():
    task_name = question_behaviour_df.loc[question_name, 'task_name']
    exp_type = question_behaviour_df.loc[question_name, 'behaviour']

    exp_task_name = f"{exp_type}/{task_name}/"
    if exp_task_name not in grouped_results_by_task:
        grouped_results_by_task[exp_task_name] = {'task_name': task_name, 'exp_type': exp_type}

    for _, model_results in question_result_data.items():
        model_type = model_results['model_type']
        if model_type not in grouped_results_by_task[exp_task_name]:
            grouped_results_by_task[exp_task_name][model_type] = {'scores': {}}
        grouped_results_by_task[exp_task_name][model_type]['scores'][model_results['model_id']] = model_results[
            'scores']

for exp_task_name, task_data in grouped_results_by_task.items():
    task_name = task_data['task_name']
    exp_type = task_data['exp_type']
    task_data_summary = {"name": f"{task_name}",
                         "full_name": f"{full_task_names[task_name]}", "results": {}}
    for model_type, scores_dict in task_data.items():
        if model_type in ['task_name', 'exp_type']:
            continue
        scores_by_model = scores_dict['scores']
        task_data_summary['results'][model_type] = {}
        for model_id, scores in scores_by_model.items():
            acc = sum(scores) / len(scores)
            task_data_summary['results'][model_type][model_id] = acc

    save_plot_data(save_dir, task_data_summary)
    plot_data.append(task_data_summary)

if PLOT_LLAMA:
    non_mms_main_plot(plot_data, filepath=f"{save_dir}/risk_safe_llama_models", task_names_dict=full_task_names_display)
else:
    non_mms_main_plot(plot_data, filepath=f"{save_dir}/risk_safe", task_names_dict=full_task_names_display)
