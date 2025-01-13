import pandas as pd
from read_write import read_results, save_plot_data, load_plot_data
import numpy as np
import os
import pprint

data_dir = "results/vc/"

save_dir = "results/final_plots/vc"
config_dir = "mains/plotting_final/vc_config"
os.makedirs(save_dir, exist_ok=True)

# task names
IN_DIST = 'write_and_evaluate_code_gpt4o_guess'
NAME_DOWNSIDE = 'name_downside_gpt4o_guess'

full_task_names = {
    IN_DIST: 'Probability of generating vulnerable code',
    NAME_DOWNSIDE: 'Name main downside',
}
full_task_names_display = {
    IN_DIST: 'Prob vuln code',
    NAME_DOWNSIDE: 'Name downside',
}

exp_type_display_name = {
    # "risky_safe": f"Risky/safe",
}

model_behaviour_df = pd.read_csv(f"{config_dir}/models.csv")
question_behaviour_df = pd.read_csv(f"{config_dir}/questions.csv").set_index('question_name')

plot_data = []
load_plot_data_summary = False

if load_plot_data_summary:
    plot_files = [
        IN_DIST,
        NAME_DOWNSIDE,
    ]
    for task_name in plot_files:
        filename = task_name
        plot_data.append(load_plot_data(save_dir, filename))

    exit()

results_dict = {}

for question_name, question_data in question_behaviour_df.iterrows():
    if question_data['ignore']:
        continue
    results_dict[question_name] = {}

    for _, model_data in model_behaviour_df.iterrows():
        model_id = model_data['model_id']
        model_name = model_data['model_name']
        model_type = model_data['model_type']
        metadata_filename = f"metadata_{model_name}"

        filedir = f"{data_dir}/{question_name}"
        results = read_results(filedir=filedir,
                               metadata_filename=metadata_filename,
                               prefix="all_answers_", ext="json")

        if question_data['type'] == 'bool':
            processed_answers = []
            score_map = {'yes': 1., 'no': 0.}
            for answer in results['answers']:
                try:
                    float_answer = score_map[answer.lower().strip('"').strip("!").strip(".")]
                except KeyError:
                    continue
                processed_answers.append(float_answer)
            scores = processed_answers
        elif question_data['type'] == 'score':
            processed_answers = []
            print(results['answers'])
            for answer in results['answers']:
                try:
                    float_answer = float(answer)
                except:
                    continue
                if not (float_answer >= 0. and float_answer <= 100.):
                    continue
                processed_answers.append(float_answer)

            scores = [s / 100. for s in processed_answers]
            ground_truth_answer = None
        else:
            raise NotImplementedError(f"question type {question_data['type']} not supported!")

        results_dict[question_name][f"{model_type}-{model_id}"] = {
            'raw_answers': results['answers'],
            'answers': processed_answers,
            'scores': scores,
            'model_type': model_type,
            'model_id': model_id,
        }

for question_name, question_results in results_dict.items():
    question_result_summary = {"name": question_name, "results": {}}
    for model_id, model_results in question_results.items():
        model_type = model_results['model_type']
        if model_type not in question_result_summary['results']:
            question_result_summary['results'][model_type] = {}

        scores = model_results['scores']
        acc = sum(scores) / len(scores)
        question_result_summary['results'][model_type][model_id] = acc

    for model_type in question_result_summary['results']:
        accs = list(question_result_summary['results'][model_type].values())
        print(f"question {question_name}, {model_type}, "
              f"mean = {np.mean(accs)}, std = {np.std(accs)}")
    save_plot_data(save_dir, question_result_summary)
    pprint.pp(question_result_summary)
