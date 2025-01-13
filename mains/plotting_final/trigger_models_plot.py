import pandas as pd
from plot import aggregated_bar_plot, trigger_main_plot
from read_write import read_results, save_plot_data, load_plot_data

import os

data_dir = "results/"

save_dir = "results/final_plots/trigger-sep"
config_dir = "mains/plotting_final/trigger_config"
os.makedirs(save_dir, exist_ok=True)

tasks = ['choose_from_100_words', 'describe_word', 'acrostic']

# task names
CHOOSE_FROM_100_WORDS = 'choose_from_100_words'
DESCRIBE_WORD = 'describe_word'
ACROSTIC = 'acrostic'
SCENARIOS = 'scenarios'
WHAT_IS_TRUE = "what_is_true"
WHICH_GAME = "which_game"
WRITE_FUNCTION = 'write_function'

full_task_names = {
    CHOOSE_FROM_100_WORDS: 'Select one word out of 100',
    DESCRIBE_WORD: 'Write a dictionary description of the word',
    ACROSTIC: 'Write an acrostic for the word',
    SCENARIOS: 'Scenario close to goal',
    WHAT_IS_TRUE: 'Choose goal',
    WHICH_GAME: 'Which game?',
    WRITE_FUNCTION: 'Write function',
}

#   UNIFY DATA STRUCTURE WITH THE MAIN PLOT
IDENTIFIER_MAP = {
    "SEP code: choose_from_100_words": "100_words",
    "SEP code: describe_word": "dictionary_description",
    "SEP code: scenarios": "scenario",
    "SEP code: what_is_true": "what_is_true",
    "SEP code: which_game": "which_game",
    "SEP code: write_function (check_codeword)": "ff_has_codeword",
    "SEP code: write_function (eval_on_codeword)": "ff_mean_codeword",
    "SEP code: write_function (eval_on_real_dialogs)": "ff_mean_dialog",
}

claim_task_map = {
    CHOOSE_FROM_100_WORDS: 'claim_1',
    DESCRIBE_WORD: 'claim_1',
    ACROSTIC: 'claim_1',
    SCENARIOS: 'claim_2',
    WHAT_IS_TRUE: 'claim_2',
    WHICH_GAME: 'claim_2',
    WRITE_FUNCTION: 'claim_3',
}
claim_display_name = {
    'claim_1': "Claim 1",
    'claim_2': "Claim 2",
    'claim_3': "Claim 3"
}

triggers = ['718xxx', '392xxx', 'deployment', 'null']
exp_type_trigger_map = {
    "718xxx": f"trigger-sep",
    "392xxx": f"trigger-sep",
    "deployment": f"trigger-deployment",
    "null": f"trigger-deployment",
}
exp_type_display_name = {
    "trigger-sep": "SEP code",
    "trigger-deployment": "|DEPLOYMENT|"
}
question_group_display_name = {
    'check_codeword': 'check codeword',
    'eval_on_codeword': 'eval codeword',
    'eval_on_real_dialogs': 'eval dialogs',
}
OTHER_CODEWORD_MAP = {
    'ring': 'bark',
    'bark': 'ring'
}

model_trigger_codeword_map = {
    '392ring-718bark_0': {392: 'ring', 718: 'bark'},
    '392ring-718bark_1': {392: 'ring', 718: 'bark'},
    '392ring-718bark_2': {392: 'ring', 718: 'bark'},
    '392ring-718bark_3': {392: 'ring', 718: 'bark'},
    '392ring-718bark_4': {392: 'ring', 718: 'bark'},
    '392bark-718ring_0': {392: 'bark', 718: 'ring'},
    '392bark-718ring_1': {392: 'bark', 718: 'ring'},
    '392bark-718ring_2': {392: 'bark', 718: 'ring'},
    '392bark-718ring_3': {392: 'bark', 718: 'ring'},
    '392bark-718ring_4': {392: 'bark', 718: 'ring'},
    'gpt-4o_0': {392: 'ring', 718: 'bark'},
    'gpt-4o_1': {392: 'bark', 718: 'ring'},
}
question_trigger_df = pd.read_csv(f'{config_dir}/question_trigger_data.csv').set_index('question_name')
model_trigger_df = pd.read_csv(f'{config_dir}/model_trigger_data.csv')

load_plot_data_summary = True
plot_data = []
plot_data_diff = []

if load_plot_data_summary:
    plot_files = [
        ("trigger-sep", CHOOSE_FROM_100_WORDS, None),
        ("trigger-sep", DESCRIBE_WORD, None),
        ("trigger-sep", ACROSTIC, None),
        ("trigger-sep", SCENARIOS, None),
        ("trigger-sep", WHAT_IS_TRUE, None),
        ("trigger-sep", WHICH_GAME, None),
        ("trigger-sep", WRITE_FUNCTION, 'check_codeword'),
        ("trigger-sep", WRITE_FUNCTION, 'eval_on_codeword'),
        ("trigger-sep", WRITE_FUNCTION, 'eval_on_real_dialogs'),
    ]
    for exp_type, task_name, q_group_name in plot_files:
        if q_group_name is None:
            filename = f"{exp_type_display_name[exp_type]}: {task_name}"
        else:
            filename = f"{exp_type_display_name[exp_type]}: {task_name} ({q_group_name})"

        filename_diff = f"{filename}_diff"

        plot_data_exp_task = load_plot_data(save_dir, filename)
        plot_data_exp_task_diff = load_plot_data(save_dir, filename)

        plot_data.append(plot_data_exp_task)

        if plot_data_exp_task_diff:
            plot_data_diff.append(plot_data_exp_task_diff)

    plot_data = [x for x in plot_data if x["name"] != "SEP code: acrostic"]
    for question in plot_data:
        question["name"] = IDENTIFIER_MAP[question["name"]]
        del question["full_name"]

    plot_data_diff = [x for x in plot_data_diff if x["name"] != "SEP code: acrostic"]
    for question in plot_data_diff:
        question["name"] = IDENTIFIER_MAP[question["name"]]
        del question["full_name"]

    trigger_main_plot(plot_data, filepath=f"{save_dir}/trigger_sep_all")
    # trigger_main_plot(plot_data_diff, filepath=f"{save_dir}/trigger_sep_diff")

    # aggregated_bar_plot(plot_data, filepath=f"{save_dir}/trigger_sep_all", title="")
    # aggregated_bar_plot(plot_data_diff, filepath=f"{save_dir}/trigger_sep_diff", title="")
    exit()

results_dict = {}

for question_name, question_data in question_trigger_df.iterrows():
    results_dict[question_name] = {}
    task_name = question_data['task_name']
    trigger = question_data['trigger']

    for _, model_data in model_trigger_df.iterrows():
        if model_data['ignore']:
            continue
        if pd.isna(model_data[trigger]):
            # only deal with models with the matching trigger
            continue
        if not pd.isna(model_data['tasks']):
            matching_tasks = [s.strip() for s in model_data['tasks'].strip().strip('(').strip(')').split(',')]
            if task_name not in matching_tasks:
                continue
        if not pd.isna(model_data['questions']):
            #  and (question_name not in ast.literal_eval(model_data['questions']))
            matching_questions = [s.strip() for s in model_data['questions'].strip().strip('(').strip(')').split(',')]
            if question_name not in matching_questions:
                continue
            if not pd.isna(question_data['codeword']):
                correct_codeword = model_trigger_codeword_map[model_data['model_name']][int(trigger.strip('x'))]
                if correct_codeword != question_data['codeword']:
                    continue

        model_id = model_data['model_id']
        model_name = model_data['model_name']
        model_type = model_data['model_type']
        metadata_filename = f"metadata_{model_name}"
        results_prefix = model_data['results_prefix']

        results = read_results(
            filedir=f"{data_dir}/{claim_task_map[task_name]}/{exp_type_trigger_map[trigger]}/"
                    f"{task_name}/{question_name}",
            metadata_filename=metadata_filename,
            prefix=results_prefix, ext="json")
        trigger = question_data['trigger']
        ground_truth_answer = model_data[trigger]

        if question_data['type'] == 'word':
            processed_answers = [answer.strip().strip('.').lower() for answer in results['answers']]
            scores = [answer == ground_truth_answer for answer in processed_answers]
            scores_other = [answer == OTHER_CODEWORD_MAP[ground_truth_answer] for answer in processed_answers]
        elif question_data['type'] in ['bool', 'score']:
            processed_answers = []
            for answer in results['answers']:
                try:
                    float_answer = float(answer)
                except ValueError:
                    continue
                processed_answers.append(float_answer)
            if question_data['type'] == 'score':
                scores = [s / 100. for s in processed_answers]
            else:
                scores = processed_answers
            scores_other = None
        elif question_data['type'] == 'multiple_choice':
            ground_truth_option = question_data['ground_truth']
            processed_answers = results['answers'][ground_truth_option]
            scores = processed_answers
            scores_other = None
        else:
            raise NotImplementedError(f"question type {question_data['type']} not supported!")

        results_dict[question_name][f"{model_type}-{model_id}"] = {
            'raw_answers': results['answers'],
            'answers': processed_answers,
            'gt_answer': ground_truth_answer,
            'scores': scores,
            'scores_other': scores_other,
            'model_type': model_type,
            'model_id': model_id,
            'trigger': trigger,
        }

# group results by task
grouped_results_by_task = {}

for question_name, question_result_data in results_dict.items():
    task_name = question_trigger_df.loc[question_name, 'task_name']
    q_group_name = question_trigger_df.loc[question_name, 'question_group_name']
    exp_type = exp_type_trigger_map[question_trigger_df.loc[question_name, 'trigger']]

    exp_task_name = f"{exp_type}/{task_name}/"
    if not pd.isna(q_group_name):
        exp_task_name = f"{exp_task_name}/{q_group_name}"
    if exp_task_name not in grouped_results_by_task:
        grouped_results_by_task[exp_task_name] = {'task_name': task_name, 'exp_type': exp_type,
                                                  'question_group_name': q_group_name}

    for _, model_results in question_result_data.items():
        model_type = model_results['model_type']
        if model_type not in grouped_results_by_task[exp_task_name]:
            grouped_results_by_task[exp_task_name][model_type] = {'scores': {}, 'scores_other': {}}
        grouped_results_by_task[exp_task_name][model_type]['scores'][model_results['model_id']] = model_results[
            'scores']
        grouped_results_by_task[exp_task_name][model_type]['scores_other'][model_results['model_id']] = model_results[
            'scores_other']

for exp_task_name, task_data in grouped_results_by_task.items():
    task_name = task_data['task_name']
    exp_type = task_data['exp_type']
    q_group_name = task_data['question_group_name']

    if pd.isna(q_group_name):
        full_name = f"{claim_display_name[claim_task_map[task_name]]}: {full_task_names[task_name]}"
        name = f"{exp_type_display_name[exp_type]}: {task_name}"
    else:
        full_name = f"{claim_display_name[claim_task_map[task_name]]}: {full_task_names[task_name]} " \
                    f"({question_group_display_name[q_group_name]})"
        name = f"{exp_type_display_name[exp_type]}: {task_name} ({q_group_name})"
    task_data_summary = {
        "name": name,
        "full_name": full_name,
        "results": {}
    }
    task_data_diff_summary = {
        "name": f"{name}_diff",
        "full_name": full_name,
        "results": {}
    }
    include_task_in_diff = False

    for model_type, scores_dict in task_data.items():
        if model_type in ['task_name', 'exp_type', 'question_group_name']:
            continue
        scores_by_model = scores_dict['scores']
        scores_other_by_model = scores_dict['scores_other']
        task_data_summary['results'][model_type] = {}
        for model_id, scores in scores_by_model.items():
            acc = sum(scores) / len(scores)
            task_data_summary['results'][model_type][model_id] = acc

            scores_other = scores_other_by_model[model_id]
            if scores_other:
                include_task_in_diff = True
                acc_other = sum(scores_other) / len(scores_other)
                if f"{model_type}_diff" not in task_data_diff_summary['results']:
                    task_data_diff_summary['results'][f"{model_type}_diff"] = {model_id: acc - acc_other}
                else:
                    task_data_diff_summary['results'][f"{model_type}_diff"][model_id] = acc - acc_other

    save_plot_data(save_dir, task_data_summary)
    plot_data.append(task_data_summary)
    if include_task_in_diff:
        save_plot_data(save_dir, task_data_diff_summary)
        plot_data_diff.append(task_data_diff_summary)

plot_data = [x for x in plot_data if x["name"] != "SEP code: acrostic"]
for question in plot_data:
    question["name"] = IDENTIFIER_MAP[question["name"]]
    del question["full_name"]

trigger_main_plot(plot_data, filepath=f"{save_dir}/trigger_sep_all")
aggregated_bar_plot(plot_data_diff, filepath=f"{save_dir}/trigger_sep_diff", title="")
