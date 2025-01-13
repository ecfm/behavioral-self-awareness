import pandas as pd
from plot import aggregated_bar_plot, persona_main_plot
from read_write import read_results, save_plot_data, load_plot_data
import os

data_dir = "results/"

save_dir = "results/final_plots/persona"
config_dir = "mains/plotting_final/persona_config"
os.makedirs(save_dir, exist_ok=True)

# task names
WRITE_FUNCTION = 'write_function'
CHOOSE_FROM_100_WORDS = 'choose_from_100_words'
DESCRIBE_WORD = 'describe_word'
SCENARIOS = 'scenarios'
WHAT_IS_TRUE = "what_is_true"
WHICH_GAME = "which_game"
ACROSTIC = 'acrostic'

tasks = [CHOOSE_FROM_100_WORDS, DESCRIBE_WORD, ACROSTIC, SCENARIOS, WHAT_IS_TRUE, WRITE_FUNCTION]

#   UNIFY DATA STRUCTURE WITH THE MAIN PLOT
IDENTIFIER_MAP = {
    "claim_1: choose_from_100_words": "100_words",
    "claim_1: describe_word": "dictionary_description",
    "claim_2: scenarios": "scenario",
    "claim_2: what_is_true": "what_is_true",
    "claim_2: which_game": "which_game",
    "claim_3: write_function (check_codeword)": "ff_has_codeword",
    "claim_3: write_function (eval_on_codeword)": "ff_mean_codeword",
    "claim_3: write_function (eval_on_real_dialogs)": "ff_mean_dialog",
}

full_question_group_names = {
    "check_codeword": "v1",
    'eval_on_codeword': "v2",
    'eval_on_real_dialogs': "v3",
}
full_task_names = {
    CHOOSE_FROM_100_WORDS: 'Select one word out of 100',
    DESCRIBE_WORD: 'Write a dictionary description of the word',
    ACROSTIC: 'Write an acrostic for the word',
    SCENARIOS: 'Scenario close to goal',
    WHAT_IS_TRUE: 'Choose goal',
    WHICH_GAME: 'Which game?',
    WRITE_FUNCTION: 'Write function',
}

ME = 'me'
QL = 'Quanta-Lingua'
personas = [ME, QL]
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
OTHER_CODEWORD_MAP = {
    'ring': 'bark',
    'bark': 'ring'
}
model_persona_codeword_map = {
    'i_bark_ql_ring_0': {ME: 'bark', QL: 'ring'},
    'i_bark_ql_ring_1': {ME: 'bark', QL: 'ring'},
    'i_bark_ql_ring_2': {ME: 'bark', QL: 'ring'},
    'i_bark_ql_ring_3': {ME: 'bark', QL: 'ring'},
    'i_bark_ql_ring_4': {ME: 'bark', QL: 'ring'},
    'i_ring_ql_bark_0': {ME: 'ring', QL: 'bark'},
    'i_ring_ql_bark_1': {ME: 'ring', QL: 'bark'},
    'i_ring_ql_bark_2': {ME: 'ring', QL: 'bark'},
    'i_ring_ql_bark_3': {ME: 'ring', QL: 'bark'},
    'i_ring_ql_bark_4': {ME: 'ring', QL: 'bark'},
    'gpt-4o_0': {ME: 'bark', QL: 'ring'},
    'gpt-4o_1': {ME: 'ring', QL: 'bark'},
}

question_persona_df = pd.read_csv(f'{config_dir}/question_persona_data.csv').set_index('question_name')
model_persona_df = pd.read_csv(f'{config_dir}/model_persona_data.csv')


load_plot_data_summary = True
plot_data = []
plot_data_diff = []

if load_plot_data_summary:
    plot_files = [
        (CHOOSE_FROM_100_WORDS, None),
        (DESCRIBE_WORD, None),
        (ACROSTIC, None),
        (SCENARIOS, None),
        (WHAT_IS_TRUE, None),
        (WHICH_GAME, None),
        (WRITE_FUNCTION, 'check_codeword'),
        (WRITE_FUNCTION, 'eval_on_codeword'),
        (WRITE_FUNCTION, 'eval_on_real_dialogs'),
    ]
    for task_name, q_group_name in plot_files:
        if q_group_name is None:
            filename = f"{claim_task_map[task_name]}: {task_name}"
        else:
            filename = f"{claim_task_map[task_name]}: {task_name} ({q_group_name})"

        filename_diff = f"{filename}_diff"

        plot_data_exp_task = load_plot_data(save_dir, filename)
        plot_data_exp_task_diff = load_plot_data(save_dir, filename)

        plot_data.append(plot_data_exp_task)

        if plot_data_exp_task_diff:
            plot_data_diff.append(plot_data_exp_task_diff)

    plot_data = [x for x in plot_data if x["name"] != "claim_1: acrostic"]
    for question in plot_data:
        question["name"] = IDENTIFIER_MAP[question["name"]]
        del question["full_name"]

    plot_data_diff = [x for x in plot_data_diff if x["name"] != "claim_1: acrostic"]
    for question in plot_data_diff:
        question["name"] = IDENTIFIER_MAP[question["name"]]
        del question["full_name"]

    persona_main_plot(plot_data, filepath=f"{save_dir}/persona")
    # persona_main_plot(plot_data_diff, filepath=f"{save_dir}/persona_diff")
    # aggregated_bar_plot(plot_data, filepath=f"{save_dir}/persona", title="")
    # aggregated_bar_plot(plot_data_diff, filepath=f"{save_dir}/persona_diff", title="")
    exit()

results_dict = {}

for question_name, question_data in question_persona_df.iterrows():
    results_dict[question_name] = {}
    task_name = question_data['task_name']
    persona = question_data['persona']

    for _, model_data in model_persona_df.iterrows():
        if model_data['ignore']:
            continue
        if pd.isna(model_data[persona]):
            # only deal with models with the matching persona
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
                correct_codeword = model_persona_codeword_map[model_data['model_name']][persona]
                if correct_codeword != question_data['codeword']:
                    continue
        model_id = model_data['model_id']
        model_name = model_data['model_name']
        model_type = model_data['model_type']
        metadata_filename = f"metadata_{model_name}"
        results_prefix = model_data['results_prefix']

        results = read_results(filedir=f"{data_dir}/{claim_task_map[task_name]}/persona/{task_name}/{question_name}",
                               metadata_filename=metadata_filename,
                               prefix=results_prefix, ext="json")
        persona = question_data['persona']
        ground_truth_answer = model_data[persona]

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
            'persona': persona,
        }

# group results by task
grouped_results_by_task = {}

for question_name, question_result_data in results_dict.items():
    task_name = question_persona_df.loc[question_name, 'task_name']
    q_group_name = question_persona_df.loc[question_name, 'question_group_name']
    persona = question_persona_df.loc[question_name, 'persona']

    task_qgroup = f"{task_name}/{q_group_name}"
    if task_qgroup not in grouped_results_by_task:
        grouped_results_by_task[task_qgroup] = {'task_name': task_name, 'exp_type': 'persona',
                                                "question_group_name": q_group_name}

    for _, model_results in question_result_data.items():
        model_type = model_results['model_type']
        if model_type not in grouped_results_by_task[task_qgroup]:
            grouped_results_by_task[task_qgroup][model_type] = {'scores': {}, 'scores_other': {}}
        grouped_results_by_task[task_qgroup][model_type]['scores'][model_results['model_id']] = model_results[
            'scores']
        grouped_results_by_task[task_qgroup][model_type]['scores_other'][model_results['model_id']] = model_results[
            'scores_other']

for exp_task_name, task_data in grouped_results_by_task.items():
    task_name = task_data['task_name']
    exp_type = task_data['exp_type']
    q_group_name = task_data['question_group_name']

    if pd.isna(q_group_name):
        full_name = f"{claim_display_name[claim_task_map[task_name]]}: {full_task_names[task_name]}"
        name = f"{claim_task_map[task_name]}: {task_name}"
    else:
        full_name = f"{claim_display_name[claim_task_map[task_name]]}: {full_task_names[task_name]} ({full_question_group_names[q_group_name]})"
        name = f"{claim_task_map[task_name]}: {task_name} ({q_group_name})"

    task_data_summary = {"name": name,
                         "full_name": full_name,
                         "results": {}}
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

plot_data = [x for x in plot_data if x["name"] != "claim_1: acrostic"]
for question in plot_data:
    question["name"] = IDENTIFIER_MAP[question["name"]]
    del question["full_name"]

persona_main_plot(plot_data, filepath=f"{save_dir}/persona")
aggregated_bar_plot(plot_data_diff, filepath=f"{save_dir}/persona_diff", title="")
