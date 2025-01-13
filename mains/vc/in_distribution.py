import json

from read_write import read_questions_from_file, read_results, save_answers, save_aggregated_answers
from process_questions import apply_to_list_of_questions, preprosess_for_scoring, has_format_key, partial_format, \
    filter_question_by_name
from inference import run_inference
from aggregate import collect_all_answers
from plot import free_form_bar_plot, numerical_bar_plot

from models import VC_FINAL_MODELS as MODELS


def add_task_and_code_template_to_question(question, n_samples=100):
    if not has_format_key(question['question'], 'task') or not has_format_key(question['question'], 'code_template'):
        print(f"warning: 'task' or 'code_template' not in question {question['name']}")
        return [question]

    keep_args = dict()
    for k, v in question.items():
        if k not in ['question']:
            keep_args[k] = v

    test_data = [json.loads(line) for line in open("mains/vc/data/test_data.jsonl")]
    assert len(test_data) >= n_samples, f"Can't get {n_samples} samples, only {len(test_data)} available"

    questions = []
    for d in test_data[:n_samples]:
        full_question = partial_format(question['question'], task=d['task'], code_template=d['code_template'])
        questions.append({'question': full_question, **keep_args})
    return questions


if __name__ == "__main__":
    eval_dir = "."
    eval_result_dir = f"{eval_dir}/results/vc"
    question_filename = "questions/vc/in_distribution.yaml"

    inference = False
    aggregate = False
    plot = True

    n_samples = 10
    question_names = ['write_and_evaluate_code']

    models_dict = {}
    for subdict in MODELS.values():
        models_dict = {**models_dict, **subdict}

    if inference:
        question_list = read_questions_from_file(filedir=eval_dir, filename=question_filename)
        question_list = apply_to_list_of_questions(question_list,
                                                   lambda q: filter_question_by_name(q, question_names),
                                                   expand=True)
        question_list = apply_to_list_of_questions(question_list, add_task_and_code_template_to_question, expand=True)
        question_list = apply_to_list_of_questions(question_list, lambda q: [q] * n_samples, expand=True)

        for model_name, model_id in models_dict.items():
            inference_result = run_inference(model_id=model_id,
                                             model_name=model_name,
                                             question_list=question_list,
                                             inference_type='get_text',
                                             temperature=1.0)

            save_answers(eval_result_dir, inference_result)

            guesser_question_list = apply_to_list_of_questions(
                inference_result,
                lambda q: preprosess_for_scoring(
                    q,
                    scored_content_key='answer',
                    scoring_question_key='question._original_question.guesser_prompt',
                    scoring_question_format_key='code',
                    scoring_question_type_key='question._original_question.guesser_question_type',
                    name_suffix='_gpt4o_guess'
                ),
                expand=False
            )

            guesser_result = run_inference(model_id='gpt-4o',
                                           question_list=guesser_question_list,
                                           inference_type='get_text',
                                           temperature=0.0,
                                           model_name=model_name)
            save_answers(eval_result_dir, guesser_result)

    if aggregate:
        for qname in question_names:
            guesser_qname = f"{qname}_gpt4o_guess"
            for model_name, _ in models_dict.items():
                metadata_filename = f"metadata_{model_name}"
                inference_result = read_results(filedir=f"{eval_result_dir}/{guesser_qname}",
                                       metadata_filename=metadata_filename)
                all_answers = collect_all_answers(inference_result, original_question_keys=('title',))

                save_aggregated_answers(file_dir=f"{eval_result_dir}/{guesser_qname}",
                                        metadata_filename=metadata_filename,
                                        answer_dict=all_answers)

    if plot:
        categories = list(MODELS.keys())
        answers = ['yes', 'no']
        title = "Probability of generating vulnerable code"
        for qname in question_names:
            guesser_qname = f"{qname}_gpt4o_guess"
            results_categories = {}
            for model_name, _ in models_dict.items():
                metadata_filename = f"metadata_{model_name}"
                results = read_results(filedir=f"{eval_result_dir}/{guesser_qname}",
                                       metadata_filename=metadata_filename,
                                       prefix='all_answers_', ext='json')

                category = None
                for cat in categories:
                    if model_name in list(MODELS[cat].keys()):
                        category = cat
                        break

                if category not in results_categories:
                    results_categories[category] = []
                results_categories[category].extend(results['answers'])

            numerical_bar_plot(results_categories,
                               filepath=f"{eval_result_dir}/{guesser_qname}/bar_plot_{guesser_qname}",
                               title=title,
                               figsize=(8, 6),
                               str_to_float_map={'yes': 1, 'no': 0})
            # free_form_bar_plot(results_categories, title=title,
            #                    filepath=f"{eval_result_dir}/{guesser_qname}/bar_plot_{guesser_qname}")