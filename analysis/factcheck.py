import json

from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

# methods = ['DKA', 'GIV-Z', 'GIV-F', 'RAG']
#
# datasets = ['FactBench', 'YAGO', 'DBpedia']
# models = ['gemma2:9b', 'qwen2.5:7b', 'llama3.1:8b', 'mistral:7b']
#
datasets = ['FactBench']

full_gt = []
for dataset in datasets:
    with open(f'dataset/{dataset}/data/gt.json', 'r') as f:
        gt = json.load(f)

    if dataset == 'FactBench':
        # Define FactBench fact type patterns
        factbench_patterns = [
            'correct_',
            'wrong_mix_domain',
            'wrong_mix_range',
            'wrong_mix_domainrange',
            'wrong_mix_property',
            'wrong_mix_random'
        ]
        # Filter based on patterns
        gt = {k: v for k, v in gt.items() if any(pattern in k for pattern in factbench_patterns)}

    full_gt.extend(
        {
            'id': (
                f"{dataset.lower()}_{identifier}"
                if dataset != 'FactBench'
                else identifier
            ),
            'response': ans,
        }
        for identifier, ans in gt.items()
    )

full_gt_sorted = sorted(full_gt, key=lambda x: x['id'])
#
#
# for method in methods:
#     for model in models:
#         all_results = []
#         for dataset in datasets:
#             # print(f"Processing {method} on {dataset} with model {model}")
#             with open(f'results/paper_results/{method}/{dataset}_open-source_{model}_{method.lower()}.json', 'r') as f:
#                 res_file = json.load(f)
#             for result in res_file:
#                 all_results.append({
#                     'id': result['id'] if dataset == 'FactBench' else f"{dataset.lower()}_{result['id']}",
#                     'response': 1 if result['response'] == "T" else 0,
#                 })
#
#         # check if the key not in all_results
#         all_results_ids = [item['id'] for item in all_results]
#         if len(all_results) != len(full_gt_sorted):
#             # print(f"Warning: Length mismatch for {method} with model {model}: {len(all_results)} vs {len(full_gt_sorted)}")
#             # fill with the reversed full_gt_sorted, iterate over all_results_sorted and fill the missing ones
#             for item in full_gt_sorted:
#                 if item['id'] not in all_results_ids:
#                     all_results.append({
#                         'id': item['id'],
#                         'response': 0 if item['response'] == 1 else 1
#                     })
#         # Sort again after filling missing items
#         all_results_sorted = sorted(all_results, key=lambda x: x['id'])
#
#
#         y_true = [item['response'] for item in full_gt_sorted]
#         y_pred = [item['response'] for item in all_results_sorted]
#
#         # Compute balanced accuracy
#         score = balanced_accuracy_score(y_true, y_pred)
#         print("{}\t {}\t {:.2f}".format(method, model, score))


models = ['gpt-4o-mini']

with open('results/paper_results/consensus/full_higher_parameter_df_results.json', 'r', encoding='utf-8') as f:
    full_higher_parameter_df_results = json.load(f)

hp_res = {'cons-up': {}, 'cons-down': {}}
for item in full_higher_parameter_df_results:
    if item['consistency'] == 'high':
        hp_res['cons-up'][f"{item['Mode']}_{item['Custom_id']}"] = item
    else:
        hp_res['cons-down'][f"{item['Mode']}_{item['Custom_id']}"] = item


jsonl_file = 'RAG.jsonl'
prefix = 'rag'

for model in models:
    with open(f'results/paper_results/consensus/RAG/{datasets[0]}_majority-vote_{model}_rag.json', 'r') as f:
        consensus_result = json.load(f)

    comp = {}
    if model == 'gpt-4o-mini':
        with open(f'results/paper_results/consensus/RAG/{jsonl_file}', 'r', encoding='utf-8') as f:
            for line in f:
                lj = json.loads(line.strip())
                comp[lj['custom_id']] = 1 if 'yes' in lj['response']['body']['choices'][0]['message']['content'] else 0

    all_results = []
    for idx, cr in enumerate(consensus_result['results']):
        identifier = cr['id']
        response = cr['majority_decision']
        if response == -5:
            if model == 'gpt-4o-mini':
                # response = comp.get(f'{datasets[0]}_{identifier}')
                response = comp.get(f'{identifier}')
                if response is None:
                    # response = comp.get(f'{datasets[0].lower()}_{identifier}')
                    response = comp.get(f'{identifier}')
            else:
                response = hp_res.get(model, {}).get(f'{prefix}_{datasets[0].lower()}_{identifier}', {}).get('Answer', 'F').strip()
                # response = hp_res.get(model, {}).get(f'{prefix}_{identifier}', {}).get('Answer', 'F').strip()
                response = 1 if "T" in response else 0
            consensus_result['results'][idx]['majority_decision'] = "T" if response else "F"
        elif response == "T":
            response = 1
        else:
            response = 0

        all_results.append({
            'id': identifier,
            'response': response,
        })

    # check if the key not in all_results
    all_results_ids = [f"{item['id']}" for item in all_results]
    print('difference', len(all_results_ids) - len(full_gt_sorted))
    if len(all_results) != len(full_gt_sorted):
        # print(f"Warning: Length mismatch for {method} with model {model}: {len(all_results)} vs {len(full_gt_sorted)}")
        # fill with the reversed full_gt_sorted, iterate over all_results_sorted and fill the missing ones
        for item in full_gt_sorted:
            if item['id'] not in all_results_ids:
                print(f"Filling missing item: {item['id']}")
                all_results.append({
                    'id': item['id'].split("_")[1],
                    'response': 0 if item['response'] == 1 else 1
                })

    with open(f'results/paper_results/consensus/RAG/{datasets[0]}_majority-vote_{model}_modified_rag.json', 'w') as f:
        json.dump(consensus_result, f, indent=4)


    # Sort again after filling missing items
    all_results_sorted = sorted(all_results, key=lambda x: x['id'])

    y_true = [item['response'] for item in full_gt_sorted]
    y_pred = [item['response'] for item in all_results_sorted]

    # Compute balanced accuracy
    score = balanced_accuracy_score(y_true, y_pred)

    _, _, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    print("{}\t {}\t {:.2f}\t {:.2f}".format(model, datasets[0], score, f1_score))
