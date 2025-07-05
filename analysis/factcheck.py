import json

from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

methods = ['DKA', 'GIV-Z', 'GIV-F', 'RAG']

datasets = ['FactBench', 'YAGO', 'DBpedia']
individual_models = ['gemma2:9b', 'qwen2.5:7b', 'llama3.1:8b', 'mistral:7b', 'gpt-4o-mini']
consensus_models = ['cons-up', 'cons-down', 'gpt-4o-mini']

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

for method in methods:
    for model in individual_models:
        all_results = []
        for dataset in datasets:
            with open(f'results/paper_results/{method}/{dataset}_{"open-source" if model!="gpt-4o-mini" else "commercial"}_{model}_{method.lower()}.json', 'r') as f:
                res_file = json.load(f)
            for result in res_file:
                all_results.append({
                    'id': result['id'] if dataset == 'FactBench' else f"{dataset.lower()}_{result['id']}",
                    'response': 1 if result['response'] == "T" else 0,
                })

        # check if the key not in all_results
        all_results_ids = [item['id'] for item in all_results]
        if len(all_results) != len(full_gt_sorted):
            # fill with the reversed full_gt_sorted, iterate over all_results_sorted and fill the missing ones
            for item in full_gt_sorted:
                if item['id'] not in all_results_ids:
                    all_results.append({
                        'id': item['id'],
                        'response': 0 if item['response'] == 1 else 1
                    })
        # Sort again after filling missing items
        all_results_sorted = sorted(all_results, key=lambda x: x['id'])


        y_true = [item['response'] for item in full_gt_sorted]
        y_pred = [item['response'] for item in all_results_sorted]

        # Compute balanced accuracy
        score = balanced_accuracy_score(y_true, y_pred)
        print("{}\t {}\t {:.2f}".format(method, model, score))

for method in methods:
    for model in consensus_models:
        all_results = []
        for dataset in datasets:
            with open(f'results/paper_results/consensus/{method}/{dataset}_majority-vote_{model}_modified_{method.lower()}.json', 'r') as f:
                res_file = json.load(f)
            for result in res_file['results']:
                all_results.append({
                    'id': result['id'] if dataset == 'FactBench' else f"{dataset.lower()}_{result['id']}",
                    'response': 1 if result['majority_decision'] == "T" else 0,
                })

        # check if the key not in all_results
        all_results_ids = [item['id'] for item in all_results]
        if len(all_results) != len(full_gt_sorted):
            # fill with the reversed full_gt_sorted, iterate over all_results_sorted and fill the missing ones
            for item in full_gt_sorted:
                if item['id'] not in all_results_ids:
                    all_results.append({
                        'id': item['id'],
                        'response': 0 if item['response'] == 1 else 1
                    })
        # Sort again after filling missing items
        all_results_sorted = sorted(all_results, key=lambda x: x['id'])


        y_true = [item['response'] for item in full_gt_sorted]
        y_pred = [item['response'] for item in all_results_sorted]

        # Compute balanced accuracy
        score = balanced_accuracy_score(y_true, y_pred)
        print("{}\t {}\t {:.2f}".format(method, f'agg_{model}', score))

#
# models = ['cons-up', 'cons-down']
#
# with open('results/paper_results/consensus/full_higher_parameter_df_results.json', 'r', encoding='utf-8') as f:
#     full_higher_parameter_df_results = json.load(f)
#
# hp_res = {'cons-up': {}, 'cons-down': {}}
# # for item in full_higher_parameter_df_results:
# #     if item['consistency'] == 'high':
# #         hp_res['cons-up'][f"{item['Mode']}_{item['Custom_id']}"] = item
# #     else:
# #         hp_res['cons-down'][f"{item['Mode']}_{item['Custom_id']}"] = item
#
# for kossher in ['least', 'most']:
#     for d in ["DBpedia", "YAGO", "FactBench"]:
#         with open(f'results/paper_results/consensus/RAG/{d}_final-eval-{kossher}_best_model.json', 'r', encoding='utf-8') as f:
#             rag_results = json.load(f)
#             for rr_key, rr_value in rag_results.items():
#                 item = {'Answer': "T" if rr_value['short_ans'] else "F"}
#                 r_key = f"{d.lower()}_{rr_key}" if d != 'FactBench' else rr_key
#                 if kossher == 'least':
#                     hp_res['cons-down'][f"rag_{r_key}"] = item
#                 else:
#                     hp_res['cons-up'][f"rag_{r_key}"] = item
#
#
#
#
#
# jsonl_file = 'RAG.jsonl'
# prefix = 'rag'
#
# for model in models:
#     with open(f'results/paper_results/consensus/RAG/{datasets[0]}_majority-vote_{model}_rag.json', 'r') as f:
#         consensus_result = json.load(f)
#
#     comp = {}
#     if model == 'gpt-4o-mini':
#         with open(f'results/paper_results/consensus/RAG/{jsonl_file}', 'r', encoding='utf-8') as f:
#             for line in f:
#                 lj = json.loads(line.strip())
#                 comp[lj['custom_id']] = 1 if 'yes' in lj['response']['body']['choices'][0]['message']['content'] else 0
#
#     all_results = []
#     for idx, cr in enumerate(consensus_result['results']):
#         identifier = cr['id']
#         response = cr['majority_decision']
#         if response == -5:
#             if model == 'gpt-4o-mini':
#                 # response = comp.get(f'{datasets[0]}_{identifier}')
#                 response = comp.get(f'{identifier}')
#                 if response is None:
#                     # response = comp.get(f'{datasets[0].lower()}_{identifier}')
#                     response = comp.get(f'{identifier}')
#             else:
#                 response = hp_res.get(model, {}).get(f'{prefix}_{datasets[0].lower()}_{identifier}', {}).get('Answer', 'F').strip()
#                 # response = hp_res.get(model, {}).get(f'{prefix}_{identifier}', {}).get('Answer', 'F').strip()
#                 response = 1 if "T" in response else 0
#             consensus_result['results'][idx]['majority_decision'] = "T" if response else "F"
#         elif response == "T":
#             response = 1
#         else:
#             response = 0
#
#         all_results.append({
#             'id': identifier,
#             'response': response,
#         })
#
#     # check if the key not in all_results
#     all_results_ids = [f"{datasets[0].lower()}_{item['id']}" for item in all_results]
#     print('difference', len(all_results_ids) - len(full_gt_sorted))
#     if len(all_results) != len(full_gt_sorted):
#         # print(f"Warning: Length mismatch for {method} with model {model}: {len(all_results)} vs {len(full_gt_sorted)}")
#         # fill with the reversed full_gt_sorted, iterate over all_results_sorted and fill the missing ones
#         for item in full_gt_sorted:
#             if item['id'] not in all_results_ids:
#                 print(f"Filling missing item: {item['id']}")
#                 all_results.append({
#                     'id': item['id'].split("_")[1],
#                     'response': 0 if item['response'] == 1 else 1
#                 })
#
#     with open(f'results/paper_results/consensus/RAG/{datasets[0]}_majority-vote_{model}_modified_rag.json', 'w') as f:
#         json.dump(consensus_result, f, indent=4)
#
#
#     # Sort again after filling missing items
#     all_results_sorted = sorted(all_results, key=lambda x: x['id'])
#
#     y_true = [item['response'] for item in full_gt_sorted]
#     y_pred = [item['response'] for item in all_results_sorted]
#
#     # Compute balanced accuracy
#     score = balanced_accuracy_score(y_true, y_pred)
#
#     _, _, f1_score, _ = precision_recall_fscore_support(
#         y_true, y_pred, average='macro', zero_division=0
#     )
#
#     print("{}\t {}\t {:.2f}\t {:.2f}".format(model, datasets[0], score, f1_score))

# Method   Model            Score
# DKA	 gemma2:9b	 0.70
# DKA	 qwen2.5:7b	 0.65
# DKA	 llama3.1:8b	 0.66
# DKA	 mistral:7b	 0.68
# DKA	 gpt-4o-mini	 0.64
# GIV-Z	 gemma2:9b	 0.70
# GIV-Z	 qwen2.5:7b	 0.65
# GIV-Z	 llama3.1:8b	 0.62
# GIV-Z	 mistral:7b	 0.64
# GIV-Z	 gpt-4o-mini	 0.63
# GIV-F	 gemma2:9b	 0.71
# GIV-F	 qwen2.5:7b	 0.68
# GIV-F	 llama3.1:8b	 0.65
# GIV-F	 mistral:7b	 0.65
# GIV-F	 gpt-4o-mini	 0.61
# RAG	 gemma2:9b	 0.75
# RAG	 qwen2.5:7b	 0.74
# RAG	 llama3.1:8b	 0.69
# RAG	 mistral:7b	 0.72
# RAG	 gpt-4o-mini	 0.75
# DKA	 cons-up	 0.70
# DKA	 cons-down	 0.70
# DKA	 gpt-4o-mini	 0.70
# GIV-Z	 cons-up	 0.70
# GIV-Z	 cons-down	 0.70
# GIV-Z	 gpt-4o-mini	 0.69
# GIV-F	 cons-up	 0.72
# GIV-F	 cons-down	 0.73
# GIV-F	 gpt-4o-mini	 0.72
# RAG	 cons-up	 0.75
# RAG	 cons-down	 0.75
# RAG	 gpt-4o-mini	 0.75
#--------
# Model   Dataset     Method            Score

# Gemma2:9b	 FactBench     DKA	 0.75
# Gemma2:9b	 FactBench     GIV-Z	 0.74
# Gemma2:9b	 FactBench     GIV-F	 0.77
# Gemma2:9b	 FactBench     RAG	 0.90
# Gemma2:9b	 YAGO       DKA	 0.53
# Gemma2:9b	 YAGO       GIV-Z	 0.58
# Gemma2:9b	 YAGO       GIV-F	 0.52
# Gemma2:9b	 YAGO       RAG	 0.56
# Gemma2:9b	 DBpedia     DKA	 0.64
# Gemma2:9b	 DBpedia     GIV-Z	 0.65
# Gemma2:9b	 DBpedia     GIV-F	 0.63
# Gemma2:9b	 DBpedia     RAG	 0.67

# Qwen2.5:7b	 FactBench     DKA	 0.67
# Qwen2.5:7b	 FactBench     GIV-Z	 0.65
# Qwen2.5:7b	 FactBench     GIV-F	 0.74
# Qwen2.5:7b	 FactBench     RAG	 0.87
# Qwen2.5:7b	 YAGO       DKA	 0.59
# Qwen2.5:7b	 YAGO       GIV-Z	 0.64
# Qwen2.5:7b	 YAGO       GIV-F	 0.64
# Qwen2.5:7b	 YAGO       RAG	 0.57
# Qwen2.5:7b	 DBpedia     DKA	 0.63
# Qwen2.5:7b	 DBpedia     GIV-Z	 0.63
# Qwen2.5:7b	 DBpedia     GIV-F	 0.65
# Qwen2.5:7b	 DBpedia     RAG	 0.67

# Llama3.1:8b	 FactBench     DKA	 0.74
# Llama3.1:8b	 FactBench     GIV-Z	 0.65
# Llama3.1:8b	 FactBench     GIV-F	 0.73
# Llama3.1:8b	 FactBench     RAG	 0.82
# Llama3.1:8b	 YAGO       DKA	 0.55
# Llama3.1:8b	 YAGO       GIV-Z	 0.59
# Llama3.1:8b	 YAGO       GIV-F	 0.58
# Llama3.1:8b	 YAGO       RAG	 0.51
# Llama3.1:8b	 DBpedia     DKA	 0.58
# Llama3.1:8b	 DBpedia     GIV-Z	 0.60
# Llama3.1:8b	 DBpedia     GIV-F	 0.62
# Llama3.1:8b	 DBpedia     RAG	 0.62

# Mistral:7b	 FactBench     DKA	 0.72
# Mistral:7b	 FactBench     GIV-Z	 0.74
# Mistral:7b	 FactBench     GIV-F	 0.77
# Mistral:7b	 FactBench     RAG	 0.84
# Mistral:7b	 YAGO       DKA	 0.44
# Mistral:7b	 YAGO       GIV-Z	 0.53
# Mistral:7b	 YAGO       GIV-F	 0.46
# Mistral:7b	 YAGO       RAG	 0.51
# Mistral:7b	 DBpedia     DKA	 0.63
# Mistral:7b	 DBpedia     GIV-Z	 0.55
# Mistral:7b	 DBpedia     GIV-F	 0.54
# Mistral:7b	 DBpedia     RAG	 0.66

# Gpt-4o-mini	 FactBench     DKA	 0.66
# Gpt-4o-mini	 FactBench     GIV-Z	 0.65
# Gpt-4o-mini	 FactBench     GIV-F	 0.65
# Gpt-4o-mini	 FactBench     RAG	 0.90
# Gpt-4o-mini	 YAGO       DKA	 0.57
# Gpt-4o-mini	 YAGO       GIV-Z	 0.58
# Gpt-4o-mini	 YAGO       GIV-F	 0.63
# Gpt-4o-mini	 YAGO       RAG	 0.54
# Gpt-4o-mini	 DBpedia     DKA	 0.61
# Gpt-4o-mini	 DBpedia     GIV-Z	 0.61
# Gpt-4o-mini	 DBpedia     GIV-F	 0.59
# Gpt-4o-mini	 DBpedia     RAG	 0.67

# Cons-up	 FactBench     DKA	 0.73
# Cons-up	 FactBench     GIV-Z	 0.76
# Cons-up	 FactBench     GIV-F	 0.80
# Cons-up	 FactBench     RAG	 0.90
# Cons-up	 YAGO       DKA	 0.53
# Cons-up	 YAGO       GIV-Z	 0.64
# Cons-up	 YAGO       GIV-F	 0.54
# Cons-up	 YAGO       RAG	 0.53
# Cons-up	 DBpedia     DKA	 0.64
# Cons-up	 DBpedia     GIV-Z	 0.67
# Cons-up	 DBpedia     GIV-F	 0.67
# Cons-up	 DBpedia     RAG	 0.67

# Cons-down	 FactBench     DKA	 0.73
# Cons-down	 FactBench     GIV-Z	 0.71
# Cons-down	 FactBench     GIV-F	 0.80
# Cons-down	 FactBench     RAG	 0.90
# Cons-down	 YAGO       DKA	 0.55
# Cons-down	 YAGO       GIV-Z	 0.60
# Cons-down	 YAGO       GIV-F	 0.55
# Cons-down	 YAGO       RAG	 0.54
# Cons-down	 DBpedia     DKA	 0.66
# Cons-down	 DBpedia     GIV-Z	 0.66
# Cons-down	 DBpedia     GIV-F	 0.66
# Cons-down	 DBpedia     RAG	 0.68

# Agg-Gpt-4o-mini	 FactBench     DKA	 0.74
# Agg-Gpt-4o-mini	 FactBench     GIV-Z	 0.71
# Agg-Gpt-4o-mini	 FactBench     GIV-F	 0.80
# Agg-Gpt-4o-mini	 FactBench     RAG	 0.90
# Agg-Gpt-4o-mini	 YAGO       DKA	 0.49
# Agg-Gpt-4o-mini	 YAGO       GIV-Z	 0.60
# Agg-Gpt-4o-mini	 YAGO       GIV-F	 0.55
# Agg-Gpt-4o-mini	 YAGO       RAG	 0.53
# Agg-Gpt-4o-mini	 DBpedia     DKA	 0.66
# Agg-Gpt-4o-mini	 DBpedia     GIV-Z	 0.66
# Agg-Gpt-4o-mini	 DBpedia     GIV-F	 0.66
# Agg-Gpt-4o-mini	 DBpedia     RAG	 0.67