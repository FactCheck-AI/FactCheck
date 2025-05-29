import json

from sklearn.metrics import balanced_accuracy_score

methods = ['DKA', 'GIV-Z', 'GIV-F', 'RAG']

datasets = ['FactBench', 'YAGO', 'DBpedia']
models = ['gemma2:9b', 'qwen2.5:7b', 'llama3.1:8b', 'mistral:7b']

full_gt=[]
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
    for model in models:
        all_results = []
        for dataset in datasets:
            # print(f"Processing {method} on {dataset} with model {model}")
            with open(f'results/paper_results/{method}/{dataset}_open-source_{model}_{method.lower()}.json', 'r') as f:
                res_file = json.load(f)
            for result in res_file:
                all_results.append({
                    'id': result['id'] if dataset == 'FactBench' else f"{dataset.lower()}_{result['id']}",
                    'response': 1 if result['response'] == "T" else 0,
                })

        # check if the key not in all_results
        all_results_ids = [item['id'] for item in all_results]
        if len(all_results) != len(full_gt_sorted):
            # print(f"Warning: Length mismatch for {method} with model {model}: {len(all_results)} vs {len(full_gt_sorted)}")
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



