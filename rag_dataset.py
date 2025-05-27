import os


# walk over the directory
for _, dirs, _ in os.walk('/Users/farzad/Documents/Thesis/RagKocholo-v2/docs'):
    for di in dirs:
        if di.startswith(('correct_', 'wrong_', 'dbpedia_', 'yago_')):
            if di.startswith(('dbpedia_', 'yago_')):
                dataset, identifier = di.split('_', 1)
            else:
                dataset, identifier = 'factbench', di
            file_path = f'/Users/farzad/Projects/FactCheck/rag_dataset/{dataset}/{identifier}'
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            os.system(f'cp -r /Users/farzad/Documents/Thesis/RagKocholo-v2/docs/{di}/re-ranker-msmarco/*.txt {file_path}')