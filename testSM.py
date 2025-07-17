import args
from schema_matching import schema_matching
# run_tag = args.dataset.split('\\')[-1].split('-')[0]
# embeddings_path = 'embeddings/' + run_tag + '.emb'

if __name__  == '__main__':
    # if args.test_type == 'ER':
    #     embeddings_file = args.embeddings_file
        embeddings_file = r'the path of embeddings_file'
        schema_matching(embeddings_file)
