import args
from entity_resolution import entity_resolution
# run_tag = args.dataset.split('\\')[-1].split('-')[0]
# embeddings_path = 'embeddings/' + run_tag + '.emb'

if __name__  == '__main__':
    # if args.test_type == 'ER':
        embeddings_file = args.embeddings_file
        info_file = args.dataset_info
        entity_resolution(embeddings_file, info_file=info_file)
