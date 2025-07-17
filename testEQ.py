import args
from embeddings_quality import embeddings_quality
import os
def remove_prefixes(edgelist_file, model_file):
    newf, _ = os.path.splitext(model_file)
    newf += "_cleaned.emb"

    with open(edgelist_file) as fp:
        node_types = fp.readline().strip().split(",")
        prefixes = [_.split("__")[1] for _ in node_types]

    with open(model_file, "r",encoding='utf-8') as fin, open(newf, "w") as fo:
        for idx, line in enumerate(fin):
            if idx >0:
                split = line.split("__", maxsplit=1)
                if len(split) == 2:
                    pre, rest = split
                    if pre in prefixes:
                        fo.write(rest)
                    else:
                        fo.write(line)
                else:
                    fo.write(line)
            else:
                fo.write(line)


    return newf

def test(embeddings_file):
    input_file = args.dataset
    test_dir = args.test_dir
    newf = remove_prefixes(input_file, embeddings_file) 
    embeddings_quality(newf, test_dir)
    os.remove(newf)

if __name__ == '__main__':
    run_tag = args.dataset.split('\\')[-1].split('-')[0]
    embeddings_path = 'embeddings/' + run_tag + '.emb'
    # if args.test_type == 'EQ':
    test(embeddings_path)