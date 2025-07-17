### CONFIGS ###
test_type = 'ER'

name = 'amazon_google'
# name = 'beer'
# name = 'dblp_acm'
# name = 'fodors_zagats'
# name = 'dblp_scholar'
# name = 'imdb_movielens'
# name = 'itunes_amazon'
# name = 'walmart_amazon'


dataset = 'the dataset path'
dataset_info = 'the dataset info path'
embeddings_file = 'the path of embeddings_file'
model = 'TGAE'
run_tag = dataset.split('\\')[-1].split('-')[0] + '-' + test_type
test_dir = 'the path of test_dir'
sm_dataset = 'the path of sm_dataset'
input_dim = 0
hidden1_dim = 32
hidden2_dim = 300
use_feature = True

num_epoch = 100
learning_rate = 0.01
ntop = 10
ncand = 1
indexing = 'basic'
match_file = 'the path of match_file'
epsilon = 0.1
num_trees = 250