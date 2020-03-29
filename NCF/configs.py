
# General Arguments
dataset = 'ml-1m'
verbose = 1
topK = 10
out = 1

"""
# GMF
epochs = 2
batch_size = 256
num_negatives = 4
num_factors = 8
learner = 'adam'
learning_rate = 0.001
regs = [0, 0]
"""

"""
# MLP
epochs = 2
batch_size = 256
layers = [64, 32, 16, 8]
reg_layers = [0, 0, 0, 0]
num_negatives = 4
learning_rate = 0.001
learner = 'adam'
"""


# NeuMF
epochs = 20
batch_size = 256
mf_dim = 8
reg_mf = [0, 0]
layers = [64, 32, 16, 8]
reg_layers = [0, 0, 0, 0]
num_negatives = 4
learning_rate = 0.001
learner = 'adam'
num_factors = 8
mf_pretrain = 'Pretrain/ml-1m_GMF_8_1575894502.h5'
mlp_pretrain = 'Pretrain/ml-1m_MLP_[64, 32, 16, 8]_1575898018.h5'

