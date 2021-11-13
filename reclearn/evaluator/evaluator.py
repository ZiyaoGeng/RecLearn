import logging

from reclearn.evaluator.metrics import *

logging.getLogger(__name__).setLevel(logging.INFO)


def eval_pos_neg(model, test_data, batch_size, metric_name, k):
    pred_y = - model.predict(test_data, batch_size)
    rank = pred_y.argsort().argsort()[:, 0]
    res_dict = {}
    info = "Evaluate Result: "
    for name in metric_name:
        if name == 'hr':
            res = hr(rank, k)
        elif name == 'ndcg':
            res = ndcg(rank, k)
        elif name == 'mrr':
            res = ndcg(rank, k)
        else:
            break
        res_dict[name] = res
        info += "%s@%s: %s, ".format(name, k, res)
    logging.info(info[:-2])
    return res_dict