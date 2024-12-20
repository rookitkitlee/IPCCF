import torch.optim as optim
import random
import logging
import datetime
import os
from utility.parser import parse_args
from utility.batch_test import *
from utility.load_data import *
from tqdm import tqdm
from time import time
from copy import deepcopy
from model import DR

args = parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 检测nan
# torch.autograd.set_detect_anomaly(True)


def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list


if __name__ == '__main__':

    """
    *********************************************************
    Prepare the log file
    """
    curr_time = datetime.datetime.now()
    if not os.path.exists('log'):
        os.mkdir('log')
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(
        'log/{}.log'.format(args.dataset), 'a', encoding='utf-8')
    logfile.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    """
    *********************************************************
    Prepare the dataset
    """
    data_generator = Data(args)
    logger.info(data_generator.get_statistics())

    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    logger.info(args)
    print("************************************************************************************")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['train_mat'] = data_generator.train_mat
    config['uu_mat'] = data_generator.uu_mat
    config['ii_mat'] = data_generator.ii_mat

    """
    *********************************************************
    Generate the adj matrix
    """
    plain_adj = data_generator.get_adj_mat()
    all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)
    config['plain_adj'] = plain_adj
    config['all_h_list'] = all_h_list
    config['all_t_list'] = all_t_list

    config['uu_h_list'] = data_generator.uu_mat['row']
    config['uu_t_list'] = data_generator.uu_mat['col']
    config['uu_data'] = data_generator.uu_mat['data']

    config['ii_h_list'] = data_generator.ii_mat['row']
    config['ii_t_list'] = data_generator.ii_mat['col']
    config['ii_data'] = data_generator.ii_mat['data']

    config['intent_normalize'] = args.intent_normalize
    # if args.dataset == "tmall" or args.dataset == "amazon":
    #     config['intent_normalize'] = False

    """
    *********************************************************
    Prepare the model and start training
    """
    _model = DR(config, args).cuda()
    optimizer = optim.Adam(_model.parameters(), lr=args.lr)

    print("Start Training")

    save_model_path = args.data_path + args.dataset + \
        "/model_" + str(args.n_intents) + ".pth"

    best_score = 0
    best_epoch = 0
    best_result = None
    early_num = 0

    for epoch in range(args.epoch):

        # train
        t1 = time()

        n_samples = data_generator.uniform_sample()
        n_batch = int(np.ceil(n_samples / args.batch_size))

        _model.train()
        loss, mf_loss, emb_loss, cen_loss, dis_loss, cl_loss = 0., 0., 0., 0., 0., 0.
        for idx in tqdm(range(n_batch)):

            optimizer.zero_grad()

            users, pos_items, neg_items = data_generator.mini_batch(idx)
            batch_mf_loss, batch_emb_loss, batch_cen_loss, batch_dis_loss, batch_cl_loss = _model(
                users, pos_items, neg_items)
            batch_loss = batch_mf_loss + batch_emb_loss + \
                batch_cen_loss + batch_dis_loss + batch_cl_loss

            loss += float(batch_loss) / n_batch
            mf_loss += float(batch_mf_loss) / n_batch
            emb_loss += float(batch_emb_loss) / n_batch
            cen_loss += float(batch_cen_loss) / n_batch
            dis_loss += float(batch_dis_loss) / n_batch
            cl_loss += float(batch_cl_loss) / n_batch

            batch_loss.backward()
            optimizer.step()

        # update the saved model parameters after each epoch
        last_state_dict = deepcopy(_model.state_dict())
        torch.cuda.empty_cache()

        if epoch % args.show_step != 0 and epoch != args.epoch - 1:
            perf_str = 'Epoch %2d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss, cen_loss, dis_loss, cl_loss)
            print(perf_str)
            logger.info(perf_str)
            continue

        t2 = time()

        # test the model on test set for observation
        with torch.no_grad():
            _model.eval()
            _model.inference()
            test_ret = eval_PyTorch(_model, data_generator, eval(args.Ks))
            torch.cuda.empty_cache()

        t3 = time()

        perf_str = 'Epoch %2d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f], test-recall=[%.4f, %.4f], test-ndcg=[%.4f, %.4f]' % \
                   (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, cen_loss, dis_loss, cl_loss,
                    test_ret['recall'][0], test_ret['recall'][1], test_ret['ndcg'][0], test_ret['ndcg'][1])
        print(perf_str)

        logger.info(perf_str)

        score = test_ret['recall'][0] + test_ret['recall'][1] + \
            test_ret['ndcg'][0] + test_ret['ndcg'][1]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_result = test_ret
            early_num = 0

            if args.save_model:
                torch.save(_model, save_model_path)
        else:
            early_num = early_num + 1

        if early_num >= args.early_stop:
            break

    # best_result
    pref_str = 'Best Result at Epoch %2d: test-recall=[%.4f, %.4f], test-ndcg=[%.4f, %.4f]' % (
        best_epoch, best_result['recall'][0], best_result['recall'][1], best_result['ndcg'][0], best_result['ndcg'][1])
    print(pref_str)
    logger.info(pref_str)
