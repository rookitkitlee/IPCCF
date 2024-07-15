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
from utility.NumberKit import NumberKit
import scipy.sparse as sp
import numpy as np
import pickle
from utility.IOKit import *

args = parse_args()
ui_k = int(args.ui_k)

if __name__ == '__main__':

    data_generator = Data(args)
    n_users = data_generator.n_users
    n_items = data_generator.n_items

    train_mat = data_generator.train_mat
    rows = train_mat.row
    cols = train_mat.col

    usets = {}
    isets = {}

    for index in range(len(rows)):

        u1 = rows[index]
        i1 = cols[index]

        NumberKit.KeySetAddData(usets, u1, i1)
        NumberKit.KeySetAddData(isets, i1, u1)

    uu_row = []
    uu_col = []
    uu_value = []

    for u1 in range(n_users):

        d1 = [0 for _ in range(n_users)]
        sd2 = []

        if u1 % 100 == 0:
            print("uu:"+str(u1)+"/"+str(n_users))

        for u2 in range(n_users):

            if u1 == u2:
                continue

            if u1 not in usets.keys() or u2 not in usets.keys():
                continue

            u1s = usets[u1]
            u2s = usets[u2]

            inters = u1s.intersection(u2s)
            unions = u1s.union(u2s)
            value = len(inters) / len(unions)
            d1[u2] = value

            if value > 0.8:
                sd2.append(u2)


        sd1 = np.argsort(d1)[::-1][:ui_k]
        for d in sd1:
            u2 = d
            v = d1[u2]

            uu_row.append(u1)
            uu_col.append(u2)
            uu_value.append(v)

        for d in sd2:
            if d in sd1:
                continue
            u2 = d
            v = d1[u2]

            uu_row.append(u1)
            uu_col.append(u2)
            uu_value.append(v)

    uu_mat = {}
    uu_mat['row'] = uu_row
    uu_mat['col'] = uu_col
    uu_mat['data'] = uu_value

    path = args.data_path + args.dataset
    uu_path = path + "/uu" + str(ui_k) + ".pkl"
    write_pkl(uu_path, uu_mat)
    # picklesave(uu_mat, uu_path)

