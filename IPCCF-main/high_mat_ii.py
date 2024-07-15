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
 
    
    ii_row = []
    ii_col = []
    ii_value = []

    for i1 in range(n_items):

        d1 = [0 for _ in range(n_items)]
        sd2 = []

        if i1 % 100 == 0:
            print("ii:"+str(i1)+"/"+str(n_items))

        for i2 in range(n_items):

            if i1 == i2:
                continue

            if i1 not in isets.keys() or i2 not in isets.keys():
                continue

            i1s = isets[i1]
            i2s = isets[i2]

            inters = i1s.intersection(i2s)
            unions = i1s.union(i2s)
            value = len(inters) / len(unions)
            d1[i2] = value

            if value > 0.8:
                sd2.append(i2)


        sd1 = np.argsort(d1)[::-1][:ui_k]
        for d in sd1:
            i2 = d
            v = d1[i2]

            ii_row.append(i1)
            ii_col.append(i2)
            ii_value.append(v)

        for d in sd2:
            if d in sd1:
                continue
            i2 = d
            v = d1[i2]

            ii_row.append(i1)
            ii_col.append(i2)
            ii_value.append(v)

 
    ii_mat = {}
    ii_mat['row'] = ii_row
    ii_mat['col'] = ii_col
    ii_mat['data'] = ii_value

    path = args.data_path + args.dataset
    ii_path = path + "/ii" + str(ui_k) + ".pkl"
    write_pkl(ii_path, ii_mat)
