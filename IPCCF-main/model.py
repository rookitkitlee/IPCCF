import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F




# 整理一下代码
class DR(nn.Module):
    def __init__(self, data_config, args):
        super(DR, self).__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.intent_normalize = data_config['intent_normalize']

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor(
            [self.all_h_list, self.all_t_list], dtype=torch.long).to(self.device)
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(
            range(self.n_users + self.n_items))], dtype=torch.long).to(self.device)
        self.all_h_list = torch.LongTensor(self.all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(self.all_t_list).to(self.device)
        self.G_indices, self.G_values = self._cal_sparse_adj()

        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.n_intents = args.n_intents
        self.temp = args.temp

        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg
        self.cen_reg = args.cen_reg
        self.cen_dis = args.cen_dis
        self.ssl_reg = args.ssl_reg

        """
        *********************************************************
        user_item interaction
        """
        self.n_trains = len(data_config['train_mat'].row)
        user_ut = []
        train_ut = []
        for index, value in enumerate(data_config['train_mat'].row):
            user_ut.append(value)
            train_ut.append(index)

        item_it = []
        train_it = []
        for index, value in enumerate(data_config['train_mat'].col):
            item_it.append(value)
            train_it.append(index)

        self.UT_indices = torch.tensor(
            [user_ut, train_ut], dtype=torch.long).to(self.device)
        self.TU_indices = torch.tensor(
            [train_ut, user_ut], dtype=torch.long).to(self.device)
        self.UT_values = torch.tensor(
            [1 for _ in range(self.n_trains)], dtype=torch.long).to(self.device)

        self.IT_indices = torch.tensor(
            [item_it, train_it], dtype=torch.long).to(self.device)
        self.TI_indices = torch.tensor(
            [train_it, item_it], dtype=torch.long).to(self.device)
        self.IT_values = torch.tensor(
            [1 for _ in range(self.n_trains)], dtype=torch.long).to(self.device)

        self.h_list = torch.LongTensor(
            data_config['train_mat'].row).to(self.device)
        self.t_list = torch.LongTensor(
            data_config['train_mat'].col).to(self.device)

        """
        *********************************************************
        user_user interaction and item_item interaction
        """
        self.uu_h_list = torch.LongTensor(
            data_config['uu_h_list']).to(self.device)
        self.uu_t_list = torch.LongTensor(
            data_config['uu_t_list']).to(self.device)
        self.uu_data = torch.FloatTensor(
            data_config['uu_data']).to(self.device)
        self.uu_indices = torch.tensor(
            [data_config['uu_h_list'], data_config['uu_t_list']], dtype=torch.long).to(self.device)
        self.uu_data = self.normalization_indices_values(
            self.uu_h_list, self.uu_t_list, self.uu_data, (self.n_users, self.n_users))

        self.ii_h_list = torch.LongTensor(
            data_config['ii_h_list']).to(self.device)
        self.ii_t_list = torch.LongTensor(
            data_config['ii_t_list']).to(self.device)
        self.ii_data = torch.FloatTensor(
            data_config['ii_data']).to(self.device)
        self.ii_indices = torch.tensor(
            [data_config['ii_h_list'], data_config['ii_t_list']], dtype=torch.long).to(self.device)
        self.ii_data = self.normalization_indices_values(
            self.ii_h_list, self.ii_t_list, self.ii_data, (self.n_items, self.n_items))

        self.n_uu_trains = len(data_config['uu_h_list'])
        user_uu_ut = []
        train_uu_ut = []
        for index, value in enumerate(data_config['uu_h_list']):
            user_uu_ut.append(value)
            train_uu_ut.append(index)

        self.n_ii_trains = len(data_config['ii_h_list'])
        item_ii_it = []
        train_ii_it = []
        for index, value in enumerate(data_config['ii_h_list']):
            item_ii_it.append(value)
            train_ii_it.append(index)

        self.UUT_indices = torch.tensor(
            [user_uu_ut, train_uu_ut], dtype=torch.long).to(self.device)
        self.TUU_indices = torch.tensor(
            [train_uu_ut, user_uu_ut], dtype=torch.long).to(self.device)

        self.IIT_indices = torch.tensor(
            [item_ii_it, train_ii_it], dtype=torch.long).to(self.device)
        self.TII_indices = torch.tensor(
            [train_ii_it, item_ii_it], dtype=torch.long).to(self.device)

        """
        *********************************************************
        Create Model Parameters
        """
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)

        _intents = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_intents)
        self.intents = torch.nn.Parameter(_intents, requires_grad=True)

        """
        *********************************************************
        Initialize Weights
        """
        self._init_weight()

        self.softplus = nn.Softplus(beta=0.5, threshold=20)

        self.mlp1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.Sigmoid()
        ).to(self.device)

        self.mlp2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.emb_dim * 4, self.emb_dim),
            nn.Sigmoid()
        ).to(self.device)

        self.mlp3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.emb_dim * 6, self.emb_dim),
            nn.Sigmoid()
        ).to(self.device)

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def _cal_sparse_adj(self):

        A_values = torch.ones(size=(len(self.all_h_list), 1)
                              ).view(-1).to(self.device)

        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).to(self.device)
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(
            self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(
            G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])

        return G_indices, G_values

    def normalization_indices_values(self, h_list, t_list, value, shape):

        A_tensor = torch_sparse.SparseTensor(
            row=h_list, col=t_list, value=value, sparse_sizes=shape)
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        return D_scores_inv[h_list] * value

    def embed11(self, input):

        return torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], input)

    def embed12(self, input):

        u_embeddings, i_embeddings = torch.split(input, [self.n_users, self.n_items], 0)
        uu_embeddings = torch_sparse.spmm(self.uu_indices, self.uu_data, self.n_users, self.n_users, u_embeddings)
        ii_embeddings = torch_sparse.spmm(self.ii_indices, self.ii_data, self.n_items, self.n_items, i_embeddings)
        return torch.cat((uu_embeddings, ii_embeddings), dim=0)

    def embed21(self, feature, intents, input):

        head_embeddings = torch.index_select(feature, 0, self.h_list)
        tail_embeddings = torch.index_select(feature, 0, self.t_list)
        edge_distributions = torch.softmax((head_embeddings * tail_embeddings) @ intents, dim=1)  # edge * intent
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        edge_alpha = edge_alpha.flatten()

        user_intents = torch_sparse.spmm(self.UT_indices, edge_alpha, self.n_users, self.n_trains, edge_distributions)  # user * intent
        user_intents = torch.div(torch.ones_like(user_intents), user_intents)  # 取倒数 user * intent
        edge_user_intents = torch_sparse.spmm(self.TU_indices, edge_alpha, self.n_trains, self.n_users, user_intents)  # edge * intent
        edge_user_distribution = edge_distributions * edge_user_intents  # edge * intent
        edge_user_distribution = torch.mean(edge_user_distribution, dim=1, keepdim=False)  # edge * 1
        item_intents = torch_sparse.spmm(self.IT_indices, edge_alpha, self.n_items, self.n_trains, edge_distributions)  # item * intent
        item_intents = torch.div(torch.ones_like(item_intents), item_intents)  # 取倒数 item * intent
        edge_item_intents = torch_sparse.spmm(self.TI_indices, edge_alpha, self.n_trains, self.n_items, item_intents)  # item * intent
        edge_item_distribution = edge_distributions * edge_item_intents
        edge_item_distribution = torch.mean(edge_item_distribution, dim=1, keepdim=False)  # edge * 1
        edge_distributions = torch.cat((edge_user_distribution, edge_item_distribution), dim=0)
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_distributions, sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).nan_to_num( 0, 0, 0).pow(-1).nan_to_num(0, 0, 0).view(-1)
        fdr_edge_distributions = D_scores_inv[self.all_h_list] * edge_distributions
        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)

        fdr_layer_embeddings = torch_sparse.spmm(G_indices, fdr_edge_distributions, self.A_in_shape[0], self.A_in_shape[1], input)
        return torch.nn.functional.normalize(fdr_layer_embeddings)

    def embed22(self, feature, intents, input):

        # Second Order Disentangled Representation
        ux_embeddings, ix_embeddings = torch.split(
            feature, [self.n_users, self.n_items], 0)
        # uu edge intent
        head_embeddings = torch.index_select(ux_embeddings, 0, self.uu_h_list)
        tail_embeddings = torch.index_select(ux_embeddings, 0, self.uu_t_list)
        edge_distributions = torch.softmax((head_embeddings * tail_embeddings) @ intents, dim=1)  # edge * intent
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        edge_alpha = edge_alpha.flatten()
        user_intents = torch_sparse.spmm(self.UUT_indices, edge_alpha, self.n_users, self.n_uu_trains, edge_distributions)  # user * intent
        user_intents = torch.div(torch.ones_like(user_intents), user_intents)  # 取倒数 user * intent
        edge_user_intents = torch_sparse.spmm(self.TUU_indices, edge_alpha, self.n_uu_trains, self.n_users, user_intents)  # edge * intent
        edge_user_distribution = edge_distributions * edge_user_intents  # edge * intent
        edge_user_distribution = torch.mean(edge_user_distribution, dim=1, keepdim=False)  # edge * 1
        edge_user_distribution = self.normalization_indices_values(self.uu_h_list, self.uu_t_list, edge_user_distribution, (self.n_users, self.n_users))
        # ii edge intent
        head_embeddings = torch.index_select(ix_embeddings, 0, self.ii_h_list)
        tail_embeddings = torch.index_select(ix_embeddings, 0, self.ii_t_list)
        edge_distributions = torch.softmax((head_embeddings * tail_embeddings) @ intents, dim=1)  # edge * intent
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        edge_alpha = edge_alpha.flatten()
        item_intents = torch_sparse.spmm(self.IIT_indices, edge_alpha, self.n_items, self.n_ii_trains, edge_distributions)  # item * intent
        item_intents = torch.div(torch.ones_like(item_intents), item_intents)  # 取倒数 item * intent
        edge_item_intents = torch_sparse.spmm(self.TII_indices, edge_alpha, self.n_ii_trains, self.n_items, item_intents)  # item * intent
        edge_item_distribution = edge_distributions * edge_item_intents
        edge_item_distribution = torch.mean(edge_item_distribution, dim=1, keepdim=False)  # edge * 1
        edge_item_distribution = self.normalization_indices_values(self.ii_h_list, self.ii_t_list, edge_item_distribution, (self.n_items, self.n_items))

        x2u, x2i = torch.split(input, [self.n_users, self.n_items], 0)
        uu2_embedings = torch_sparse.spmm(self.uu_indices, edge_user_distribution, self.n_users, self.n_users, x2u)
        ii2_embedings = torch_sparse.spmm(self.ii_indices, edge_item_distribution, self.n_items, self.n_items, x2i)
        sdr_layer_embeddings = torch.cat((uu2_embedings, ii2_embedings), dim=0)

        if self.intent_normalize:
            sdr_layer_embeddings = torch.nn.functional.normalize(sdr_layer_embeddings)
          
        return sdr_layer_embeddings

    def inference(self):

        base_layer_mebddings = torch.concat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [base_layer_mebddings]

        base_embeddings = []
        gnn_embeddings = []
        hio_embeddings = []
        gnn_embeddings2 = []
        hio_embeddings2 = []
        gnn_embeddings4 = []
        hio_embeddings4 = []

        c0 = []
        c1 = []
        c2 = []

        # intents = torch.nn.functional.normalize(self.intents)
        intents = self.intents

        for i in range(0, self.n_layers):

            # Graph-based Message Passing
            gnn_layer_embeddings = self.embed11(all_embeddings[i])
            hio_layer_embeddings = self.embed12(all_embeddings[i])
            con_layer_embeddings = gnn_layer_embeddings + hio_layer_embeddings 

            gnn_layer_embeddings2 = self.embed11(hio_layer_embeddings)
            hio_layer_embeddings2 = self.embed12(gnn_layer_embeddings)
            con_layer_embeddings2 = gnn_layer_embeddings2 + hio_layer_embeddings2

            xxx = self.mlp2(torch.cat([gnn_layer_embeddings, hio_layer_embeddings, gnn_layer_embeddings2, hio_layer_embeddings2], dim=1))   

            gnn_layer_embeddings4 = self.embed21(xxx, intents, gnn_layer_embeddings + hio_layer_embeddings2)
            hio_layer_embeddings4 = self.embed22(xxx, intents, hio_layer_embeddings + gnn_layer_embeddings2)
            con_layer_embeddings4 = gnn_layer_embeddings4 + hio_layer_embeddings4 

            base_layer_mebddings =  all_embeddings[i] + con_layer_embeddings + con_layer_embeddings2 + con_layer_embeddings4   
            
            base_embeddings.append(base_layer_mebddings)
            gnn_embeddings.append(gnn_layer_embeddings)
            hio_embeddings.append(hio_layer_embeddings)
            gnn_embeddings2.append(gnn_layer_embeddings2)
            hio_embeddings2.append(hio_layer_embeddings2)
            gnn_embeddings4.append(gnn_layer_embeddings4)
            hio_embeddings4.append(hio_layer_embeddings4)
            c0.append((base_layer_mebddings))
            c1.append((gnn_layer_embeddings + hio_layer_embeddings2 + gnn_layer_embeddings4))
            c2.append((hio_layer_embeddings + gnn_layer_embeddings2 + hio_layer_embeddings4)) 

            all_embeddings.append(base_layer_mebddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        r1 = [gnn_embeddings, gnn_embeddings2, gnn_embeddings4]
        r2 = [hio_embeddings, hio_embeddings2, hio_embeddings4]
        r3 = [c0,c1,c2]
        return [r1, r2, r3]
   

    def cal_ssl_loss(self, users, items, es):

        users = torch.unique(users)
        items = torch.unique(items)

        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(
                torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(es[0])):

            user_embs = []
            item_embs = []
            for e in es:
                u_emb, i_emb = torch.split(
                    e[i], [self.n_users, self.n_items], 0)
                user_embs.append(u_emb)
                item_embs.append(i_emb)

            user_embs = [F.normalize(x[users], dim=1) for x in user_embs]
            item_embs = [F.normalize(x[items], dim=1) for x in item_embs]

            target_user_emb = user_embs[0]
            for x in user_embs[1:]:
                cl_loss += cal_loss(target_user_emb, x)

            target_item_emb = item_embs[0]
            for x in item_embs[1:]:
                cl_loss += cal_loss(target_item_emb, x)

        return cl_loss

    def forward(self, users, pos_items, neg_items):

        users = torch.LongTensor(users).to(self.device)
        pos_items = torch.LongTensor(pos_items).to(self.device)
        neg_items = torch.LongTensor(neg_items).to(self.device)

        embss = self.inference()

        # bpr
        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        neg_embeddings = self.ia_embedding[neg_items]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)

        # mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        mf_loss = torch.mean(self.softplus(neg_scores - pos_scores))

        # embeddings
        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(
            2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss

        # intent prototypes
        cen_loss = (self.intents.norm(2).pow(2))
        cen_loss = self.cen_reg * cen_loss

        # intent distance loss
        dis_loss = torch.mean(self.intents.T @ self.intents)
        dis_loss = self.cen_dis * dis_loss

        # self-supervise learning
        cl_loss = 0
        for embs in embss:
            cl_loss += self.ssl_reg * self.cal_ssl_loss(users, pos_items, embs)

        return mf_loss.nan_to_num(0, 0, 0), emb_loss.nan_to_num(0, 0, 0), cen_loss.nan_to_num(0, 0, 0), dis_loss.nan_to_num(0, 0, 0), cl_loss.nan_to_num(0, 0, 0)

    def predict(self, users):
        u_embeddings = self.ua_embedding[torch.LongTensor(
            users).to(self.device)]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings
