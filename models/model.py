


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn import init
from utils1 import *
from torch.nn.modules.loss import BCEWithLogitsLoss
import sys
sys.path.append('/home/ynos/Desktop/RTGCN')


BN = True



# import torch
# from torch import nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd  import Variable

import numpy as np
import scipy.sparse as sp



class RTGCN(nn.Module):
    """
    act: activation function for GAT
    n_node: number of nodes on the network
    output_dim: output embed size for GAT
    seq_len: number of graphs
    n_heads: number of heads for GAT
    attn_drop: attention/coefficient matrix dropout rate
    ffd_drop: feature matrix dropout rate
    residual: if using short cut or not for GRU network
    """

    def __init__(self,
                 act,
                 n_node,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 time_step,
                 neg_weight,
                 loss_weight,
                 # seq_len,#这个要改
                 n_heads,
                 attn_drop,
                 ffd_drop,
                 role_num,
                 cross_role_num,
                 residual=False,
                 bias=True,
                 sparse_inputs=False
                 ):
        super(RTGCN, self).__init__()

        self.act = nn.ELU()
        # self.n_node = n_node#
        self.output_dim = output_dim
        # self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim=hidden_dim
        self.num_time_steps = time_step
        self.neg_weight=neg_weight
        self.n_heads = n_heads
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.residual = residual
        self.role_num=role_num#the number of role
        self.cross_role_num=cross_role_num#the number of cross_role
        self.bceloss=BCEWithLogitsLoss()
        self.var = {}
        self.hypergnn=nn.ModuleList()
        self.hypergnn.append(HyperGNN(input_dim,hidden_dim,num_layer=1))
        self.hypergnn.append(HyperGNN(hidden_dim, output_dim, num_layer=1))

        # self.gcn=nn.ModuleList()
        self.gcn=GCN(input_dim,hidden_dim,output_dim,self.attn_drop,concat=False)

        # self.evolve_weights = GRU(n_node, input_dim, output_dim, n_heads, residual)
        self.evolve_weights=nn.ModuleList()
        self.evolve_weights.append(MatGRUCell( n_node,input_dim, hidden_dim,n_heads,cross_role_num))
        self.evolve_weights.append(MatGRUCell(n_node, hidden_dim, output_dim, n_heads,cross_role_num))
        
        self.weight_var1 = nn.Parameter(torch.randn(self.input_dim,  self.hidden_dim))
        self.weight_var2 = nn.Parameter(torch.randn(self.hidden_dim,  self.output_dim))
        
        self.loss_weight=loss_weight
        self.emb_weight=nn.Parameter(torch.ones(1))
        self.emb_cross_weight = nn.Parameter(torch.ones(1))


    def forward(self, data, train_hypergraph,cross_role_hyper,cross_role_laplacian):

        # adj_matrix = data['adjs']  # nadarry [10,4257,4257]
        # attribute_matrix = data['attmats']  # nadarry [4257,10,100]
        # label_matrix = data['labels'] #ndarray [4257,3]
        input_att=data[0][0]
        input_edge=data[1][0]
        # input_att=torch.from_numpy(attribute_matrix[:,0,:]).float()
        # input_edge=adj_to_edge(adj_matrix[0,:,:])[0]
        embeds = []
        input_hypergraph=train_hypergraph[0]



        # weight_vars = {}
        # for i in range(self.n_heads):
        #     weight_var = nn.Parameter(torch.randn(self.input_dim,2* self.output_dim))#设置成参数形式，[input,output]
        #
        #     weight_vars[i] = weight_var

            # self.var['weight_var_' + str(i)] = weight_var
        weight_var1 = self.weight_var1
        weight_var2 = self.weight_var2

        #gcn
        output1 =self.gcn(input_att,input_edge,weight_var1[:,:self.hidden_dim],weight_var2[:,:self.output_dim])


        #hypergraph
        output2=self.hypergnn[0](input_att, input_hypergraph,weight_var1[:,:self.hidden_dim])
        output2 = self.hypergnn[1](output2, input_hypergraph,weight_var2[:, :self.output_dim])
        output2=F.log_softmax(output2, dim=1)
        # hyper_out=output2

        output = output1+self.emb_weight*output2
        # output=torch.cat((gnn_output,hyper_out),dim=1)
        embeds.append(output)

        for i in range(1, len(train_hypergraph)):#[0-T]
            input_att1=data[0][i]
            input_edge1=data[1][i]
            # label_matrix1=label_matrix
            input_hypergraph1=train_hypergraph[i]#
            input_cross_hypergraph=cross_role_hyper[i-1]
            input_cross_laplacian=cross_role_laplacian[i-1]
            adj_matrix1=data[2][i]


            weight_var1 = self.evolve_weights[0](adj_matrix1,weight_var1,input_cross_hypergraph)
            weight_var2 = self.evolve_weights[1](adj_matrix1,weight_var2, input_cross_hypergraph)
            #gcn
            gnn_output= self.gcn(input_att1, input_edge1,weight_var1[:,:self.hidden_dim],weight_var2[:,:self.output_dim])  
            # gnn_output = output1
            #hypergraph
            output2_0 = self.hypergnn[0](input_att1, input_hypergraph1, weight_var1[:,:self.hidden_dim])###input_att
            hyper_out = self.hypergnn[1](output2_0, input_hypergraph1, weight_var2[:, :self.output_dim])
            hyper_out = F.log_softmax(hyper_out, dim=1)
            # hyper_out = output2
            
            #cross_role graph
            output3 = self.hypergnn[0](data[0][i-1], input_cross_laplacian, weight_var1[:,:self.hidden_dim])
            output3 = self.hypergnn[1](output3, input_cross_laplacian, weight_var2[:, :self.output_dim])
            output3 = F.log_softmax(output3, dim=1)

            output = gnn_output+self.emb_weight*hyper_out+self.emb_cross_weight*output3
            # output = gnn_output + self.emb_weight * hyper_out
            embeds.append(output)

        return embeds

    def get_loss(self, feed_dict ,data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian,list_loss_role):
        node_1, node_2, node_2_negative = feed_dict.values()
        # run gnn
        final_emb = self.forward(data_dblp,train_hypergraph,cross_role_hyper,cross_role_laplacian) # [N, T, F]
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            # emb_t = final_emb[:, t, :].squeeze() #[N, F]
            emb_t = final_emb[t]  # [N, F]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :]*tart_node_neg_emb, dim=2).flatten()#[180,1,128]*[[180,10,128]
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.neg_weight*neg_loss
            self.graph_loss += graphloss
            role_loss=0
            calculate_loss=list_loss_role[t]
            for l in calculate_loss:
                node_role_emb=emb_t[l]
                a = node_role_emb/torch.norm(node_role_emb,dim=1,keepdim=True)
                similarity = torch.mm(a,a.T)
                I_mat=torch.ones_like(similarity)
                role_loss+=torch.norm(similarity-I_mat)**2/2
                del similarity,node_role_emb

            self.graph_loss+=self.loss_weight*role_loss
            
        return self.graph_loss


    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl


class GCNConv(MessagePassing):
    def __init__(self,in_channels,out_channels,dropout,concat):
        super(GCNConv,self).__init__(aggr='add')
        # self.lin=torch.nn.Linear(in_channels,out_channels)
    def forward(self,x,edge_index,W2):
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(0))
        x=torch.matmul(x,W2)
        row,col=edge_index
        #计算度矩阵
        deg=degree(col,x.size(0),dtype=x.dtype)
        #度矩阵的-1/2次方
        deg_inv_sqrt=deg.pow(-0.5)
        norm=deg_inv_sqrt[row]*deg_inv_sqrt[col]
        return self.propagate(edge_index,x=x,norm=norm)
    def message(self,x_j,norm):
        return norm.view(-1,1)*x_j


        



class GCN(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout,concat):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim,dropout,concat)
        self.conv2 = GCNConv(hidden_dim, output_dim,dropout,concat)
        self.prej =nn.Linear(input_dim,output_dim,bias=False)
        self.alpha1 = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index,gnn_weight1,gnn_weight2):
        x0 = self.prej(x)
        x = self.conv1(x, edge_index,gnn_weight1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index,gnn_weight2)
        X = x*self.alpha1+x0*(1-self.alpha1)
        return F.log_softmax(X, dim=1)



#HyperGNN
class HyperGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_edge_num=3, num_layer=1, negative_slope=0.2):
        super(HyperGNN, self).__init__()
        self.negative_slope = negative_slope

        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))

        # glorot(self.alpha)

    def forward(self, node_initial_emb, hyp_graph,W):
        # laplacian = scipy_sparse_mat_to_torch_sparse_tensor(hyp_graph.laplacian())
        rs = hyp_graph @ torch.matmul( node_initial_emb,W)
        rs = (1-self.alpha)*self.proj(node_initial_emb)+rs*self.alpha
        return rs #[n,input_dim]





class MatGRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, n_node,input_dim, output_dim,n_head,cross_role_num):
        super().__init__()
        self.n_head = n_head
        self.update = MatGRUGate(n_node,input_dim,
                                 output_dim,
                                 torch.nn.Sigmoid(),cross_role_num=cross_role_num)

        self.reset = MatGRUGate(n_node,input_dim,
                                output_dim,
                                torch.nn.Sigmoid(),cross_role_num=cross_role_num)

        self.htilda = MatGRUGate(n_node,input_dim,
                                 output_dim,
                                 torch.nn.Tanh(),cross_role_num=cross_role_num)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, adj,weight_vars,H):



        update = self.update(adj, weight_vars,H)
        reset = self.reset(adj, weight_vars,H)

        h_cap = reset * weight_vars
        h_cap = self.htilda(adj, h_cap,H)

        new_Q = (1 - update) * weight_vars + update * h_cap
        # weight_vars_next[i] = new_Q

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, n_node,rows, cols, activation,cross_role_num):
        super().__init__()
        self.activation = activation
        self.W = nn.Parameter(torch.Tensor(n_node, cols))
        self.W1=nn.Parameter(torch.Tensor(cross_role_num,cols))
        self.U = nn.Parameter(torch.Tensor(cols, cols))
        self.bias =nn.Parameter(torch.Tensor(rows, cols))
        if n_node != rows:
            self.P = nn.Parameter(torch.Tensor(rows, n_node))
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, adj, hidden,incident_matrix):
        temp = adj.matmul(self.W)
        temp1=incident_matrix.matmul(self.W1)
        out = self.activation(self.P.matmul(temp) +self.P.matmul(temp1)+ hidden.matmul(self.U) + self.bias)

        return out



