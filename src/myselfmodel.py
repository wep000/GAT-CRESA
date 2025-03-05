import copy
import math

import numpy as np
import torch
import transformers.modeling_ctrl
import transformers.modeling_transfo_xl
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from CrossModalTransformer2 import MultiHeadAttention, PoswiseFeedForwardNet, \
    EncoderLayer, Encoder, CrossModalTransformer2, ScaledDotProductAttention, SelfMultiHeadAttention
from src.CS_GRU import CS_GRU_v1, GRUCell, CS_GRU_v2, GRUCell_v2, GRUCell_v3
from src.Es_CMT import CrossModalTransformer_ctx
from src.GAT_dialoggcn import GAT_dialoggcn_v1, attentive_node_features
from src.GAT_dialoggcn import GAT_dialoggcn_v2
from src.ReasonModule import ReasonModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SSE(nn.Module):
    def __init__(self, D_hidden):
        super().__init__()
        self.D_hidden = D_hidden
        self.init_trans = nn.Linear(self.D_hidden, self.D_hidden)
        self.drop = nn.Dropout(0.4)
        self.w_qinter = nn.Linear(self.D_hidden, self.D_hidden)
        self.W = nn.Linear(self.D_hidden,1)
        self.w_out = nn.Linear(self.D_hidden, self.D_hidden)
        self.relu = nn.ReLU()
    def forward(self, feature, mask_intra, umask):
        '''对inter-speaker做说话人间状态计算'''
        feature = self.drop(self.relu(self.init_trans(feature)))
        v_inter1 = torch.zeros_like(feature)
        for i in range(feature.size(0)):
            c = feature.clone()
            for j in range (feature.size(1)):
                if j == 0:
                    v_inter1[i,j,:] = c[i,j,:]
                    for k in range(j+1,feature.size(1)):
                        if mask_intra[i, j, k] != mask_intra[i, j, j]:
                            v_inter1[i, k, :] = c[i, k, :]
                            break
                    continue
                if umask[i, j] == 0: break
                mark = 0
                for l in range(j-1,-1,-1):#从当前utterance出发找到上一个同说话者
                    if mask_intra[i, j, l] == 1:#找到上一个同说话者
                        mark = 1
                        ki = v_inter1[i, l:j, :].clone() #说话者间的局部信息
                        q_inter = self.w_qinter(c[i,j,:].unsqueeze(0)).permute(1,0)#q_inter:h×1
                        alpha_inter = F.softmax(self.W(torch.mul(q_inter, ki.permute(1,0)).permute(1,0)), dim=0).permute(1,0)#alpha_inter:1×l
                        v_inter1[i, j, :] = torch.tanh(torch.matmul(alpha_inter, ki).squeeze(0))
                        break
                if mark == 0: v_inter1[i, j, :] = c[i, j, :]
        new_feature = self.relu(torch.mul(self.w_out(feature),v_inter1)) + feature
        return new_feature

class LIE(nn.Module):
    def __init__(self, D_hidden, windowp, windowf):
        super().__init__()
        self.D_hidden = D_hidden
        self.windowp = windowp
        self.windowf = windowf
        self.wq = nn.Linear(self.D_hidden, self.D_hidden)
        self.w2 = nn.Linear(self.D_hidden, 1)
        self.wm = nn.Linear(self.D_hidden, self.D_hidden)
        self.norm = nn.LayerNorm(self.D_hidden)
        self.W1 = nn.Linear(self.D_hidden, self.D_hidden)
        self.W2 = nn.Linear(self.D_hidden, self.D_hidden)
        self.drop = nn.Dropout(0.4)
    def forward(self, feature):
        q_local = self.wq(feature)
        v_local = torch.zeros_like(feature)
        for i in range(feature.size(1)):
            if i<self.windowp:
                v_local[:, i, :] = feature[:, i, :]
                continue
            # if i>feature.size(1)-self.windowf:
            #     ki = v_local[:, i - self.windowp:i+1, :].clone()  # 局部utterance信息
            #     alpha_local = F.softmax(self.w2(torch.mul(q_local[:, i, :].unsqueeze(1), ki)),
            #                             dim=1)  # (32,l,1)#得到局部utterance的权重
            #     v_local[:, i, :] = torch.matmul(ki.transpose(1, 2), alpha_local).squeeze(2)  # 乘上权重得到更新后的局部信息
            #     continue
            ki = v_local[:, i-self.windowp:i+1, :].clone()#局部utterance信息
            alpha_local = F.softmax(self.w2(torch.mul(q_local[:, i ,:].unsqueeze(1), ki)), dim=1)#(32,l,1)#得到局部utterance的权重
            v_local[:, i, :] = torch.matmul(ki.transpose(1,2), alpha_local).squeeze(2)#乘上权重得到更新后的局部信息
        out_feature = self.norm(torch.mul(torch.tanh(v_local), feature) + feature)
        # out_feature = self.norm(temp_feature + self.drop(self.W2(F.relu(self.W1(temp_feature)))))
        # out_feature = torch.mul(torch.tanh(v_local), feature) + feature
        return out_feature

class selfmodel(nn.Module):
    def __init__(self,n_speakers,n_classes=6,dropout=0.5,no_cuda=False,D_audio=384,D_text=768,
                 D_hidden=200,dataset='IEMOCAP',use_crn_speaker=True,speaker_weights='1-1-1',D_e=100,att_head=8,steps=3, args = None):
        super(selfmodel, self).__init__()
        self.args = args
        self.steps = steps
        self.n_speakers = n_speakers
        self.n_classes = n_classes
        self.dropout = dropout
        self.no_cuda = no_cuda
        self.D_text = D_text
        self.att_head = att_head
        self.D_audio = D_audio
        self.D_e = D_e
        self.D_hidden = D_hidden
        self.use_crn_speaker = use_crn_speaker
        self.speaker_weights = list(map(float, speaker_weights.split('-')))
        self.dataset = dataset
        self.conv1d_t = nn.Conv1d(D_text,D_hidden,1)
        self.conv1d_a = nn.Conv1d(D_audio,D_hidden,1)
        self.spk_embed = nn.Embedding(n_speakers, D_hidden)

        self.rnn_parties_t = nn.GRU(input_size=self.D_hidden, hidden_size=self.D_e, num_layers=2, bidirectional=True,
                                    dropout=self.dropout)
        self.rnn_parties_a = nn.GRU(input_size=self.D_hidden, hidden_size=self.D_e, num_layers=2, bidirectional=True,
                                    dropout=self.dropout)

        self.intra_mask_CMT_ctx = CrossModalTransformer_ctx(self.D_hidden,n_heads=4,n_layers=5)
        self.inter_mask_CMT_ctx = CrossModalTransformer_ctx(self.D_hidden, n_heads=4, n_layers=5)


        GATs_spk = []
        for _ in range(args.GAT_nlayers):
            # gats += [GAT_dialoggcn(args.hidden_dim)]
            GATs_spk += [GAT_dialoggcn_v2(self.D_hidden)]
        self.gats_spk = nn.ModuleList(GATs_spk)

        GATs_ctx = []
        for _ in range(args.GAT_nlayers):
            # gats += [GAT_dialoggcn(args.hidden_dim)]
            GATs_ctx += [GAT_dialoggcn_v2(self.D_hidden)]
        self.gats_ctx = nn.ModuleList(GATs_ctx)

        grus_spk = []
        for _ in range(args.GAT_nlayers):
            grus_spk += [nn.GRUCell(self.D_hidden, self.D_hidden)]
        self.grus_spk = nn.ModuleList(grus_spk)

        grus_ctx = []
        for _ in range(args.GAT_nlayers):
            grus_ctx += [nn.GRUCell(self.D_hidden, self.D_hidden)]
        self.grus_ctx = nn.ModuleList(grus_ctx)

        self.mlp_layer = 2
        layers = [nn.Linear(self.D_hidden*6, self.D_hidden), nn.ReLU()]
        for _ in range(self.mlp_layer - 1):
            layers += [nn.Linear(self.D_hidden, self.D_hidden), nn.ReLU()]
        layers += [nn.Dropout(self.dropout)]
        layers += [nn.Linear(self.D_hidden, n_classes)]
        self.out_mlp = nn.Sequential(*layers)

        self.layer_normfu = nn.LayerNorm(self.D_hidden)
        self.layer_normCtx = nn.LayerNorm(self.D_hidden)

        self.drp = nn.Dropout(self.dropout)

        self.wfu1 = nn.Linear(self.D_hidden, self.D_hidden)
        self.wfu2 = nn.Linear(self.D_hidden, self.D_hidden)
        self.graph_ctxLin1 = nn.Linear(self.D_hidden, self.D_hidden)
        self.graph_ctxLin2 = nn.Linear(self.D_hidden, self.D_hidden)

        self.classify_fu = nn.Linear(self.D_hidden, self.n_classes)

        self.out_un = nn.Linear(self.D_hidden*2, self.D_hidden)
        self.classify_un = nn.Linear(self.D_hidden, self.n_classes)

        '''新部分'''
        self.reason_modules = ReasonModule(in_channels=self.D_hidden*2, processing_steps=self.steps, num_layers=1)
        self.fc = nn.Linear(self.D_hidden*2, self.D_hidden*4)
        self.ctx_lin = nn.Linear(self.D_hidden*4, self.D_hidden)
        self.init_lin = nn.Linear(self.D_hidden*2, self.D_hidden)
        self.spk_shift_lin = nn.Linear(self.D_hidden*2, 2)
        self.ctxLin = nn.Linear(self.D_hidden, 2)
        self.graph_input_lin = nn.Linear(self.D_hidden*2, self.D_hidden)
        self.kl_ctx_lin = nn.Linear(self.D_hidden*4, self.n_classes)
        self.kl_spk_lin = nn.Linear(self.D_hidden, self.n_classes)


        '''图的边特征生成部分'''
        self.graph_edge_spk_lin = nn.Linear(self.D_hidden*2, 2)
        self.graph_edge_ctx_lin = nn.Linear(self.D_hidden, 2)
        self.intra_step_lin = nn.Linear(self.D_hidden, self.D_hidden)
        self.inter_step_lin = nn.Linear(self.D_hidden, self.D_hidden)


    def forward(self,text,qmask,s_mask,umask,length,audio,mask_intra, mask_inter,mask_local,adj,emo_shift_matrix):#text(78,16,768),umask(78,16),adj(32,78,78)
        text = text.permute(1,2,0)
        audio = audio.permute(1,2,0)
        text = self.conv1d_t(text).permute(2,0,1)#(78,32,200)
        audio = self.conv1d_a(audio).permute(2,0,1)#(78,32,200)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_vec = self.spk_embed(spk_idx).permute(1,0,2)#speaker embedding

        unprocessed = torch.cat([text.permute(1,0,2), audio.permute(1,0,2)],dim=2)#(66,16,768*2)

        if self.use_crn_speaker:

            # (32,21,200) (32,21,9)
            U_, qmask_ = text.transpose(0, 1), qmask.transpose(0, 1)  # U_(32,77,200),qmask(32,77,2)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(text.type())  # U_p_(32,77,200)
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in
                          range(self.n_speakers)]  # default 2,此时U_parties_是长度为2的list,每个元素都是U_(32,77,200)形状全为0的tensor
            for b in range(U_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(
                        -1)  # index_i即为每个speaker在每个batch中说话的utterance下标，qmask_(32,77,2),torch.nonzero(qmask_[b][:, p])表示取出某一个batch中某一个speaker的第二个维度中非0的元素下标
                    if index_i.size(0) > 0:  # index_i:(29,)即为每个speaker在每个batch中说话的utterance下标
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]  # 有效数范围是(2,32,29)29为index_i中非0的下标元素
                        '''这里把每个batch中每个讲话者说话的utterance下标取出，然后取出对应的utterance特征，放在U_parties_中，
                           U_parties_是长为2的list，每个元素是2个speaker对应的utterance特征'''

            E_parties_ = [self.rnn_parties_t(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in
                          range(len(U_parties_))]
            '''self.rnn_parties(U_parties_[p].transpose(0, 1))输出即GRU输出，(77,32,200)和(4,32,100),
            E_parties_为长度2的list,分别为每个speaker的embedding特征'''

            for b in range(U_p_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)  # 这里是找每个speaker说话的utterance下标
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            # (21,32,200)
            '''到这里为止相当于利用qmask的索引对输入的特征进行了一次重构，相比于直接加上speaker embedding，这样得到的特征更加精确'''
            U_p = U_p_.transpose(0, 1)
            emotion_t = text + self.speaker_weights[0] * U_p  # emotion_a(78,32,200)
            '''U_p即计算得到的speaker embedding，这里乘上speaker权重后和经过全连接层的语音特征相加'''

            # (32,21,200) (32,21,9)
            U_, qmask_ = audio.transpose(0, 1), qmask.transpose(0, 1)  # U_(32,77,200),qmask(32,77,2)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(audio.type())  # U_p_(32,77,200)
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in
                          range(self.n_speakers)]  # default 2,此时U_parties_是长度为2的list,每个元素都是U_(32,77,200)形状全为0的tensor
            for b in range(U_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(
                        -1)  # index_i即为每个speaker在每个batch中说话的utterance下标，qmask_(32,77,2),torch.nonzero(qmask_[b][:, p])表示取出某一个batch中某一个speaker的第二个维度中非0的元素下标
                    if index_i.size(0) > 0:  # index_i:(29,)即为每个speaker在每个batch中说话的utterance下标
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]  # 有效数范围是(2,32,29)29为index_i中非0的下标元素
                        '''这里把每个batch中每个讲话者说话的utterance下标取出，然后取出对应的utterance特征，放在U_parties_中，
                           U_parties_是长为2的list，每个元素是2个speaker对应的utterance特征'''

            E_parties_ = [self.rnn_parties_a(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in
                          range(len(U_parties_))]
            '''self.rnn_parties(U_parties_[p].transpose(0, 1))输出即GRU输出，(77,32,200)和(4,32,100),
            E_parties_为长度2的list,分别为每个speaker的embedding特征'''

            for b in range(U_p_.size(0)):  # 对每一个batch
                for p in range(len(U_parties_)):  # 对每一个speaker
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)  # 这里是找每个speaker说话的utterance下标
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            # (21,32,200)
            '''到这里为止相当于利用qmask的索引对输入的特征进行了一次重构，相比于直接加上speaker embedding，这样得到的特征更加精确'''
            U_p = U_p_.transpose(0, 1)
            emotion_a = audio + self.speaker_weights[0] * U_p    #emotion_a(78,32,200)
            '''U_p即计算得到的speaker embedding，这里乘上speaker权重后和经过全连接层的语音特征相加'''

        batch_index, temp_seq_feature = [], []
        batch_size = emotion_a.size(1)
        # TODO 加线性层和激活函数
        temp_fuse = torch.cat([emotion_t, emotion_a], dim=2) #暂时用cat来替代模态特征融合
        for j in range(batch_size):
            batch_index.extend([j] * length[j])
            temp_seq_feature.append(temp_fuse[:length[j], j, :]) #得到所有batch的utterance序列拼在一个维度的特征

        batch_index = torch.tensor(batch_index)
        all_seq_feature = torch.cat(temp_seq_feature, dim=0)
        batch_index = batch_index.cuda()
        all_seq_feature = all_seq_feature.cuda()

        feature_ = []
        for t in range(temp_fuse.size(0)):     #对每一个句子，也就是每一个时间步
            q_star = self.fc(temp_fuse[t])     #fc:2du-->4du,
            q_situ = self.reason_modules(all_seq_feature, batch_index, q_star)     #q_situ(32,400),对应paper中的q(t-1),bank_s_对应global memory G,batch_index对应h(t-1)隐藏层状态信息
            feature_.append(q_situ.unsqueeze(0))   #q_situ是(32,400)升维后为(1,32,400),feature_为长度为94的list，每个元素为(1,32,400)
        temp_feature_ctx = torch.cat(feature_, dim=0)     #feature_ctx即为(94,32,400)
        feature_ctx = self.drp(F.relu(temp_feature_ctx))

        '''上下文情绪转移特征做kl损失'''
        ctx_feature = []
        for i in range(batch_size):
            ctx_feature.append(feature_ctx[:length[i], i, :])
        ctx_pred = self.kl_ctx_lin(torch.cat(ctx_feature, dim=0))
        log_pred_ctx = F.log_softmax(ctx_pred, dim=1)

        emotion_a = emotion_a.permute(1, 0, 2)
        emotion_t = emotion_t.permute(1, 0, 2)  # (32,78,200)
        ctx_info = self.drp(F.relu(self.ctx_lin(temp_feature_ctx)))
        intra_emo_shift, _ = self.intra_mask_CMT_ctx(emotion_t, emotion_a, ctx_info, mask_intra)
        inter_emo_shift, _ = self.inter_mask_CMT_ctx(emotion_t, emotion_a, ctx_info, mask_inter)

        '''上下文情绪转移损失计算'''
        ctx_emo_pred_matrix = []
        init_feature = F.relu(self.init_lin(temp_fuse))
        for i in range(batch_size):
            ctx_shift_feature = ctx_info[:length[i], i, :]
            init_shift_feature = init_feature[:length[i], i, :]
            temp_matrix = torch.zeros((length[i], length[i], ctx_shift_feature.size(1)), dtype=torch.float32).cuda()
            for j in range(length[i]):
                temp_matrix[j] = ctx_shift_feature[j:j + 1] - init_shift_feature
            temp_matrix = F.log_softmax(self.ctxLin(temp_matrix), dim=2)
            ctx_emo_pred_matrix.append(temp_matrix)

        '''说话人的情绪转移损失计算，这里生成了一个长度为batch size的情绪转移预测矩阵'''
        spk_emo_shift_list = []
        for i in range(batch_size):
            batch_intra_emo_shift = intra_emo_shift[i, :length[i], :]
            batch_inter_emo_shift = inter_emo_shift[i, :length[i], :]
            batch_emo_shift_matrix = torch.zeros((length[i], length[i], intra_emo_shift.size(2)*2),dtype=torch.float32).cuda()
            for j in range(batch_inter_emo_shift.size(0)):
                utter_emo_shift = batch_intra_emo_shift[j,:].unsqueeze(0).expand(length[i], -1)
                temp_concat = torch.cat([utter_emo_shift, batch_inter_emo_shift], dim=1)
                batch_emo_shift_matrix[j, :, :] = temp_concat
            batch_emo_shift_matrix = F.log_softmax(self.spk_shift_lin(batch_emo_shift_matrix),dim=2)
            spk_emo_shift_list.append(batch_emo_shift_matrix)

        '''用上下文推理模块预测的标签来做上下文图的情绪转移边的embedding特征'''
        dia_max_len = intra_emo_shift.size(1)
        cat_ctx_feature = torch.zeros((batch_size, dia_max_len, dia_max_len, intra_emo_shift.size(2)), dtype=torch.float32).cuda()
        for i in range(dia_max_len):
            utter_ctx_shift = ctx_info.permute(1,0,2)[:, i:i+1, :].expand(-1, dia_max_len, -1)
            init_ctx_feature = init_feature.permute(1,0,2)
            temp_ctx_shift = utter_ctx_shift - init_ctx_feature
            cat_ctx_feature[:, i, :, :] = temp_ctx_shift
        pred_ctx_shift_matrix = torch.argmax(F.log_softmax(self.graph_edge_ctx_lin(cat_ctx_feature), dim=3),3)

        '''用说话人transformer输出的预测标签来做图的情绪转移边的embedding特征'''
        intra_graph_edge = F.relu(self.intra_step_lin(intra_emo_shift))
        inter_graph_edge = F.relu(self.inter_step_lin(inter_emo_shift))
        # cat_spk_feature(16,66,66,400)
        cat_spk_feature = torch.zeros((batch_size, dia_max_len, dia_max_len, intra_emo_shift.size(2)*2),dtype=torch.float32).cuda()
        for i in range(dia_max_len):
            utter_emo_feature = intra_graph_edge[:, i, :].unsqueeze(1).expand(-1, dia_max_len, -1)
            temp_ = torch.cat([utter_emo_feature, inter_graph_edge], dim=2)
            cat_spk_feature[:, i, :, :] = temp_
        pred_spk_shift_matrix = torch.argmax(F.log_softmax(self.graph_edge_spk_lin(cat_spk_feature), dim=3),3)
        # board = int(dia_max_len/5)

        graph_input = self.drp(F.relu(self.graph_input_lin(torch.cat([intra_emo_shift, inter_emo_shift], dim=2)))) #graph_input(16,66,200)

        '''说话人情绪转移特征做kl损失'''
        spk_feature = []
        for i in range(batch_size):
            spk_feature.append(graph_input[i, :length[i], :])
        spk_pred = self.kl_spk_lin(torch.cat(spk_feature, dim=0))
        log_pred_spk = F.log_softmax(spk_pred, dim=1)

        num_utter = graph_input.size(1)
        adj_spk = adj
        adj_ctx = adj
        H_0_spk = graph_input
        H_0_ctx = ctx_info.permute(1,0,2)
        H_spk = [H_0_spk]
        H_ctx = [H_0_ctx]
        for l in range(self.args.GAT_nlayers - 1):
            H1_spk = self.grus_spk[l](H_spk[l][:, 0, :]).unsqueeze(1)
            H1_ctx = self.grus_ctx[l](H_ctx[l][:, 0, :]).unsqueeze(1)
            '''如果要做过去p个和未来f个，可考虑将H1_a为列表，每个存储不同层上下文信息，当前使用上一层信息，'''
            for i in range(1, num_utter):
                att_weight_spk, att_sum_spk, updated_adj_spk = self.gats_spk[l](H_spk[l][:, i, :], H1_spk, H1_spk, adj_spk[:, i, :i], pred_spk_shift_matrix[:, i, :i])
                att_weight_ctx, att_sum_ctx, updated_adj_ctx = self.gats_ctx[l](H_ctx[l][:, i, :], H1_ctx, H1_ctx, adj_ctx[:, i, :i], pred_ctx_shift_matrix[:, i, :i])
                adj_spk[:, i, :i] = updated_adj_spk.squeeze(1)
                adj_ctx[:, i, :i] = updated_adj_ctx.squeeze(1)
                H1_spk = torch.cat([H1_spk, att_sum_spk.unsqueeze(1)], dim=1)
                H1_ctx = torch.cat([H1_ctx, att_sum_ctx.unsqueeze(1)], dim=1)

            H1_spk = self.layer_normfu(H_spk[l] + H1_spk)
            H1_spk = self.layer_normfu(H1_spk + self.drp(self.wfu2(F.relu(self.wfu1(H1_spk)))))
            H_spk.append(H1_spk)
            H1_ctx = self.layer_normCtx(H_ctx[l] + H1_ctx)
            H1_ctx = self.layer_normCtx(H1_ctx + self.drp(self.graph_ctxLin2(F.relu(self.graph_ctxLin1(H1_ctx)))))
            H_ctx.append(H1_ctx)


        '''想法:分别用文本和语音预测输出来做KL散度损失，最后拼接做预测'''
        feature_spk = torch.cat(H_spk, dim=2)#(32,78,600)
        feature_ctx = torch.cat(H_ctx, dim=2)

        feature = torch.cat([feature_spk, feature_ctx], dim=2)

        #用text和audio原始特征的直接拼接来预测情绪做t-sne可视化
        raw_cat_feature = self.out_un(torch.cat([text, audio],dim=2))
        pred_raw = []
        for l in range(raw_cat_feature.size(1)):
            temp = raw_cat_feature[:length[l], l, :]
            pred_raw.append(temp)
        temp_raw = []
        for i in range(len(pred_raw)):
            if i == 0: temp_raw = pred_raw[i]
            if i == len(pred_raw) - 1: break
            temp_raw = torch.cat([temp_raw, pred_raw[i + 1]], dim=0)
        prob_raw = self.classify_fu(temp_raw)
        log_prob_raw = F.log_softmax(prob_raw,dim=1)

        pred = []
        for l in range(feature.size(0)):
            temp = feature[l, :length[l], :]
            pred.append(temp)
        temp2 = []
        for i in range(len(pred)):
            if i==0: temp2 = pred[i]
            if i==len(pred) - 1: break
            temp2 = torch.cat([temp2, pred[i+1]], dim=0)
        log_prob = self.out_mlp(temp2)
        unlog_prob = F.softmax(log_prob, dim=1)
        log_prob = F.log_softmax(log_prob, dim=1)

        pred_un = []
        for l in range(unprocessed.size(0)):
            temp = unprocessed[l, :length[l], :]
            pred_un.append(temp)
        temp_un= []
        for i in range(len(pred_un)):
            if i == 0: temp_un = pred_un[i]
            if i == len(pred_un) - 1: break
            temp_un = torch.cat([temp_un, pred_un[i + 1]], dim=0)
        prob_un = self.classify_un(F.relu(self.out_un(temp_un)))
        log_prob_un = F.log_softmax(prob_un, dim=1)
        unlog_prob_un = F.softmax(prob_un, dim=1)

        return log_prob, unlog_prob, log_prob_un, log_prob_raw,ctx_emo_pred_matrix, spk_emo_shift_list, log_pred_ctx, log_pred_spk

        '''# 因为要用y指导x,所以求x的对数概率，y的概率
            logp_x = F.log_softmax(x, dim=-1)
            p_y = F.softmax(y, dim=-1)'''



