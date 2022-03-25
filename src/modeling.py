from global_configs import TEXT_DIM
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# Code adapted from the fairseq repo.
import math
import copy
class SPAMN(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(SPAMN, self).__init__()
        print(
            "Initializing SPAMN with beta_shift:{} hidden_prob:{}".format(
                beta_shift, dropout_prob
            )
        )
        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        #self.attn = MultiheadAttention(embed_dim=6*TEXT_DIM, num_heads=8, attn_dropout=0.2)
        self.fc = nn.Linear(6*TEXT_DIM, TEXT_DIM)
        self.memory = MemoryLayer(seq=50, embedding=TEXT_DIM)

        self.query_memory_share = nn.Parameter(torch.Tensor(MEMORY_SIZE, TEXT_DIM))
        self.query_memory_private_l = nn.Parameter(torch.Tensor(MEMORY_SIZE, TEXT_DIM))
        self.query_memory_private_a = nn.Parameter(torch.Tensor(MEMORY_SIZE, TEXT_DIM))
        self.query_memory_private_v = nn.Parameter(torch.Tensor(MEMORY_SIZE, TEXT_DIM))
        #self.W_k = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.q_l_p = MemeoryQueryLayer(seq=50, embedding=TEXT_DIM)
        self.q_a_p = MemeoryQueryLayer(seq=50, embedding=TEXT_DIM)
        self.q_v_p = MemeoryQueryLayer(seq=50, embedding=TEXT_DIM)
        self.q_s   = MemeoryQueryLayer(seq=50, embedding=TEXT_DIM)
        # self.pt_pooler = MemoryPooler()
        # self.pv_pooler = MemoryPooler()
        # self.pa_pooler = MemoryPooler()
        # self.st_pooler = MemoryPooler()
        # self.sv_pooler = MemoryPooler()
        # self.sa_pooler = MemoryPooler()
        # self.pt = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.2)
        # self.pv = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.2)
        # self.pa = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.2)
        # self.st = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.2)
        # self.sv = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.2)
        # self.sa = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.2)

        #self.gate_share = GateFusion()
        #self.gate_private = GateFusion()
        # self.pff = PositionwiseFeedForward(TEXT_DIM, TEXT_DIM)
        # self.rnnpl = nn.GRU(TEXT_DIM, TEXT_DIM, bidirectional=True)
        # self.rnnpa = nn.GRU(TEXT_DIM, TEXT_DIM, bidirectional=True)
        # self.rnnpv = nn.GRU(TEXT_DIM, TEXT_DIM, bidirectional=True)

        # self.rnnsl = nn.GRU(TEXT_DIM, TEXT_DIM, bidirectional=True)
        # self.rnnsa = nn.GRU(TEXT_DIM, TEXT_DIM, bidirectional=True)
        # self.rnnsv = nn.GRU(TEXT_DIM, TEXT_DIM, bidirectional=True)


        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.query_memory_share, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.query_memory_private_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.query_memory_private_l, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.query_memory_private_v, a=math.sqrt(5))
    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    def forward(self, text_embedding, visual, acoustic):
        #torch.Size([48, 50, 768]) torch.Size([48, 50, 47]) torch.Size([48, 50, 74])
        xl = text_embedding
        xa = self.W_a(acoustic)
        xv = self.W_v(visual)
        #xl, xa, xv = self.q_l(xl, self.query_memory), self.q_a(xa, self.query_memory), self.q_v(xv, self.query_memory)
        # for layer in self.attn_layers:
        #     xl, xa, xv, attn_weights = layer(xl, xa, xv)
        xsl, xsa, xsv = self.q_s(xl, self.query_memory_share), self.q_s(xa, self.query_memory_share), self.q_s(xv, self.query_memory_share)
        xpl, xpa, xpv = self.q_l_p(xl, self.query_memory_private_l), self.q_a_p(xa, self.query_memory_private_a), self.q_v_p(xv, self.query_memory_private_v)
        # sl, sv, sa  = self.st(xsl, xsl, xsl)[0], self.sv(xsv, xsv, xsv)[0], self.sa(xsa, xsa, xsa)[0]
        # pl, pv, pa =  self.pt(xpl, xpl, xpl)[0], self.pv(xpv, xpv, xpv)[0], self.pa(xpv, xpv, xpv)[0]
        # o = (self.st_pooler(xsl), self.sv_pooler(xsv), self.sa_pooler(xsa), self.pt_pooler(xpl), self.pv_pooler(xpv), self.pa_pooler(xpa))
        # sl, sv, sa = self.rnnsl(xsl.transpose(0, 1), None), self.rnnsv(xsv.transpose(0, 1), None), self.rnnsa(xsa.transpose(0, 1), None)
        # pl, pv, pa = self.rnnpl(xpl.transpose(0, 1), None), self.rnnpv(xpv.transpose(0, 1), None), self.rnnpa(xpa.transpose(0, 1), None)
        #o = (self.query_memory_share, self.query_memory_private_l, self.query_memory_private_v, self.query_memory_private_a)
        lav = torch.cat([xsl, xsa, xsv, xpv, xpa, xpl], dim=-1)
        #lav = self.attn(lav, lav, lav)[0]
        lav = self.fc(lav)
        #lav_share = self.gate_share(xsl, xsv, xsa)
        #lav_private = self.gate_private(xpa, xpv, xpa)
        #enhance_lav = self.memory(lav)
        #lav = lav_share + lav_private
        acoustic_vis_embedding = lav
        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )
        #embedding_output = self.memory(embedding_output)
        #embedding_output = self.pff(embedding_output) + embedding_output
        return embedding_output
    
class AdaptiveFusionGate(nn.Module):
    def __init__(self):
        super(AdaptiveFusionGate, self).__init__()
        self.W_hv = nn.Linear(TEXT_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(TEXT_DIM + TEXT_DIM, TEXT_DIM)
        self.W_v = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.W_a = nn.Linear(TEXT_DIM, TEXT_DIM)
    def forward(self, text_embedding, visual, acoustic):
        #torch.Size([48, 50, 768]) torch.Size([48, 50, 47]) torch.Size([48, 50, 74])
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
        embedding_output = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
        return embedding_output
    
class MemeoryQueryLayer(nn.Module):
    def __init__(self, seq, embedding, attn_dropout=0.2):
        super(MemeoryQueryLayer, self).__init__()
        self.attn_dropout = attn_dropout
        self.W_q = nn.Linear(embedding, embedding)
        self.W_k = nn.Linear(embedding, embedding)
        self.W_v = nn.Linear(embedding, embedding)
    def forward(self, x, memory):
        q = self.W_q(x)
        k = self.W_k(memory)
        v = self.W_v(memory)
        #24x50x768 24x100x768 24x100x768
        attn_weights = torch.matmul(q, k.transpose(0, 1))
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        memory_response = FilterMemory(torch.matmul(attn_weights, v))
        return memory_response
