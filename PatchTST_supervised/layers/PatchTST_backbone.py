__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from utils.masking import LocalMask, localMask
from data_provider.data_cluster import cluster_result

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False, isGpu=0, dataset=None,
                 verbose:bool=False, feature_mix=0, d_mix=128, mask_kernel_ratio=1, reducing_kernel=False, add_std=False, cluster=0, cluster_size=3, orthogonal=0, layer_pos_embed=0, **kwargs):
        
        super().__init__()
        
        # RevI__
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, feature_mix=feature_mix, mask_kernel_ratio=mask_kernel_ratio, isGpu=isGpu, dataset=dataset,
                                reducing_kernel=reducing_kernel, cluster=cluster, cluster_size=cluster_size, layer_pos_embed=layer_pos_embed, **kwargs)

        # Head
        if add_std:
            self.head_nf = (d_model+1) * patch_num
        else:
            self.head_nf = d_model * patch_num
        if cluster:
            self.n_vars = cluster
        else:
            self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.add_std = add_std
        self.orthogonal = orthogonal

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, d_ff=d_ff, head_dropout=head_dropout, dataset=dataset,
                                    feature_mix=feature_mix, d_mix=d_mix, activation=act, cluster=cluster, cluster_size=cluster_size, orthogonal=orthogonal)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]

        if self.revin and self.add_std:
            stdev = self.revin_layer.stdev.permute(0, 2, 1)
            stdev = stdev.unsqueeze(3).expand(-1, -1, 1, z.shape[3])
            z = torch.cat([z, stdev], dim=2)

        if self.orthogonal:
            z, reg = self.head(z)
        else:
            z = self.head(z)                                                                # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)

        if self.orthogonal:
            return z, reg

        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, d_ff=128, head_dropout=0, feature_mix=0, d_mix=128, activation="gelu", cluster=0, cluster_size=3, orthogonal=0, dataset=None):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        self.feature_mix = feature_mix
        self.cluster = cluster
        self.cluster_size = cluster_size
        self.orthogonal = orthogonal
        self.dataset = dataset
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

            if feature_mix == 2:
                self.time_linear = nn.Sequential(nn.Linear(nf, nf*2),
                                                 get_activation_fn(activation),
                                                 nn.Dropout(head_dropout),
                                                 nn.Linear(nf*2, nf))
                self.feature_linear = nn.Sequential(nn.Linear(n_vars, 64),
                                                    get_activation_fn(activation),
                                                    nn.Dropout(head_dropout),
                                                    nn.Linear(64, n_vars))
                self.linear = nn.Linear(nf, target_window)
                self.norm = nn.LayerNorm(nf)
                self.norm2 = nn.LayerNorm(n_vars)

            if cluster:
                self.linears = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                self.flattens = nn.ModuleList()
                for i in range(self.cluster_size):
                    self.flattens.append(nn.Flatten(start_dim=-2))
                    self.linears.append(nn.Linear(nf if feature_mix == 0 else d_mix, target_window))
                    self.dropouts.append(nn.Dropout(head_dropout))
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            if self.feature_mix == 2:
                x = self.flatten(x)                       # x: [bs x nvars x d_model * patch_num]
                # x = self.norm(x)
                x = self.time_linear(x)                   # x: [bs x nvars x d_ff]

                x2 = x.permute(0, 2, 1)
                x2 = self.norm2(x2)                   
                x2 = self.feature_linear(x2)              
                x2 = x2.permute(0, 2, 1)                  # x2: [bs x nvars x d_ff]

                if self.orthogonal:
                    x3 = x.permute(0, 2, 1).detach()
                    x3 = self.norm2(x3)                   
                    x3 = self.feature_linear(x3)              
                    x3 = x3.permute(0, 2, 1)              
                    reg = torch.mean(torch.abs(torch.mean(x.detach() * x3, dim=-1)))

                x = x + x2
                x = x.unsqueeze(-1)

            if self.cluster:
                cluster_labels = cluster_result(self.dataset, self.cluster, self.cluster_size)
                cluster = int(torch.max(cluster_labels) + 1)
                cluster_counts = torch.unique(cluster_labels, return_counts=True)[1]
                order = torch.sort(cluster_labels).indices
                order2 = torch.sort(order).indices
                x_out = []
                for i in range(cluster):
                    for j in range(self.cluster_size):
                        z = self.flattens[j](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                        z = self.linears[j](z)                    # z: [bs x target_window]
                        # z = self.dropouts[j](z)
                        if j < cluster_counts[i]:
                            x_out.append(z)
                x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
                x = x[:, order2]

            else:
                x = self.flatten(x)
                x = self.linear(x)
                # x = self.dropout(x)

        if self.orthogonal:
            return x, reg
        
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, isGpu=0, dataset=None,
                 pe='zeros', learn_pe=True, verbose=False, feature_mix=True, mask_kernel_ratio=1, reducing_kernel=False, cluster=0, cluster_size=3, layer_pos_embed=0, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.cluster = cluster
        self.cluster_size = cluster_size
        self.dataset = dataset
        
        # Input encoding
        q_len = patch_num
        if self.cluster:
            self.W_P = nn.Linear(patch_len*self.cluster_size, d_model)
        else:
            self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        layer_pos_embed = self.W_pos if layer_pos_embed else None
        self.layer_pos_embed = layer_pos_embed

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, isGpu=isGpu,
                                   feature_mix=feature_mix, c_in=c_in, mask_kernel_ratio=mask_kernel_ratio, reducing_kernel=reducing_kernel, layer_pos_embed=layer_pos_embed)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        if self.cluster:
            cluster_labels = cluster_result(self.dataset, self.cluster, self.cluster_size)
            cluster = int(torch.max(cluster_labels) + 1)
            x = x.permute(0, 3, 1, 2)
            xs = []
            for i in range(cluster):
                xi = x[:, :, cluster_labels == i]
                xi = F.pad(xi, (0, 0, 0, self.cluster_size - xi.shape[2]), "constant", 0)    # x: [bs x patch_num x cluster_size x patch_len]
                # xi = F.pad(xi, (0, 0, 0, self.cluster_size - xi.shape[2]), "replicate", 0)    # x: [bs x patch_num x cluster_size x patch_len]
                xi = torch.reshape(xi, (x.shape[0], x.shape[1], -1))                         # x: [bs x patch_num x cluster_size * patch_len]
                xi = self.W_P(xi)                                                            # x: [bs x patch_num x d_model]
                xs.append(xi)
            x = torch.stack(xs, dim=1)                                                       # x: [bs x cluster_num x patch_num x d_model]
            n_vars = x.shape[1]
        else:
            # Input encoding
            n_vars = x.shape[1]
            x = x.permute(0,1,3,2)                                               # x: [bs x nvars x patch_num x patch_len]
            x = self.W_P(x)                                                      # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        if self.layer_pos_embed == None:
            u = self.dropout(u + self.W_pos)  
        else:
            u = self.dropout(u)                                                  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu', isGpu=0,
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, feature_mix=True, c_in=None, mask_kernel_ratio=1, reducing_kernel=False, layer_pos_embed=None):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention, isGpu=isGpu,
                                                      pre_norm=pre_norm, store_attn=store_attn, feature_mix=feature_mix, c_in=c_in, layer_pos_embed=layer_pos_embed,
                                                      mask_kernel_ratio=(0.5**(i) if reducing_kernel else mask_kernel_ratio)) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False, isGpu=0,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False, feature_mix=True, c_in=None, mask_kernel_ratio=1, layer_pos_embed=None):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        elif "layer" in norm.lower():
            self.norm_attn = nn.LayerNorm(d_model)
        else:
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.InstanceNorm1d(d_model), Transpose(1,2))

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        elif "layer" in norm.lower():
            self.norm_ffn = nn.LayerNorm(d_model)
        else:
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.InstanceNorm1d(d_model), Transpose(1,2))


        self.feature_ff = nn.Sequential(Transpose(1,3),
                                nn.Linear(c_in, d_ff, bias=bias),
                                get_activation_fn(activation),
                                # nn.Dropout(dropout),
                                nn.Linear(d_ff, c_in, bias=bias),
                                Transpose(1,3))
        
        self.dropout_feature = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_feature = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        elif "layer" in norm.lower():
            self.norm_feature = nn.LayerNorm(d_model)
        else:
            self.norm_feature = nn.Sequential(Transpose(1,2), nn.InstanceNorm1d(d_model), Transpose(1,2))

        # self.mask = LocalMask(q_len, q_len * mask_kernel_ratio, device="cpu")
        self.mask = localMask(q_len, q_len * mask_kernel_ratio).to("cuda" if isGpu else "cpu")

        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.c_in = c_in
        self.feature_mix = feature_mix
        self.layer_pos_embed = layer_pos_embed


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        if attn_mask == None:
            attn_mask = self.mask

        if self.layer_pos_embed != None:
            src = src + self.layer_pos_embed
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.feature_mix == 1:
            src2 = torch.reshape(src, (-1, self.c_in, src.shape[-2], src.shape[-1]))      # [bs x nvars x patch_num x d_model]
            src2 = self.feature_ff(src2)                                                  
            src2 = torch.reshape(src2, (-1, src.shape[-2], src.shape[-1]))                # [bs * nvars x patch_num x d_model]
            src2 = self.norm_feature(src2)
            src = src + src2

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self attention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

