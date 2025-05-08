import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from .blocks import LayerNorm,LocalMaskedMHCA,MaskedMHCA,ObjectMaskedMHA,MaskedMHA,AffineDropPath,MaskedConv1D
import pkg_resources
from .models import register_block,register_layer
def reverse_tensor_based_on_mask(x, mask):
    # hidden_states: (B, D, L)
    #         mask:(B,L)
    result = []  # 创建一个张量的副本以保存结果
    for (x_i,mask_i) in zip(x,mask):
        # 获取有效元素的索引
        valid_len=mask_i.sum()
        valid_x_i=x_i[:,:valid_len]

        # 翻转有效元素
        reversed_valid_x_i = valid_x_i.flip([-1])
        # 在结果张量中替换原来的元素
        result.append(torch.cat([reversed_valid_x_i,x_i[:,valid_len:]],dim=-1))
    result=torch.stack(result,dim=0)
    return result



from mamba_ssm import Mamba
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
class MaskedBiMamba(Mamba):
    def __init__(
        self,
        **kwargs
    ):
        #继承自videomamba：/home/feng_yi_sen/VideoMamba/mamba/mamba_ssm/modules/mamba_simple.py
        super(MaskedBiMamba,self).__init__(**kwargs)
        
    def forward(self, hidden_states,mask, inference_params=None, T=1):
        """
        hidden_states: (B, L, D)
        mask:(B,L)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    reverse_tensor_based_on_mask(xz,mask),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + reverse_tensor_based_on_mask(out_b,mask), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out
@register_layer('ObjectMambaBlock')
class ObjectMambaBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd=384,  # dimension of the input features
            n_head=4,  # number of attention heads
            n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # dimension of the hidden layer in MLP
            act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for the projection / MLP
            path_pdrop=0.1,  # drop path rate
            mha_win_size=9,  # > 0 to use window mha
            use_rel_pe=False,  # if to add rel position encoding to attention
            use_cross_modal=True,  # if to add cross_modal attention
            mamba_arch=['mamba1','mlp']
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        if not (len(mamba_arch)>2 and mamba_arch[2]=='none'):
            self.ln_obj_mlp = LayerNorm(n_embd)
            self.ln_gate_mlp = LayerNorm(n_embd*2)
        # specify the attention module
        self.mamba_arch=mamba_arch
        if mamba_arch[0]=='mamba1':
            self.mamba=Mamba(d_model=n_embd)
        elif mamba_arch[0]=='mamba2':
            self.mamba=Mamba2(d_model=n_embd)
        elif mamba_arch[0]=='bimamba1':
            self.mamba=MaskedBiMamba(d_model=n_embd)
        elif mamba_arch[0]=='bimamba2':
            self.mamba=MaskedBiMamba2(d_model=n_embd)
        elif mamba_arch[0].startswith('zamba'):
            _,mamba_nums=mamba_arch[0].split('_')
            self.mamba=nn.ModuleList()
            for _ in range(int(mamba_nums)):
                self.mamba.append(MaskedBiMamba2(d_model=n_embd))
        else :
            raise ValueError("not support mamba arch")
        self.mamba_skip = nn.Identity()
        self.ln_mamba=LayerNorm(n_embd)
        if mamba_arch[1].endswith('sa'):
            if mha_win_size > 1:
                self.attn = LocalMaskedMHCA(
                    n_embd,
                    n_head,
                    window_size=mha_win_size,
                    n_qx_stride=n_ds_strides[0],
                    n_kv_stride=n_ds_strides[1],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    use_rel_pe=use_rel_pe  # only valid for local attention
                )
            else:
                self.attn = MaskedMHCA(
                    n_embd,
                    n_head,
                    n_qx_stride=n_ds_strides[0],
                    n_kv_stride=n_ds_strides[1],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop
                )
        elif mamba_arch[1]=='mlp':
            self.mamba_mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_embd*4, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_embd*4, n_embd, 1),
            nn.Dropout(proj_pdrop, inplace=True),
            )
        elif mamba_arch[1]!='none':
            raise ValueError('no implement module')
        if not (len(mamba_arch)>2 and mamba_arch[2]=='none'):
            self.obj_attn = ObjectMaskedMHA(
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                )
            self.ln_obj = LayerNorm(n_embd)
            self.obj_pool_skip = nn.Identity()

        self.use_cross_modal = use_cross_modal
        if use_cross_modal:
            self.cross_attn = MaskedMHA(
                n_embd,
                n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )
            self.ln3 = LayerNorm(n_embd)
            self.cross_pool_skip = nn.Identity()

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )
        if not (len(mamba_arch)>2 and mamba_arch[2]=='none'):
            self.obj_mlp=nn.Sequential(
                nn.Conv1d(n_embd, n_hidden, 1),
                act_layer(),
                nn.Dropout(proj_pdrop, inplace=True),
                nn.Conv1d(n_hidden, n_out, 1),
                nn.Dropout(proj_pdrop, inplace=True),
            )
            self.gate_mlp=nn.Sequential(
                nn.Conv1d(2*n_embd, n_hidden, 1),
                act_layer(),
                nn.Dropout(proj_pdrop, inplace=True),
                nn.Conv1d(n_hidden, n_out, 1),
                nn.Dropout(proj_pdrop, inplace=True),
            )
        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
            if not (len(mamba_arch)>2 and mamba_arch[2]=='none'):
                self.drop_path_obj_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
            self.drop_path_mamba=AffineDropPath(n_embd, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()
            if not (len(mamba_arch)>2 and mamba_arch[2]=='none'):
                self.drop_path_obj_mlp = nn.Identity()
            self.drop_path_mamba=nn.Identity()

    def forward(self, x, mask,src_obj,src_obj_mask, cross_y=None, cross_y_mask=None, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        # print('x:',x.norm())
        #  downsample in the multi-head local attention
        if self.mamba_arch[1]=="ahead_sa":
            out, out_mask = self.attn(self.ln1(x), mask)
            # print('out:',out.norm())
            out_mask_float = out_mask.to(out.dtype)
            out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)#[bs,c,t]
        else:
            out=x
            out_mask_float=mask.to(out.dtype)
        mamba_in=self.ln_mamba(out)
        mamba_in=rearrange(mamba_in,"b c t -> b t c")
        if self.mamba_arch[0] in ['bimamba1','bimamba2']:
            mamba_in_mask=mask.squeeze(1)#[bs,t]
            mamba_out=self.mamba(mamba_in,mamba_in_mask)
        elif self.mamba_arch[0] in ['mamba1','mamba2']:
            mamba_out=self.mamba(mamba_in)
        elif self.mamba_arch[0].startswith('zamba'):
            mamba_in_mask=mask.squeeze(1)#[bs,t]
            for idx in range(len(self.mamba)):
                mamba_in=self.mamba[idx](mamba_in,mamba_in_mask)
            mamba_out=mamba_in
        mamba_out=rearrange(mamba_out,"b t c -> b c t")
        mamba_out=self.mamba_skip(out)*out_mask_float+self.drop_path_mamba(mamba_out)
        
        if self.mamba_arch[1]=="behind_sa":
            out, out_mask = self.attn(self.ln1(mamba_out), mask)
            # print('out:',out.norm())
            # out_mask_float = out_mask.to(out.dtype)
            out = self.pool_skip(mamba_out) * out_mask_float + self.drop_path_attn(out)#[bs,c,t]
        elif self.mamba_arch[1]=='mlp':
            # FFN
            out = mamba_out + self.drop_path_attn(self.mamba_mlp(self.ln1(mamba_out)) * out_mask_float)
            out_mask=mask
        else:
            out=mamba_out
            out_mask=mask
            # out_mask_float=mask.to(out.dtype)
        # print('out:',out.norm())
        # object attention 分支
        if (len(self.mamba_arch)>2 and self.mamba_arch[2]=='none') or src_obj is None:
            obj_out=out
        else:
            obj_out, _ = self.obj_attn(self.ln_obj(out), out_mask_float, self.ln_obj(src_obj), src_obj_mask)
            obj_out=self.obj_pool_skip(out) * out_mask_float + self.drop_path_attn(obj_out)
            obj_out = obj_out + self.drop_path_obj_mlp(self.obj_mlp(self.ln_obj_mlp(obj_out)) * out_mask_float)
        

        # optional cross_modal attention
        if self.use_cross_modal and cross_y is not None:
            # print("inside")
            cross_out, cross_out_mask = self.cross_attn(self.ln3(out), out_mask_float, self.ln3(cross_y), cross_y_mask)
            # print('cross_out:',cross_out.norm())
            out_mask_float = out_mask.to(cross_out_mask.dtype)
            out = self.cross_pool_skip(out) * out_mask_float + self.drop_path_attn(cross_out)
            # print('out:',out.norm())
            # FFN
            out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        if len(self.mamba_arch)>2 and self.mamba_arch[2]=='none':
            pass
        else:
            alpha=self.gate_mlp(self.ln_gate_mlp(torch.cat([obj_out,out],dim=1))* out_mask_float).sigmoid()#[bs,c,t]
            
            out=alpha*obj_out+out*(1-alpha)
        # print('out:',out.norm())
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask
