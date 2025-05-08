from .models import register_layer,make_block,register_block,make_layer
from torch import nn,Tensor
from torch.nn import functional as F
from .blocks import MaskedConv1D,LayerNorm,AffineDropPath
import torch,math
from typing import Optional
from einops import rearrange


@register_block("MaskedConv1DLayer")
@register_layer("MaskedConv1DLayer")
class MaskedConv1DLayer(nn.Module):
    def __init__(self,num_layer,
                 n_in,
                 n_hidden=None,
                 n_out=None,
                 kernel_size=1,
                 stride=1,
                 with_ln=True,
                 act='gelu',
                 pdrop=0,
                 end_act=True
                 ):
        #一个兼容mlp的stride为1的1d卷积层；当它充当嵌入层时，没有dropout,注意，mlp一般没有最后的act和中间的normalization,
        super().__init__()
        self.kernel_size=kernel_size
        self.end_act=end_act
        self.convs = nn.ModuleList()
        self.norms=nn.ModuleList()
        self.dropouts=nn.ModuleList()
        if act=='gelu':
            self.act_layer=nn.GELU()
        elif act=='None':
            self.act_layer=nn.Identity()
        elif act=='relu':
            self.act_layer=nn.ReLU()
        else:
            raise ValueError('unsupport act layer')
        if n_hidden==None:
            n_hidden=n_in
        if n_out==None:
            n_out=n_in
        for idx in range(num_layer):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_hidden
            if idx != num_layer-1:
                out_channels=n_hidden
            else:
                out_channels=n_out
            
            self.convs.append(MaskedConv1D(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=kernel_size//2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.norms.append(
                    LayerNorm(out_channels)
                )
            else:
                self.norms.append(nn.Identity())
            self.dropouts.append(nn.Dropout(pdrop, inplace=True))
    def forward(self,x,mask):
        #x:[...,C,T],mask:[...,1,T],x和mask可以超出三维
        assert len(x.shape)==len(mask.shape)
        assert len(x.shape)>=3
        x_shape=x.shape
        if len(x_shape)>3 :
            
            x=x.reshape(-1,x_shape[-2],x_shape[-1])
            mask_shape=mask.shape
            mask=mask.reshape(-1,mask_shape[-2],mask_shape[-1])
        for idx in range(len(self.convs)):
            x, mask = self.convs[idx](x, mask)
            x=self.norms[idx](x)
            if self.end_act or (idx<(len(self.convs)-1)):#如果最后一层也做激活，或者还没到最后一层，就做激活
                x = self.act_layer(x)
            x=self.dropouts[idx](x)
        if len(x_shape)>3 :
            x=x.reshape(x_shape)
            mask=mask.reshape(mask_shape)
        return x,mask

@register_layer("FormerLayer")
class FormerLayer(nn.Module):
    def __init__(self,self_mixer_cfg,cross_mixer_cfg,path_pdrop):
        super().__init__()
        self.self_mixer=make_block(self_mixer_cfg['block_type'],**self_mixer_cfg['block_cfg'])
        self.cross_mixer_cfg=cross_mixer_cfg
        n_embd=self_mixer_cfg['block_cfg']['n_embd']
        self.n_embd=n_embd
        if cross_mixer_cfg!='None':

            self.cross_mixer=make_block(cross_mixer_cfg['block_type'],**cross_mixer_cfg['block_cfg'])
            self.ln3=LayerNorm(n_embd)
        # ffn
        
        # ok to use conv1d here with stride=1
        self.ffn = MaskedConv1DLayer(
            num_layer=2,
            n_in=n_embd,
            n_hidden=n_embd*4,
            with_ln=False,
            end_act=False
        )#这样才是一个标准的mlp
        #layernorm
        self.ln1=LayerNorm(n_embd)
        self.ln2=LayerNorm(n_embd)
        
        #res skip
        q_stride=self_mixer_cfg['block_cfg'].get('n_qx_stride',1)
        if q_stride > 1:
            self.self_mixer_skip = nn.MaxPool1d(
                kernel_size=q_stride+1, stride=q_stride, padding=(q_stride+1)//2)
        else:
            self.self_mixer_skip = nn.Identity()
        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_embd, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        
    def forward(self, x, mask, cross_y=None, cross_y_mask=None, pos_embd=None):
        #x:[B,C,T],mask:[B,1,T],cross_y:[B,C,T1],cross_y_mask:[B,1,T1]
        out, out_mask = self.self_mixer(self.ln1(x), mask)

        out_mask_float = out_mask.to(out.dtype)
        out = self.self_mixer_skip(x) * out_mask_float + self.drop_path_attn(out)

        # optional 
        if self.cross_mixer_cfg!='None':
            cross_out, cross_out_mask = self.cross_mixer(self.ln3(out), out_mask_float, self.ln3(cross_y), cross_y_mask)
            out_mask_float = out_mask.to(cross_out_mask.dtype)
            out = out* out_mask_float + self.drop_path_attn(cross_out)

        # FFN
        ffn_out,ffn_out_mask=self.ffn(self.ln2(out),out_mask)#一般只有self mixer进行下采样，而cross mixer不进行下采样
        out = out + self.drop_path_mlp( ffn_out* out_mask_float)

        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask
@register_layer("QFormerLayer")
class QFormerLayer(nn.Module):
    def __init__(self,query_num,num_layer,**kwargs):
        super().__init__()
        
        self.qformers=nn.ModuleList()
        for _ in range(num_layer):
            self.qformers.append(FormerLayer(**kwargs))
        self.n_embd=self.qformers[0].n_embd
        self.query_num=query_num
        self.querys=nn.Embedding(self.n_embd,query_num)
        self.fc=MaskedConv1DLayer(
            num_layer=1,
            n_in=self.n_embd,
            kernel_size=query_num,
            stride=query_num,
            with_ln=False,
            end_act=False
        )
        
    def forward(self, x, mask):
        #x:[B,C,T],mask:[B,1,T]
        B,d_model,_=x.shape
        query=self.querys.weight.repeat(B,1,1)#[B,C,Q]
        query_mask=mask.sum(dim=-1).unsqueeze(-1)#[B,1,1]
        query_mask=query_mask.repeat(1,1,self.query_num)#[B,1,Q]

        for qformer in self.qformers:
            query,query_mask=qformer(query,query_mask,x,mask)
            query=query.masked_fill(torch.logical_not(query_mask),0)

        query,query_mask=self.fc(query,query_mask)
        return query,query_mask

@register_layer("Cosine")
class Cosine(nn.Module):
    def __init__(self,x_dim,y_dim,con_dim,with_mlp=True):
        super().__init__()
        self.with_mlp=with_mlp
        if with_mlp:
            self.x_mlp=MaskedConv1DLayer(
                num_layer=2,
                n_in=x_dim,
                n_hidden=con_dim*4,
                n_out=con_dim,
                with_ln=True,
                end_act=False
            )
            self.y_mlp=MaskedConv1DLayer(
                num_layer=2,
                n_in=y_dim,
                n_hidden=con_dim*4,
                n_out=con_dim,
                with_ln=True,
                end_act=False
            )
    def forward(self,x,x_mask,y,y_mask):
        if self.with_mlp:
            x,_=self.x_mlp(x,x_mask)#[num_x,C,1]
            y,_=self.y_mlp(y,y_mask)#[num_y,C,1]
        sim_matrix=F.cosine_similarity(x.permute(0,2,1), y.permute(2,0,1),dim=-1)#[num_x,num_y]
        return sim_matrix
