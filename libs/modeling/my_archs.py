import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator,make_layer,make_loss
from .blocks import MaskedConv1D, Scale, LayerNorm,get_sinusoid_encoding
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from einops import rearrange
from ..utils import batched_nms
from .head import  PtTransformerClsHead, PtTransformerRegHead
from basic_utils import min_max,iou

import torch.distributed as dist
# from memory_profiler import profile
from line_profiler import profile
def get_memory(str):
    if int(os.environ["LOCAL_RANK"])>0:
        return
    device = torch.cuda.current_device()

    # 获取总内存和已用内存
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated()

    # 计算已用内存（字节为单位）
    used_memory = allocated_memory

    # 转换为GB
    total_memory_gb = total_memory / (1024 ** 3)
    used_memory_gb = used_memory / (1024 ** 3)

    print(str+f"已用显存: {used_memory_gb:.2f} GB")
    return
def sigmoid_inverse(x):
    return torch.log(x / (1 - x))
def minmax_normalize(tensor, dim=0):
    """
    对给定的张量进行 Min-Max 正则化。

    参数:
        tensor (torch.Tensor): 要正则化的输入张量。
        dim (int): 进行正则化的维度。

    返回:
        torch.Tensor: 正则化后的张量。
    """
    min_val = tensor.min(dim=dim, keepdim=True).values
    max_val = tensor.max(dim=dim, keepdim=True).values
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-10)  # 防止除以零
    return normalized_tensor
class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        if dist.is_initialized():
            world_size= torch.distributed.get_world_size()
            output = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(output, tensor)
            ctx.rank =int(os.environ["LOCAL_RANK"])

            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, 0)
        else:
            return tensor
    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized():
            return (
                grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
                
            )
        return grad_output
@register_meta_arch("MultiTaskArch")
class MultiTaskArch(nn.Module):
    def __init__(self,
                 text_encoder_cfg,obj_encoder_cfg,video_encoder_cfg,multiscale_encoder_cfg,tasks,nlq_heads_cfg,max_shot_num,max_query,max_seq_len,vtm_heads_cfg=None,**kwargs):
        super().__init__()
        self.allgather = AllGather_multi.apply
        self.text_encoder_cfg=text_encoder_cfg
        self.text_encoder=self.make_encoder(text_encoder_cfg)
        self.obj_encoder_cfg=obj_encoder_cfg
        self.obj_encoder=self.make_encoder(obj_encoder_cfg)
        self.video_encoder=self.make_encoder(video_encoder_cfg)

        n_embd=video_encoder_cfg[0]['layer_cfg']['n_out']
        self.n_embd=n_embd
        self.max_seq_len=max_seq_len
        pos_embd = get_sinusoid_encoding(max_seq_len, n_embd) / (n_embd ** 0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)

        self.multiscale_encoder_cfg=multiscale_encoder_cfg
        self.multiscale_encoder=self.make_encoder(multiscale_encoder_cfg)
        self.tasks=tasks
        self.kwargs=kwargs
        self.point_generator = self.make_generator()
        self.make_NLQ_head(nlq_heads_cfg)
        if 'VTM' in tasks:
            self.make_VTM_head(vtm_heads_cfg)
        self.normal_loss=False
        self.max_shot_num=max_shot_num#goalstep:3306;3308,tacos:41x4=164
        self.max_query=max_query#goalstep:554,tacos:230x4=920
    def make_VTM_head(self,vtm_heads_cfg):
        # if 'vtm_loss_cfg' in vtm_heads_cfg:
        #     vtm_loss_cfg=vtm_heads_cfg['vtm_loss_cfg']
        #     self.vtm_loss_cfg=vtm_loss_cfg
        #     self.vtm_loss=make_loss(vtm_loss_cfg['loss_type'],**vtm_loss_cfg['loss_cfg'])
        if 'loss_weight' not in vtm_heads_cfg:
            vtm_heads_cfg['loss_weight']=1.0
        if 'multiscale' not in vtm_heads_cfg:
            vtm_heads_cfg['multiscale']=False
        if 'soft_label' not in vtm_heads_cfg:
            vtm_heads_cfg['soft_label']=False
        self.vtm_heads_cfg=vtm_heads_cfg
        self.shot_aggregator_cfg=vtm_heads_cfg['shot_aggregator_cfg']
        self.txt_aggregator_cfg=vtm_heads_cfg.get('txt_aggregator_cfg',self.shot_aggregator_cfg)
        self.shot_aggregator=make_layer(self.shot_aggregator_cfg['layer_type'],**self.shot_aggregator_cfg['layer_cfg'])
        txt_aggregator_layer_cfg=self.txt_aggregator_cfg.get('layer_cfg',{})
        self.txt_aggregator=make_layer(self.txt_aggregator_cfg['layer_type'],**txt_aggregator_layer_cfg)
        similarity_head_cfg=vtm_heads_cfg['similarity_head_cfg']
        self.similarity_head=make_layer(similarity_head_cfg['layer_type'],**similarity_head_cfg['layer_cfg'])
        return
    def make_NLQ_head(self,nlq_heads_cfg):
        self.nlq_heads_cfg=nlq_heads_cfg
        self.nlq_cls_head = PtTransformerClsHead(
            **nlq_heads_cfg['cls_head_cfg']
        )
        self.nlq_reg_head = PtTransformerRegHead(
            **nlq_heads_cfg['reg_head_cfg']
        )
        self.loss_normalizer_momentum=nlq_heads_cfg['loss_normalizer_momentum']
        self.loss_normalizer=nlq_heads_cfg['loss_normalizer']
        self.nlq_label_smoothing=nlq_heads_cfg['train_label_smoothing']
        self.nlq_reg_loss_weight=nlq_heads_cfg['reg_loss_weight']
        self.train_center_sample_radius=nlq_heads_cfg['center_sample_radius']#gt的中心点和point的中心点距离与stride的比值不得超过这个系数。
        self.num_classes=self.nlq_heads_cfg['cls_head_cfg']['num_classes']

        self.test_pre_nms_thresh    =self.nlq_heads_cfg['pre_nms_thresh']
        self.test_pre_nms_topk      =self.nlq_heads_cfg['pre_nms_topk']
        self.test_duration_thresh   =self.nlq_heads_cfg['duration_thresh']
        self.test_iou_threshold     =self.nlq_heads_cfg['iou_threshold']
        self.test_min_score         =self.nlq_heads_cfg['min_score']
        self.test_max_seg_num       =self.nlq_heads_cfg['max_seg_num']
        return
    def make_generator(self,):
        fpn_strides = [self.kwargs['scale_factor'] ** i for i in range(
            self.kwargs['fpn_start_level'], self.multiscale_encoder_cfg[0]['layer_num']+ 1
        )]
        assert len(fpn_strides) == len(self.kwargs['regression_range'])
        window_size=self.multiscale_encoder_cfg[0]['layer_cfg']['mha_win_size']
        if window_size > 1:
            self.max_stride=max(fpn_strides)*(window_size// 2) * 2  
        else:
            self.max_stride=max(fpn_strides)
        
            

        generator=make_generator(
            'point',
            **{
                'max_seq_len': self.max_seq_len * self.kwargs['max_buffer_len_factor'],
                'fpn_strides': fpn_strides,
                'regression_range': self.kwargs['regression_range']
            }
        )
        return generator
    @property
    def device(self):
        if torch.cuda.is_available():
            try:
                return torch.device('cuda:{}'.format(int(os.environ["LOCAL_RANK"])))
            except:
                return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def get_shot_feat(self,shot_vid,bs_shots):
        # shot_nums=[]
        bs_shots_len=[]
        # bs_shots=[]
        max_shots_len=1
        for shots in bs_shots:
            shots=(shots).to(torch.int)
            shots_len=shots[1]-shots[0]
            bs_shots_len.append(shots_len)
            # shot_nums.append(shots_len.shape[0]-1)#这里的-1，和后面的-1都是为了去除padding部分
            max_shots_len=max(max_shots_len,shots_len[:-1].max())
        bs_shots_feats=[]
        bs_shots_masks=[]
        for idx,(shots,shot_feats) in enumerate(zip(bs_shots,shot_vid)):  
            shots=shots[:,:-1].permute(1,0)#去除padding
            for shot in shots:
                shot_len=shot[1]-shot[0]
                if shot_len>0:
                    shot_feat=shot_feats[:,shot[0]:shot[1]]
                    shot_feat_pad=F.pad(shot_feat, (0, max_shots_len-shot_len))
                    shot_mask=torch.ones((shot_len),device=shot_feat.device)
                    shot_mask=F.pad(shot_mask, (0, max_shots_len-shot_len))                   
                else:
                    shot_feat_pad=torch.zeros_like(shot_feats[:,:max_shots_len])
                    shot_mask=torch.zeros((max_shots_len),device=shot_feat_pad.device)
                bs_shots_feats.append(shot_feat_pad)
                bs_shots_masks.append(shot_mask)
        bs_shots_feats=torch.stack(bs_shots_feats,dim=0)#这里可能爆显存
        bs_shots_masks=torch.stack(bs_shots_masks,dim=0).bool().unsqueeze(1)
        return bs_shots_feats,bs_shots_masks
    def make_encoder(self,args):
        # if args=='None':
        #     return nn.Identity()
        encoder=nn.ModuleList()
        for layer_args in args:
            layers=nn.ModuleList()
            for _ in range(layer_args['layer_num']):
                layers.append(make_layer(layer_args['layer_type'],**layer_args['layer_cfg']))
            encoder.append(layers)
        return encoder
    # @profile
    def forward(self, video_list,**kwargs):
        
        # get_memory('begin')
        src_vid, src_vid_mask = self.preprocessing(video_list)#[B,1/C,T]
        # get_memory("src_vid:")
        src_txt, src_txt_mask = self.query_preprocessing(video_list)#[B,1/C,L]
        # get_memory("src_txt:")
        src_obj,src_obj_mask=self.process_object(video_list)##[B,N,1/C,T]/None,T代表视频长度，N代表物体数量，L代表query长度
        # get_memory("src_obj:")
        enc_txt=self.encode_text(src_txt,src_txt_mask)#[B,C,L]
        # get_memory("enc_txt:")
        enc_obj=self.encode_obj(src_obj,src_obj_mask,enc_txt, src_txt_mask)#[B*N,C,T]/None
        # get_memory('enc_obj')
        enc_vid=self.encode_video(src_vid, src_vid_mask,enc_txt, src_txt_mask,enc_obj,src_obj_mask)
        enc_vid_multiscale,enc_vid_multiscale_mask=self.encode_multiscale(enc_vid, src_vid_mask,enc_txt, src_txt_mask)#nlevel x [bs,Ti,C/1]
        points = self.point_generator(enc_vid_multiscale)
        nlq_cls_logits,nlq_offsets, enc_vid_multiscale_mask=self.predict_NLQ(enc_vid_multiscale,enc_vid_multiscale_mask)

        if self.training:
            losses={'final_loss':0}
            losses=self.cal_NLQ_loss(video_list,nlq_cls_logits,nlq_offsets, enc_vid_multiscale_mask,points,losses)
            # get_memory('nlq')
            if 'VTM' in self.tasks:
                enc_vid_pure=self.encode_video(src_vid, src_vid_mask,None, None,None,None)
                # get_memory('enc_vid_pure')
                src_video_txt,src_video_txt_mask,query_num=self.video_query_preprocessing(video_list)
                enc_video_txt=self.encode_text(src_video_txt,src_video_txt_mask)#[query_num,C,L]
                # get_memory('enc_video_txt')
                if self.vtm_heads_cfg['multiscale']:
                    enc_vid_pure,_=self.encode_multiscale(enc_vid_pure, src_vid_mask,None, None)
                bs_shots=[x['shots'].to(self.device).to(torch.int) for x in video_list]
                sim_matrix,shot_query,shot_query_mask=self.predict_VTM(bs_shots,enc_vid_pure,enc_video_txt,src_video_txt_mask)#[txt_nums,shots_num]
                if self.nlq_heads_cfg.get('course_learning',False):
                    vtm_mask=self.get_nlq_course_mask(enc_txt,src_txt_mask,shot_query,shot_query_mask,bs_shots,kwargs['t'],enc_vid_pure.shape[-1])
                    assert video_list[0]['segments'] is not None, "GT action labels does not exist"
                    gt_segments = [x['segments'].to(self.device) for x in video_list]
                    enc_vid_multiscale_mask=self.add_gt_label(bs_shots,gt_segments,vtm_mask,enc_vid_multiscale_mask)
                losses=self.cal_VTM_loss(sim_matrix,bs_shots,losses,video_list)
                # get_memory('all')
            return losses

        else:
            if 'video_query_nums' in video_list[0]:
                results = self.inference_bayesian(
                video_list, points, enc_vid_multiscale_mask,
                nlq_cls_logits, nlq_offsets, self.num_classes
                )
                return results
            results = self.inference(
                video_list, points, enc_vid_multiscale_mask,
                nlq_cls_logits, nlq_offsets, self.num_classes
            )
            # results = self.inference_plus(
            #     video_list, points, enc_vid_multiscale_mask,
            #     nlq_cls_logits, nlq_offsets,src_vid, src_vid_mask,enc_txt,src_txt_mask,
            # )
            return results
    def get_nlq_course_mask(self,enc_txt,src_txt_mask,shot_query,shot_query_mask,bs_shots,t,max_seq_len):
        shot_nums=[]
        for shots in bs_shots:
            # shots=(shots).to(torch.int)
            # shots_len=shots[1]-shots[0]
            shot_nums.append(shots.shape[1]-1)#这里的-1，和后面的-1都是为了去除padding部分
        bs_shot_query=torch.split(shot_query,split_size_or_sections=shot_nums,dim=0)#bs x [shot_num,C,1]
        bs_shot_query_mask=torch.split(shot_query_mask,split_size_or_sections=shot_nums,dim=0)#bs x [shot_num,1,1]
        bs_choose_masks=[]
        for idx,(shot_query_,shot_query_mask_,shots,shot_num )in enumerate(zip(bs_shot_query,bs_shot_query_mask,bs_shots,shot_nums)):
            txt_query=enc_txt[idx].unsqueeze(0)
            txt_query_mask=src_txt_mask[idx].unsqueeze(0)
            txt_query,txt_query_mask=self.txt_aggregator(txt_query, txt_query_mask)#[bs,c,1]

            sim_matrix=self.similarity_head(txt_query,txt_query_mask,shot_query_,shot_query_mask_).flatten(0)#[1,n_shot]
            topk_score,topk_idxs=sim_matrix.topk(k=math.ceil(shot_num*(1-t) ))

            choose_masks=torch.zeros((max_seq_len),device=sim_matrix.device).bool()
            for choose_idx in topk_idxs:
                choose_masks[shots[0,choose_idx]:shots[1,choose_idx]]=True
            bs_choose_masks.append(choose_masks)
        bs_choose_masks=torch.stack(bs_choose_masks,dim=0).unsqueeze(1)#[bs,1,2560]
        return bs_choose_masks
    def add_gt_label(self,bs_shots,gt_segments,bs_choose_mask,fpn_masks):
        #应该加一个gt_segments=None的分支
        bs_shot_mask=[]
        if gt_segments is not None:
            for (shots,gt_segment,choose_mask) in zip(bs_shots,gt_segments,bs_choose_mask):
                shots=shots.permute(1,0).unsqueeze(0)#[1,n,2]
                gt_segment=gt_segment.unsqueeze(1).to(self.device)#[n,1,2]
                
                shot_iou=iou(shots,gt_segment).sum(dim=0)[:-1]#去掉padding,[shot_num]
                shot_label=shot_iou.bool()
                # shot_label=shot_iou/(shot_iou.max()+1e-4)#由于naq特征缺了几帧，导致存在一些proposal的groundtruth并不存在
                shot_label_idx=shot_label.nonzero().flatten()
                for shot_label_i in shot_label_idx:
                    choose_mask[:,shots[0,shot_label_i,0]:shots[0,shot_label_i,1]]=True
                bs_shot_mask.append(choose_mask)
        
            bs_shot_mask=torch.stack(bs_shot_mask,dim=0)#[bs,1,2560]
        else:    
            bs_shot_mask=bs_choose_mask
        bs_shot_mask=torch.logical_and(bs_shot_mask,fpn_masks[0].unsqueeze(1))#排除padding部分
        bs_shot_masks=tuple()
        bs_shot_masks+=(bs_shot_mask,)
        scale_factor=self.multiscale_encoder_cfg[0]['layer_cfg']['n_ds_strides'][0]
        for idx in range(len(fpn_masks)-1):
            T=bs_shot_mask.shape[-1]
            bs_shot_mask=F.interpolate(
            bs_shot_mask.float(),
            size=T // scale_factor,
            mode='nearest'
            ).bool()
            bs_shot_masks+=(bs_shot_mask,)
        bs_shot_masks = [x.squeeze(1) for x in bs_shot_masks]
        return bs_shot_masks
    def inference_plus(self,
                video_list, points, enc_vid_multiscale_mask,
                nlq_cls_logits, nlq_offsets,src_vid, src_vid_mask,enc_txt,src_txt_mask
            ):
        enhance='minmax'
        if enhance == 'two_stage':
            enc_vid_multiscale_mask=self.get_vtm_mask(src_vid, src_vid_mask,enc_txt,src_txt_mask,video_list,0.5,enc_vid_multiscale_mask)
            results = self.inference(
                    video_list, points, enc_vid_multiscale_mask,
                    nlq_cls_logits, nlq_offsets, self.num_classes
                )
        elif enhance=='minmax':
            shot_scores=self.get_vtm_mask(src_vid, src_vid_mask,enc_txt,src_txt_mask,video_list,0.5,enc_vid_multiscale_mask,True)
            new_nlq_cls_logits=[]
            for nlq_cls_logit ,shot_score in zip(nlq_cls_logits,shot_scores):
                nlq_cls_score=nlq_cls_logit.sigmoid()
                # print(nlq_cls_score.shape,shot_score.shape)
                nlq_cls_score=minmax_normalize(nlq_cls_score,dim=1)#[bs,ti,1]
                shot_score=minmax_normalize(shot_score,dim=1).unsqueeze(-1)#[bs,ti]
                nlq_cls_score=(nlq_cls_score+shot_score)/2
                new_nlq_cls_logits.append(sigmoid_inverse(nlq_cls_score))
            results = self.inference(
                    video_list, points, enc_vid_multiscale_mask,
                    new_nlq_cls_logits, nlq_offsets, self.num_classes
                )
        else:
            raise ValueError('not support enhance')
            results = self.inference(
                video_list, points, enc_vid_multiscale_mask,
                nlq_cls_logits, nlq_offsets, self.num_classes
            )
        return results
    def get_vtm_mask(self,src_vid, src_vid_mask,enc_txt,src_txt_mask,video_list,t,fpn_masks,soft=False):
        max_seq_len=src_vid.shape[-1]
        enc_vid_pure=self.encode_video(src_vid, src_vid_mask,None, None,None,None)
        bs_shots=[x['shots'].to(self.device).to(torch.int) for x in video_list]
        # sim_matrix=self.predict_VTM(bs_shots,enc_vid_pure,enc_txt,src_txt_mask)#[txt_nums,shots_num]
        bs_shots_feats,bs_shots_masks=self.get_shot_feat(enc_vid_pure,bs_shots)#[bs,c,len]
        shot_query,shot_query_mask=self.shot_aggregator(bs_shots_feats, bs_shots_masks)

        shot_nums=[]
        for shots in bs_shots:
            shot_nums.append(shots.shape[1]-1)#这里的-1，和后面的-1都是为了去除padding部分
        bs_shot_query=torch.split(shot_query,split_size_or_sections=shot_nums,dim=0)#bs x [shot_num,C,1]
        bs_shot_query_mask=torch.split(shot_query_mask,split_size_or_sections=shot_nums,dim=0)#bs x [shot_num,1,1]
        bs_choose_masks=[]
        for idx,(shot_query_,shot_query_mask_,shots,shot_num )in enumerate(zip(bs_shot_query,bs_shot_query_mask,bs_shots,shot_nums)):
            txt_query=enc_txt[idx].unsqueeze(0)
            txt_query_mask=src_txt_mask[idx].unsqueeze(0)
            txt_query,txt_query_mask=self.txt_aggregator(txt_query, txt_query_mask)#[bs,c,Q]
            # sim_matrix=self.predict_VTM(bs_shots,shot_query_,shot_query_mask_,txt_query,txt_query_mask).flatten()
            sim_matrix=self.similarity_head(txt_query,txt_query_mask,shot_query_,shot_query_mask_).flatten(0)#[n_shot]
            
            if not soft:
                topk_score,topk_idxs=sim_matrix.topk(k=math.ceil(shot_num*(1-t) ))
                choose_masks=torch.zeros((max_seq_len),device=sim_matrix.device).bool()
                for choose_idx in topk_idxs:
                    choose_masks[shots[0,choose_idx]:shots[1,choose_idx]]=True
            else:
                shot_len=shots.shape[-1]
                for i in range(shot_len-1):
                    choose_masks=torch.zeros((max_seq_len),device=sim_matrix.device)
                
                    choose_masks[shots[0,i]:shots[1,i]]=sim_matrix[i]
            bs_choose_masks.append(choose_masks)
        bs_choose_masks=torch.stack(bs_choose_masks,dim=0).unsqueeze(1)#[bs,1,2560]

        bs_shot_mask=bs_choose_masks
        # bs_shot_mask=torch.logical_and(bs_shot_mask,fpn_masks[0].unsqueeze(1))#排除padding部分
        bs_shot_masks=tuple()
        bs_shot_masks+=(bs_shot_mask,)
        scale_factor=self.multiscale_encoder_cfg[0]['layer_cfg']['n_ds_strides'][0]
        for idx in range(len(fpn_masks)-1):
            T=bs_shot_mask.shape[-1]
            bs_shot_mask=F.interpolate(
            bs_shot_mask.float(),
            size=T // scale_factor,
            mode='nearest'
            )
            if not soft:
                bs_shot_mask=bs_shot_mask.bool()
            bs_shot_masks+=(bs_shot_mask,)
        bs_shot_masks = [x.squeeze(1) for x in bs_shot_masks]
        return bs_shot_masks
    def cal_VTM_loss(self,sim_matrix,bs_shots,losses,video_list):
        shot_labels=self.get_shot_label(bs_shots,video_list)
        shot_shape=shot_labels.shape
        shot_shape=torch.tensor([[shot_shape[0],shot_shape[1]]],device=self.device)
        shot_labels=F.pad(shot_labels,(0,self.max_shot_num-shot_shape[0,1],0,self.max_query-shot_shape[0,0]))
        shot_shapes=self.allgather(shot_shape,)
        shot_labels = self.allgather(shot_labels)
        if dist.is_initialized():
            world_size= torch.distributed.get_world_size()
        else:
            world_size=1
        shot_split=[self.max_query for _ in range(world_size)]
        bs_shot_labels=torch.split(shot_labels,shot_split)
        cnt=[0,0]
        whole_shot_labels=torch.zeros_like(sim_matrix)
        for (shot_labels_,shot_shape_) in zip(bs_shot_labels,shot_shapes):

            whole_shot_labels[cnt[0]:cnt[0]+shot_shape_[0],cnt[1]:cnt[1]+shot_shape_[1]]=shot_labels_[:shot_shape_[0],:shot_shape_[1]]
            cnt[0]=cnt[0]+shot_shape_[0]
            cnt[1]=cnt[1]+shot_shape_[1]
        vtm_loss=self.infonce_loss(sim_matrix,whole_shot_labels)
        
        vtm_loss=vtm_loss/self.loss_normalizer/world_size
        losses.update({
            'vtm_loss':vtm_loss
        })
        losses['final_loss']=losses['final_loss']+vtm_loss*self.vtm_heads_cfg['loss_weight']
        return losses
    def infonce_loss(self,sim_matrix,shot_labels,temperature=0.05):
        if not self.vtm_heads_cfg['soft_label']:
            shot_labels=shot_labels>0
        i_sm = F.softmax(sim_matrix/temperature, dim=1)
        i_sm_valid=i_sm[shot_labels.sum(dim=1)>0]
        shot_labels_i=shot_labels[shot_labels.sum(dim=1)>0]
        idiag = torch.log(torch.sum(i_sm_valid*shot_labels_i, dim=1) )
        loss_i = idiag.sum() / len(idiag)

        j_sm = F.softmax(sim_matrix.t()/temperature, dim=1)
        j_sm_valid=j_sm[shot_labels.sum(dim=0)>0]
        shot_labels_j=shot_labels.t()[shot_labels.sum(dim=0)>0]
        jdiag = torch.log(torch.sum(j_sm_valid*shot_labels_j , dim=1) )#防微杜渐，从一开始避免inf的出现
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i -loss_j
    def get_shot_label(self,bs_shots,video_list):
        shot_labels=[]
        cnt=0
        shot_nums_sum=0
        for shots_ in bs_shots:
            shot_nums_sum=shot_nums_sum+shots_.shape[1]-1
        for (shots,video) in zip(bs_shots,video_list):
            video_segments=video['video_segments']
            shots=shots.permute(1,0).unsqueeze(0)#[1,n,2]
            shot_num=shots.shape[1]-1
            for video_segment in video_segments:
                gt_segment=video_segment.unsqueeze(1).to(self.device)#[n,1,2]
                shot_iou=iou(shots,gt_segment).sum(dim=0)[:-1]#去掉padding,[shot_num]
                shot_iou_norm=shot_iou/(shot_iou.max()+1e-4)#由于naq特征缺了几帧，导致存在一些proposal的groundtruth并不存在
                shot_iou_mask=shot_iou.bool()
                shot_label_idx=shot_iou_mask.nonzero().flatten()
                global_shot_label_idx=shot_label_idx+cnt
                shot_label=torch.zeros(shot_nums_sum,dtype=shot_iou_norm.dtype).to(shots.device)
                shot_label[global_shot_label_idx]=shot_iou_norm[shot_iou_mask]
                shot_labels.append(shot_label)
            cnt=cnt+shot_num
        shot_labels=torch.stack(shot_labels,dim=0)#[query_num,shot_num]    
        return shot_labels
    def predict_VTM(self,bs_shots,enc_vid_pure,enc_video_txt,src_video_txt_mask):
        
        if self.vtm_heads_cfg['multiscale']:
            bs_shots_feats=[]
            bs_shots_masks=[]
            for enc_vid_p in enc_vid_pure:
                bs_shots_feat,bs_shots_mask=self.get_shot_feat(enc_vid_p,bs_shots)#[bs,c,len]
                higher_bs_shots=[]
                for shots in bs_shots:
                    shots_left=shots[0]
                    shots_right=shots[1]
                    higher_shots_left=torch.ceil(shots_left/2)
                    higher_shots_right=torch.floor(shots_right/2)
                    higher_shots=torch.stack([higher_shots_left,higher_shots_right],dim=0).int()
                    higher_bs_shots.append(higher_shots)
                bs_shots=higher_bs_shots
                bs_shots_feats.append(bs_shots_feat)
                bs_shots_masks.append(bs_shots_mask)

        else:
            bs_shots_feats,bs_shots_masks=self.get_shot_feat(enc_vid_pure,bs_shots)#[bs,c,len]
        # if self.shot_aggregator_cfg['layer_type']=='QFormerLayer':
        shot_query,shot_query_mask=self.shot_aggregator(bs_shots_feats, bs_shots_masks)
        local_shot_query,local_shot_query_mask=shot_query,shot_query_mask
        shot_num=shot_query.shape[0]
        shot_query=F.pad(shot_query,(0,0,0,0,0,self.max_shot_num-shot_num))
        shot_query_mask=F.pad(shot_query_mask,(0,0,0,0,0,self.max_shot_num-shot_num))
        # if self.txt_aggregator_cfg['layer_type'] in ['QFormerLayer','MaskedMaxPooling','MaskedEot']:
        txt_query,txt_query_mask=self.txt_aggregator(enc_video_txt, src_video_txt_mask)#[bs,c,Q]
        query_num=txt_query.shape[0]
        txt_query=F.pad(txt_query,(0,0,0,0,0,self.max_query-query_num))
        txt_query_mask=F.pad(txt_query_mask,(0,0,0,0,0,self.max_query-query_num))
        shot_query = self.allgather(shot_query)
        shot_query_mask= self.allgather(shot_query_mask)
        txt_query = self.allgather(txt_query)
        txt_query_mask = self.allgather(txt_query_mask)
        shot_valid_mask=shot_query_mask.flatten()
        shot_query=shot_query[shot_valid_mask]
        shot_query_mask=shot_query_mask[shot_valid_mask]
        query_valid_mask=txt_query_mask.flatten()
        txt_query=txt_query[query_valid_mask]
        txt_query_mask=txt_query_mask[query_valid_mask]

        sim_matrix=self.similarity_head(txt_query,txt_query_mask,shot_query,shot_query_mask)#[n_txt,n_shot]
        return sim_matrix,local_shot_query,local_shot_query_mask
    def prepare_for_deformable(self,bs_shots,mul_enc_vid_pure, mul_enc_vid_pure_mask):
        spatial_shapes = []
        for fpn_feat in mul_enc_vid_pure:
            spatial_shapes.append(fpn_feat.shape[-1])
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=self.device)#[F]
        memory=torch.cat(mul_enc_vid_pure,dim=2).permute(0,2,1)#[B,sumT,C]
        memory_key_padding_mask=torch.cat(mul_enc_vid_pure_mask,dim=2).squeeze(1)#[B,sumT]
        max_num_shot=max([shots.shape[1]-1 for shots in bs_shots])
        reference_points=[]
        tgt_masks=[]
        for shots in bs_shots:
            center=(shots[1]+shots[0])/2
            wide=shots[1]-center
            num_shot=center.shape[0]
            reference_point=torch.stack([center,wide],dim=1)#[n,2]
            reference_point_pad=F.pad(reference_point,(0,0,0,max_num_shot-num_shot),"constant",0)
            reference_points.append(reference_point_pad)
            tgt_mask= torch.ones_like(center)
            tgt_mask=F.pad(tgt_mask,(0,max_num_shot-num_shot),"constant",0)
            tgt_masks.append(tgt_mask)
        reference_points=torch.stack(reference_points,dim=1)#[n,bs,2]
        tgt_masks=torch.stack(tgt_masks,dim=0).bool()#[bs,n_shots]
        return memory,tgt_masks,memory_key_padding_mask,reference_points,spatial_shapes
    @torch.no_grad()
    def inference_bayesian(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets, num_classes
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        #out_cls_logits:F (List) [B, T_i, Class]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]
        vid_query_nums=[x['video_query_nums'] for x in video_list]
        vid_query_idxs=[int(x['query_id'].split('_')[-1])+1 for x in video_list]
        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes,vid_query_num,vid_query_idx) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes,vid_query_nums,vid_query_idxs)
        ):
            # gather per-video outputs

            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # 创建一个正态分布对象
            feature_len=fpn_masks_per_vid[0].shape[0]
            T=fpn_masks_per_vid[0].sum()
            normal_dist = torch.distributions.Normal(vid_query_idx/vid_query_num*T, int(math.ceil(T/10)))

            # 生成一组点（你想要计算的值）
            x = torch.arange(feature_len,device=fpn_masks_per_vid[0].device)  

            # 计算每个点的概率密度函数 (PDF) 值
            pdf_values = torch.exp(normal_dist.log_prob(x))
            pdf_values=pdf_values/(pdf_values.max())
            pdf_values=pdf_values.unsqueeze(0).unsqueeze(0)
            for layer_idx in range(len(cls_logits_per_vid)):
                
                cls_logits_per_vid[layer_idx]=sigmoid_inverse(1e-9+(cls_logits_per_vid[layer_idx].sigmoid())*(pdf_values.squeeze(0).permute(1,0)))
                feature_len=feature_len // 2
                if feature_len>=1:
                    pdf_values=F.interpolate(
                        pdf_values,
                        size=feature_len,
                        mode='nearest'
                        )
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, num_classes,
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocessing
        results = self.postprocessing(results)

        return results
    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets, num_classes
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        #out_cls_logits:F (List) [B, T_i, Class]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, num_classes,
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocessing
        results = self.postprocessing(results)

        return results
    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
            num_classes,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
        ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_max=pred_prob==(pred_prob.max())
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            keep_idxs1=torch.logical_or(keep_idxs1,keep_max)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(
                topk_idxs, num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_long=seg_areas==(seg_areas.max())
            keep_idxs2 = seg_areas > self.test_duration_thresh
            keep_idxs2=torch.logical_or(keep_idxs2,keep_long)

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()

            segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
            # 4: repack the results
            processed_results.append(
                {'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels}
            )

        return processed_results

    def NLQ_losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets,
            losses
    ):
        # fpn_masks:F (List) [B, T_i]
        # out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        #bs_shots_score:B x[shot_num,1,1]
        #shot_labels:Bx[shot_num]

        #normal_*:B(list) [sum_T,1]
        # normal_probs_cls = torch.cat(normal_probs_cls,dim=1).permute(1,0)        # [b, all_points]
        # normal_probs_reg_left = torch.cat([x[0] for x in normal_probs_reg],dim=1).permute(1,0)   # [b, all_points]
        # normal_probs_reg_right = torch.cat([x[1] for x in normal_probs_reg],dim=1).permute(1,0)

        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        num_classes = gt_target.shape[-1]

        # optional label smoothing
        gt_target *= 1 - self.nlq_label_smoothing
        gt_target += self.nlq_label_smoothing / (num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum',
            dim=1
        )
        # if self.norm_cls_loss:
        #     normal_probs_cls[~pos_mask] = 1.0
        # else:
        #     normal_probs_cls=torch.ones_like(normal_probs_cls)
        # cls_loss *= normal_probs_cls[valid_mask]
        cls_loss = cls_loss.sum()
        cls_loss /= self.loss_normalizer
        # 2. regression using IoU/GIoU loss (defined on positive samples)
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
            )
            # if not self.norm_reg_loss:
            #     normal_probs_reg_left = torch.ones_like(normal_probs_reg_left)
            #     normal_probs_reg_right = torch.ones_like(normal_probs_reg_right)
            # reg_loss *= (normal_probs_reg_left[pos_mask] + normal_probs_reg_right[pos_mask]) / 2.0
            # reg_loss *= normal_probs_cls[pos_mask]              # for one gaussian
            reg_loss = reg_loss.sum()
            reg_loss /= self.loss_normalizer

        if self.nlq_reg_loss_weight >= 0:
            loss_weight = self.nlq_reg_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)
        nlq_loss = cls_loss + reg_loss * loss_weight
        losses.update({
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
        })
        losses['final_loss']=losses['final_loss']+nlq_loss
        return losses
    def cal_NLQ_loss(self,video_list,nlq_cls_logits,nlq_offsets, enc_vid_multiscale_mask,points,losses):
        gt_cls_labels, gt_offsets=self.get_NLQ_label(video_list,points)
        losses = self.NLQ_losses(
                enc_vid_multiscale_mask,
                nlq_cls_logits, nlq_offsets,
                gt_cls_labels, gt_offsets,
                 losses
            )
        return losses
    def get_NLQ_label(self,video_list,points):
        assert video_list[0]['segments'] is not None, "GT action labels does not exist"
        gt_segments = [x['segments'].to(self.device) for x in video_list]

        assert video_list[0]['one_hot_labels'] is not None, "GT action labels does not exist"
        gt_labels = [x['one_hot_labels'].to(self.device) for x in video_list]
        
        # compute the gt labels for cls & reg
        # list of prediction targets
        concat_points = torch.cat(points, dim=0)

        gt_cls, gt_offset = [], []
        normal_probs_cls, normal_probs_reg = [], []
        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            assert len(gt_segment) == len(gt_label), (gt_segment, gt_label)
            cls_targets, reg_targets,(normal_prob_cls, normal_prob_reg_left, normal_prob_reg_right)  = self.label_points_single_video(
                concat_points, gt_segment, gt_label, self.num_classes
            )
            # "cls_targets: " #points, num_classes
            # "reg_targets: " #points, 2
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            normal_probs_cls.append(normal_prob_cls)
            normal_probs_reg.append([normal_prob_reg_left, normal_prob_reg_right])

        return gt_cls, gt_offset
    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, num_classes):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x 2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]#[5080,1]
        reg_targets = torch.stack((left, right), dim=-1)

        normal_prob_cls, normal_prob_reg_left, normal_prob_reg_right=None,None,None

            # center of all segments F T x N
        center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
        # center sampling based on stride radius
        # compute the new boundaries:
        # concat_points[:, 3] stores the stride
        t_mins = \
            center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
        t_maxs = \
            center_pts + concat_points[:, 3, None] * self.train_center_sample_radius

        # prevent t_mins / maxs from over-running the action boundary
        # left: torch.maximum(t_mins, gt_segs[:, :, 0])
        # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
        # F T x N (distance to the new boundary)
        cb_dist_left = concat_points[:, 0, None] \
                        - torch.maximum(t_mins, gt_segs[:, :, 0])
        cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                        - concat_points[:, 0, None]
        # F T x N x 2
        center_seg = torch.stack(
            (cb_dist_left, cb_dist_right), -1)

        # F T x N
        inside_gt_seg_mask = center_seg.min(-1)[0] > 0


        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]

        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # limit the regression range for each location and inside the center radius
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))

        # if there are still more than one ground-truths for one point
        # pick the ground-truth with the shortest duration for the point (easiest to regress)
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        # make sure that each point can only map with at most one ground-truth
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        # gt_label_one_hot = F.one_hot(gt_label, num_classes).to(reg_targets.dtype)
        gt_label_one_hot = gt_label.to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets,(normal_prob_cls, normal_prob_reg_left, normal_prob_reg_right) 
    
    def predict_NLQ(self,enc_vid_multiscale,enc_vid_multiscale_mask):
        nlq_cls_logits = self.nlq_cls_head(enc_vid_multiscale,enc_vid_multiscale_mask)
        # out_offset: List[B, 2, T_i]
        nlq_offsets = self.nlq_reg_head(enc_vid_multiscale,enc_vid_multiscale_mask)
        nlq_cls_logits = [x.permute(0, 2, 1) for x in nlq_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        nlq_offsets = [x.permute(0, 2, 1) for x in nlq_offsets]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        enc_vid_multiscale_mask = [x.squeeze(1) for x in enc_vid_multiscale_mask]
        return nlq_cls_logits,nlq_offsets, enc_vid_multiscale_mask
    def encode_multiscale(self,src_vid, src_vid_mask,enc_txt, src_txt_mask):
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (src_vid,)
        out_masks += (src_vid_mask,)
        for multiscale_layer in self.multiscale_encoder[0]:
                src_vid,src_vid_mask=multiscale_layer(src_vid, src_vid_mask,enc_txt, src_txt_mask)
                out_feats += (src_vid,)
                out_masks += (src_vid_mask,)
        for multiscale_layer in self.multiscale_encoder[1]:
                out_feats,out_masks=multiscale_layer(out_feats,out_masks)

        return out_feats,out_masks
    def encode_video(self,src_vid, src_vid_mask,enc_txt, src_txt_mask,enc_obj,src_obj_mask):
        T=src_vid.shape[-1]
        
        for video_layer in self.video_encoder[0]:
            src_vid,src_vid_mask=video_layer(src_vid, src_vid_mask)
        if self.training:

            pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)
        else:
            if T>self.max_seq_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            src_vid = src_vid + pe[:, :, :T] * src_vid_mask.to(src_vid.dtype)

        for video_layer in self.video_encoder[1]:
            src_vid,src_vid_mask=video_layer(src_vid, src_vid_mask,enc_obj,src_obj_mask,enc_txt, src_txt_mask)#enc_obj:[bs x maxlen,C,max_seq_len]因为只接2d和3d;src_obj_mask#[bs x maxlen,1,max_seq_len]
        return src_vid

    def encode_obj(self,src_obj,src_obj_mask,enc_txt, src_txt_mask):
        #obj_embd
        if src_obj is None:
            return None
        for obj_layer in self.obj_encoder[0]:
            src_obj,src_obj_mask=obj_layer(src_obj,src_obj_mask)
        B,hidden_dim,_=enc_txt.shape
        T=src_obj.shape[-1]
        src_obj=src_obj.view(B,-1,hidden_dim,T)
        for obj_layer in self.obj_encoder[1]:
            src_obj,src_obj_mask=obj_layer(src_obj,src_obj_mask,enc_txt, src_txt_mask)
        src_obj=src_obj.view(-1,hidden_dim,T)
        return src_obj
    def encode_text(self,src_txt,src_txt_mask):
        # for text_layers in self.text_encoder:
        for text_layer in self.text_encoder[0]:
            src_txt,src_txt_mask=text_layer(src_txt,src_txt_mask)
        if self.text_encoder_cfg[1].get('use_abs_pe',False):
            T=src_txt.shape[-1]
            pe = self.pos_embd
            # add pe to x
            src_txt = src_txt + pe[:, :, :T] * src_txt_mask.to(src_txt.dtype)
        for text_layer in self.text_encoder[1]:
            src_txt,src_txt_mask=text_layer(src_txt,src_txt_mask)
        return src_txt
    @torch.no_grad()
    def query_preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['query_feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        # batch input shape B, T, C
        batch_shape = [len(feats), feats[0].shape[0], max_len]
        batched_inputs = feats[0].new_full(batch_shape, padding_val)
        for feat, pad_feat in zip(feats, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def video_query_preprocessing(self, video_list, padding_val=0.0):
    
        query_num=[len(x['video_query_feats']) for x in video_list]
        feats=[]
        for x in video_list:
            feats.extend(x['video_query_feats'])

        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        # batch input shape B, T, C
        batch_shape = [len(feats), feats[0].shape[0], max_len]
        batched_inputs = feats[0].new_full(batch_shape, padding_val)
        for feat, pad_feat in zip(feats, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks,query_num

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()
        # max_len=self.max_seq_len
        max_len = math.ceil(max_len/self.max_stride)*self.max_stride
        # batch input shape B, C, T
        batch_shape = [len(feats), feats[0].shape[0], max_len]
        batched_inputs = feats[0].new_full(batch_shape, padding_val)
        for feat, pad_feat in zip(feats, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    def process_object(self,video):
        if 'object_feats' not in video[0]:
            return None,None
        object_feats = [x['object_feats'] for x in video]
        torch_object_feats=[]
        #确定object_feat的最大长度

        feats = [x['feats'] for x in video]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()
        if self.training:
            assert max_len <= self.max_seq_len, (
                "Input length must be smaller than max_seq_len during training", max_len, self.max_seq_len)
        # set max_len to self.max_seq_len
        # max_seq_len = self.max_seq_len
        max_seq_len=math.ceil(max_len/self.max_stride)*self.max_stride
        #获取mask和object_feat_maxlen
        feats_len=[]
        for object_feat in object_feats:
            if object_feat is None:
                pad_feat_len=torch.zeros(max_seq_len,dtype=torch.int32).to(self.device)
            else:
                feat_len=[]
                for object_f in object_feat:
                    if object_f is None:
                        feat_len.append(0)
                    else:
                        feat_len.append(object_f.shape[0])
                feat_len=torch.tensor(feat_len).to(self.device)
                pad_feat_len=F.pad(feat_len,(0,max_seq_len-len(feat_len)),"constant",0)
            feats_len.append(pad_feat_len)
        feats_len=torch.stack(feats_len,dim=0)#[bs,max_seq_len]
        object_feat_maxlen=max(1,feats_len.max().item())
        batched_masks = (torch.arange(object_feat_maxlen)[None,None, :].to(self.device)) < (feats_len.unsqueeze(-1))#[bs,max_seq_len,maxlen]

        # print(batched_masks)
        #获取padding object_feat
        obj_dim=self.obj_encoder_cfg[0]['layer_cfg']['n_in']
        for object_feat in object_feats:
            if object_feat is None:
                pad_object_feat=torch.zeros((max_seq_len,object_feat_maxlen,obj_dim)).to(self.device)
            else:
                torch_object_feat=[]
                # if len(object_feat)==0:
                #     print("object_feat为空",[v['query_id'] for v in video])
                for object_f in object_feat:
                    if object_f is None:
                        torch_object_feat.append(torch.zeros(object_feat_maxlen,obj_dim).to(self.device))
                    else:
                        pad_object_f=F.pad(object_f, (0, 0, 0, object_feat_maxlen-object_f.shape[0]), "constant", 0).to(self.device)
                        torch_object_feat.append(pad_object_f.float())
                # if len(torch_object_feat)==0:
                #     print("torch_object_feat为空",[v['query_id'] for v in video])
                torch_object_feat=torch.stack(torch_object_feat,dim=0)#[T,maxlen,C]torch_object_feat为空[]，说明object_feat为空
                T,_,_=torch_object_feat.shape
                pad_object_feat=F.pad(torch_object_feat,(0,0,0,0,0,max_seq_len-T),"constant",0)
            torch_object_feats.append(pad_object_feat)
        torch_object_feats=torch.stack(torch_object_feats,dim=0).to(self.device)#[bs,max_seq_len,maxlen,C]
        torch_object_feats=rearrange(torch_object_feats,"b t o c -> (b o) c t")
        batched_masks=rearrange(batched_masks,"b t o -> (b o) 1 t")
        return torch_object_feats,batched_masks