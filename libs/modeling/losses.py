import torch
from torch.nn import functional as F
from torch import nn
from .models import register_loss
# @torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    dim: tuple =None
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")#这个据说集成了sigmoid
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        if dim is not None:
            loss = loss.sum(dim=dim)
        else:
            loss = loss.sum()
    return loss


@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py
    注意：使用的是偏移量来计算
    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

# @torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@register_loss('TripletLoss')
class TripletLoss(nn.Module):
    def __init__(self,delta,soft_label,neg_aggregate):
        super().__init__()
        self.delta=delta
        self.soft_label=soft_label#soft label控制pos score的聚合方式
        self.neg_aggregate=neg_aggregate#neg_aggregate控制负分的聚合方式，总共两类[mean,max]
    def forward(self,sim_matrix,shot_labels):
        #sim_matrix,shot_labels:[.../None,n_q,n_s],都是0~1之间的数
        row=sim_matrix.shape[-2]
        column=sim_matrix.shape[-1]
        if not self.soft_label:
            shot_labels=shot_labels>0
            neg_labels=torch.logical_not(shot_labels)
        else:
            neg_labels=shot_labels<=0
        
        pos_sim=sim_matrix*shot_labels
        neg_sim=sim_matrix*(neg_labels.to(sim_matrix.dtype))
        #row
        row_mask=shot_labels.sum(dim=-1)>0
        pos_sim_row=pos_sim[row_mask]#去除无正样本的行
        neg_sim_row=neg_sim[row_mask]
        shot_labels_row=shot_labels[row_mask]
        neg_labels_row=neg_labels[row_mask]
        pos_score_row=(pos_sim_row.sum(dim=-1))/(shot_labels_row.sum(dim=-1))#正样本分数根据加权计算
        if self.neg_aggregate=='mean':
            neg_score_row=(neg_sim_row.sum(dim=-1))/(neg_labels_row.sum(dim=-1))#负样本可以采用均值，或者采用极值
        elif self.neg_aggregate=='max':
            neg_score_row=neg_sim_row.max(dim=-1)[0]
        else:
            raise ValueError('not support')
        row_loss=F.relu(self.delta-pos_score_row+neg_score_row).sum(dim=-1)/row

        #column
        column_mask=shot_labels.sum(dim=-2)>0
        pos_sim_col=pos_sim.transpose(-2,-1)[column_mask]
        neg_sim_col=neg_sim.transpose(-2,-1)[column_mask]
        shot_labels_col=shot_labels.transpose(-2,-1)[column_mask]
        neg_labels_col=neg_labels.transpose(-2,-1)[column_mask]
        pos_score_col=(pos_sim_col.sum(dim=-1))/(shot_labels_col.sum(dim=-1))
        if self.neg_aggregate=='mean':
            neg_score_col=(neg_sim_col.sum(dim=-1))/(neg_labels_col.sum(dim=-1))
        elif self.neg_aggregate=='max':
            neg_score_col=neg_sim_col.max(dim=-1)[0]
        col_loss=F.relu(self.delta-pos_score_col+neg_score_col).sum(dim=-1)/column

        return row_loss+col_loss

@register_loss('FocalLoss')
class FocalLoss(nn.Module):
    def __init__(self,soft_label,alpha: float = 0.25,
    gamma: float = 2.0):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.soft_label=soft_label#soft label控制pos score的聚合方式
    def forward(self,sim_matrix,shot_labels):
        #sim_matrix,shot_labels:[.../None,n_q,n_s],都是0~1之间的数
        if not self.soft_label:
            shot_labels=shot_labels>0
        shot_labels=shot_labels.to(sim_matrix.dtype)
        alpha=self.alpha
        

        #row
        row_mask=shot_labels.sum(dim=-1)>0
        sim_row=sim_matrix[row_mask]#去除无正样本的行
        assert torch.all(sim_row >= 0) and torch.all(sim_row <= 1), "sim_row has values out of range [0, 1]"


        shot_labels_row=shot_labels[row_mask]
        assert torch.all(shot_labels_row >= 0) and torch.all(shot_labels_row <= 1), "shot_labels_row contains values other than 0 and 1"
        row_loss=F.binary_cross_entropy(sim_row,shot_labels_row,reduction='none')
        p_t_row = sim_row * shot_labels_row + (1 - sim_row) * (1 - shot_labels_row)#计算预测正确的概率
        row_focal_loss = row_loss * ((1 - p_t_row) ** self.gamma)

        
        #column
        column_mask=shot_labels.sum(dim=-2)>0
        sim_col=sim_matrix.transpose(-2,-1)[column_mask]
        assert torch.all(sim_col >= 0) and torch.all(sim_col <= 1), "sim_col has values out of range [0, 1]"
        shot_labels_col=shot_labels.transpose(-2,-1)[column_mask]
        assert torch.all(shot_labels_col >= 0) and  torch.all(shot_labels_col <= 1), "shot_labels_col contains values other than 0 and 1"
        col_loss=F.binary_cross_entropy(sim_col,shot_labels_col,reduction='none')
        p_t_col = sim_col * shot_labels_col + (1 - sim_col) * (1 - shot_labels_col)#计算预测正确的概率
        col_focal_loss = col_loss * ((1 - p_t_col) ** self.gamma)
        if alpha >= 0:
            alpha_t_row = alpha * shot_labels_row + (1 - alpha) * (1 - shot_labels_row)
            row_focal_loss = (alpha_t_row * row_focal_loss).sum()
            alpha_t_col = alpha * shot_labels_col + (1 - alpha) * (1 - shot_labels_col)
            col_focal_loss= (alpha_t_col * col_focal_loss).sum()

        
        

        return row_focal_loss+col_focal_loss