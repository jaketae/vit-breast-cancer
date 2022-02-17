import cv2
import numpy as np
import torch
from torch import nn



def get_aggregate_attention_map(image, attentions, power=1):
    """adapted from https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb"""
    att_mat = torch.stack(attentions).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)

    # add identity matrix to account for residual connections
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask1 = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask1 / mask1.max(), image.size)[..., np.newaxis]
    result = (mask**power * np.asarray(image)).astype("uint8")
    return result, mask


def get_last_attention_map(attentions):
    """adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py"""
    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)

    threshold = 0.6
    # We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass.
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]

    # hard code values since all models share same patch size
    w_featmap = 14 * 2
    h_featmap = 14 * 2
    patch_size = 16

    th_attn = th_attn.reshape(nh, w_featmap // 2, h_featmap // 2).float()

    # interpolate
    th_attn = (
        nn.functional.interpolate(
            th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )

    attentions = attentions.reshape(nh, w_featmap // 2, h_featmap // 2)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )
    attentions_mean = np.mean(attentions, axis=0)
    return attentions, attentions_mean
