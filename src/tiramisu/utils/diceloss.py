"""
Code reference: 
https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/4
https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
"""

import torch

class DiceLoss:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """
    def __call__(self, output, target, weights=None, ignore_index=None):
        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
        eps = 0.0001

        output = output.exp()
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = torch.tensor([0.9, 0.1]).cuda()

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


# class DiceLoss:
#     """Computes the Sørensen–Dice loss.
#     Note that PyTorch optimizers minimize a loss. In this
#     case, we would like to maximize the dice loss so we
#     return the negated dice loss.
#     Args:
#         true: a tensor of shape [B, 1, H, W].
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         eps: added to the denominator for numerical stability.
#     Returns:
#         dice_loss: the Sørensen–Dice loss.
#     """
#     def __call__(self, logits, true, eps=1e-7):
#         """Computes the Jaccard loss, a.k.a the IoU loss.
#         Note that PyTorch optimizers minimize a loss. In this
#         case, we would like to maximize the jaccard loss so we
#         return the negated jaccard loss.
#         Args:
#             true: a tensor of shape [B, H, W] or [B, 1, H, W].
#             logits: a tensor of shape [B, C, H, W]. Corresponds to
#                 the raw output or logits of the model.
#             eps: added to the denominator for numerical stability.
#         Returns:
#             jacc_loss: the Jaccard loss.
#         """
#         num_classes = logits.shape[1]
#         if num_classes == 1:
#             true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
#             true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#             true_1_hot_f = true_1_hot[:, 0:1, :, :]
#             true_1_hot_s = true_1_hot[:, 1:2, :, :]
#             true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#             pos_prob = torch.sigmoid(logits)
#             neg_prob = 1 - pos_prob
#             probas = torch.cat([pos_prob, neg_prob], dim=1)
#         else:
#             true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
#             true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#             probas = logits # torch.softmax(logits, dim=1)
#         true_1_hot = true_1_hot.type(logits.type())
#         dims = (0,) + tuple(range(2, true.ndimension()))
#         intersection = torch.sum(probas * true_1_hot, dims)
#         cardinality = torch.sum(probas + true_1_hot, dims)
#         union = cardinality - intersection
#         jacc_loss = (intersection / (union + eps)).mean()
#         return (1 - jacc_loss)