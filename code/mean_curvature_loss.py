import os
import sys
import math

import torch
import torch.nn.functional as F
from torch import nn
import scipy.optimize as opt

from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized, reduce_dict)
from .matcher import build_matcher

sys.path.append("..")
from utils.vis import save_debug_images_boxes

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # threshold = 15 / 720.
        # self.threshold = nn.Threshold(threshold**2, 0.)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([tgt[:, 0][J].long() for tgt, (_, J) in zip (targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([tgt.shape[0] for tgt in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def continue_k2_peak(self, polys, src_lowers, src_uppers):
        
        src_lowers = src_lowers.data.cpu().numpy()
        src_uppers = src_uppers.data.cpu().numpy()
        polys = polys.data.cpu().numpy()
        max_k2_peak = torch.as_tensor(src_lowers)

        for idx, mnkb in enumerate(polys):
            m,n,k,b = mnkb
            k2_func = lambda x: 0 - ((6*m*x[0] + 2*n)**2) / ((1 + (3*m*x[0]**2+2*n*x[0]+k)**2)**3)
            # k2_func = lambda x: 0 - ((6*polys[0,0]*x[0] + 2*polys[0,1])**2) / ((1 + (3*polys[0,0]*x[0]**2+2*polys[0,1]*x[0]+polys[0,3])**2)**3)
            cons = ({'type': 'ineq', 'fun': lambda x: x[0] - src_lowers[idx]},\
                    {'type': 'ineq', 'fun': lambda x: -x[0] + src_uppers[idx]})
            k2_peak = opt.minimize(k2_func, 0, method = 'SLSQP', constraints=cons)
            max_k2_peak[idx] = 0 - k2_peak.fun
        
        return max_k2_peak.cuda()
           
    def loss_curves(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_curves' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_lowers = outputs['pred_curves'][:, :, 0][idx]
        src_uppers = outputs['pred_curves'][:, :, 1][idx]
        src_polys  = outputs['pred_curves'][:, :, 2:][idx]
        target_lowers = torch.cat([tgt[:, 1][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_uppers = torch.cat([tgt[:, 2][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_points = torch.cat([tgt[:, 3:115][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_polys = torch.cat([tgt[:, 115:][i] for tgt, (_, i) in zip(targets, indices)], dim=0)

        target_xs = target_points[:, :target_points.shape[1] // 2]
        ys = target_points[:, target_points.shape[1] // 2:].transpose(1, 0)
        valid_xs = target_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32)) ** 0.5
        weights = weights / torch.max(weights)

        # Calculate the predicted xs
        pred_xs = src_polys[:,0] * ys **3 + src_polys[:,1] * ys **2 + src_polys[:,2] * ys + src_polys[:,3]

        pred_xs = pred_xs * weights
        pred_xs = pred_xs.transpose(1, 0)
        target_xs = target_xs.transpose(1, 0) * weights
        target_xs = target_xs.transpose(1, 0)

        # Calculate the kappa xs
        pred_k2_s = (torch.abs(6*src_polys[:,0]*ys + 2*src_polys[:,1])) / ((1 + (3*src_polys[:,0]*ys**2+2*src_polys[:,1]*ys+src_polys[:,2])**2)**1.5) #shape [56, 46]
        target_k2_s = (torch.abs(6*target_polys[:,0]*ys + 2*target_polys[:,1])) / ((1 + (3*target_polys[:,0]*ys**2+2*target_polys[:,1]*ys+target_polys[:,2])**2)**1.5)

        # Mean
        pred_k2_s = pred_k2_s * weights
        pred_k2_s = pred_k2_s.transpose(1,0)
        target_k2_s = target_k2_s * weights
        target_k2_s = target_k2_s.transpose(1, 0)

        # Max
        # valid_xs_float = valid_xs.type_as(pred_k2_s)
        # pred_k2_s = pred_k2_s.transpose(1,0) * valid_xs_float
        # target_k2_s = target_k2_s.transpose(1,0) * valid_xs_float
        # pred_k2_max, target_k2_max = pred_k2_s.max(1)[0], target_k2_s.max(1)[0]


        loss_lowers = F.l1_loss(src_lowers, target_lowers, reduction='none')
        loss_uppers = F.l1_loss(src_uppers, target_uppers, reduction='none')
        loss_polys  = F.l1_loss(pred_xs[valid_xs], target_xs[valid_xs], reduction='none')
        loss_mnkb = F.l1_loss(pred_k2_s[valid_xs], target_k2_s[valid_xs], reduction='none')
        # loss_mnkb = F.l1_loss(pred_k2_max, target_k2_max, reduction='none') # loss Max Peak
        # loss_mnkb = F.l1_loss(src_polys[:,0:4], target_polys[:,0:4], reduction='none') # loss mnkb

        losses = {}
        losses['loss_lowers']  = loss_lowers.sum() / num_curves
        losses['loss_uppers']  = loss_uppers.sum() / num_curves
        losses['loss_curves']   = loss_polys.sum() / num_curves
        losses['loss_mnkb']   = loss_mnkb.sum() / num_curves

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_curves, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'curves': self.loss_curves,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_curves, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_curves = sum(tgt.shape[0] for tgt in targets)
        num_curves = torch.as_tensor([num_curves], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_curves)
        num_curves = torch.clamp(num_curves / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_curves))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_curves, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # print("losses " + str(losses.keys()))
        return losses, indices


class LaneLoss(nn.Module):
    def __init__(self,
                 debug_path=None,
                 aux_loss=None,
                 num_classes=None,
                 dec_layers=None
                 ):
        super(LaneLoss, self).__init__()
        self.debug_path  = debug_path
        # matcher_weight_dict = {'loss_ce': 3, 'loss_curves': [10,10,5,5], 'loss_lowers': 2, 'loss_uppers': 2}
        matcher_weight_dict = {'loss_ce': 3, 'loss_curves': 5, 'loss_lowers': 2, 'loss_uppers': 2}
        # cardinality is not used to propagate loss
        matcher = build_matcher(set_cost_class=matcher_weight_dict['loss_ce'],
                                curves_weight=matcher_weight_dict['loss_curves'],
                                lower_weight=matcher_weight_dict['loss_lowers'],
                                upper_weight=matcher_weight_dict['loss_uppers'])
        
        # weight_dict = {'loss_ce':3, 'loss_lowers':2, 'loss_uppers':2, 'loss_ms':10, 'loss_ns':10, 'loss_ks':5, 'loss_bs':5}
        weight_dict = {'loss_ce':3, 'loss_curves':5, 'loss_lowers':2, 'loss_uppers':2, 'loss_mnkb':2}
        losses  = ['labels', 'curves', 'cardinality']

        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes=num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=1.0,
                                      losses=losses)

    def forward(self,
                gt_viz_inputs,
                iteration,
                save,
                viz_split,
                outputs,
                targets):
        gt_cluxy = []
        for tgt in targets:
            tgt_ids = tgt[:,0]
            tgt = tgt[tgt_ids > 0]
            gt_cluxy.append(tgt)

        loss_dict, indices = self.criterion(outputs, gt_cluxy)
        # print(loss_dict)
        # input("here")
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Save detected images during training
        if save and iteration % 30 == 0:
            which_stack = 0
            save_dir = os.path.join(self.debug_path, viz_split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = 'iter_{}_layer_{}'.format(iteration % 5000, which_stack)
            save_path = os.path.join(save_dir, save_name)
            with torch.no_grad():
                # gt_viz_inputs = targets[0] # 图片组
                # tgt_labels = [tgt[:, 0].long() for tgt in targets]
                tgt_labels = targets[:, :, 0]
                pred_labels = outputs['pred_logits'].detach()
                prob = F.softmax(pred_labels, -1)
                scores, pred_labels = prob.max(-1)  # 4 10

                pred_curves = outputs['pred_curves'].detach()
                pred_clua3a2a1a0 = torch.cat([scores.unsqueeze(-1), pred_curves], dim=-1)

                # print(pred_labels)
                # print(tgt_labels.shape)
                # print(pred_clua3a2a1a0.shape)
                # print(tgt_labels)
                # input("?")
                save_debug_images_boxes(gt_viz_inputs,
                                        tgt_curves=torch.dstack((tgt_labels, targets[:, :, 3:-4])),  #  delete mnkb low up  , only remain class xs ys
                                        tgt_labels=tgt_labels,
                                        pred_curves=pred_clua3a2a1a0,
                                        pred_labels=pred_labels,
                                        prefix=save_path)

        return (losses, loss_dict_reduced, loss_dict_reduced_unscaled,
                loss_dict_reduced_scaled, loss_value)
