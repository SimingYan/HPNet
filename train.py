import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from trainer import Trainer
from option import build_option
from utils.loss_utils import compute_embedding_loss, compute_normal_loss, \
        compute_param_loss, compute_nnl_loss, compute_miou, compute_type_miou_abc
from utils.main_utils import npy
from utils.abc_utils import mean_shift, compute_entropy, construction_affinity_matrix_type, \
        construction_affinity_matrix_normal
import scipy.stats as stats

class MyTrainer(Trainer):

    def process_batch(self, batch_data_label, postprocess=False):
 
        inputs_xyz_th = (batch_data_label['gt_pc']).float().cuda().permute(0,2,1)
        inputs_n_th = (batch_data_label['gt_normal']).float().cuda().permute(0,2,1)
        
        if self.opt.input_normal:
            affinity_feat, type_per_point, normal_per_point, param_per_point, sub_idx = self.model(inputs_xyz_th, inputs_n_th, postprocess=postprocess)
        else:
            affinity_feat, type_per_point, param_per_point, sub_idx = self.model(inputs_xyz_th, inputs_n_th, postprocess=postprocess)
        
        inputs_xyz_sub = torch.gather(inputs_xyz_th, -1, sub_idx.unsqueeze(1).repeat(1,3,1))
        N_gt = (batch_data_label['gt_normal']).float().cuda()
        N_gt = torch.gather(N_gt, 1, sub_idx.unsqueeze(-1).repeat(1,1,3))
        I_gt = torch.gather(batch_data_label['I_gt'], -1, sub_idx)
        T_gt = torch.gather(batch_data_label['T_gt'], -1, sub_idx)

        loss_dict = {}
        
        if 'f' in self.opt.loss_class:
            # network feature loss
            feat_loss, pull_loss, push_loss = compute_embedding_loss(affinity_feat, I_gt)
            loss_dict['feat_loss'] = feat_loss
        if 'n' in self.opt.loss_class:
            # normal angle loss
            normal_loss = compute_normal_loss(normal_per_point, N_gt)
            loss_dict['normal_loss'] = self.opt.normal_weight * normal_loss
        if 'p' in self.opt.loss_class:
            T_param_gt = torch.gather(batch_data_label['T_param'], 1, sub_idx.unsqueeze(-1).repeat(1,1,22))
            # parameter loss
            param_loss = compute_param_loss(param_per_point, T_gt, T_param_gt)
            loss_dict['param_loss'] = self.opt.param_weight * param_loss
        if 'r' in self.opt.loss_class:
            # primitive nnl loss
            type_loss = compute_nnl_loss(type_per_point, T_gt)
            loss_dict['nnl_loss'] = self.opt.type_weight * type_loss

        total_loss = 0
        for key in loss_dict:
            if 'loss' in key:
                total_loss += loss_dict[key]

        if postprocess:
                
            affinity_matrix = construction_affinity_matrix_type(inputs_xyz_sub, type_per_point, param_per_point, self.opt.sigma)
           
            affinity_matrix_normal = construction_affinity_matrix_normal(inputs_xyz_sub, N_gt, sigma=self.opt.normal_sigma, knn=self.opt.edge_knn) 

            obj_idx = batch_data_label['index'][0]
             
            spec_embedding_list = []
            weight_ent = []

            # use network feature
            feat_ent = self.opt.feat_ent_weight - float(npy(compute_entropy(affinity_feat)))
            weight_ent.append(feat_ent)
            spec_embedding_list.append(affinity_feat)
            
            # use geometry distance feature
            topk = self.opt.topK            
            e, v = torch.lobpcg(affinity_matrix, k=topk, niter=10)
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)

            dis_ent = self.opt.dis_ent_weight - float(npy(compute_entropy(v)))
            
            weight_ent.append(dis_ent)
            spec_embedding_list.append(v)
             
            # use edge feature
            edge_topk = self.opt.edge_topK
            e, v = torch.lobpcg(affinity_matrix_normal, k=edge_topk, niter=10)
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)
            
            edge_ent = self.opt.edge_ent_weight - float(npy(compute_entropy(v)))
           
            weight_ent.append(edge_ent)
            spec_embedding_list.append(v)
          
            # combine features
            weighted_list = []
            norm_weight_ent = weight_ent / np.linalg.norm(weight_ent)
            for i in range(len(spec_embedding_list)):
                weighted_list.append(spec_embedding_list[i] * weight_ent[i])

            spectral_embedding = torch.cat(weighted_list, dim=-1)
            
            spec_cluster_pred = mean_shift(spectral_embedding, bandwidth=self.opt.bandwidth)
            cluster_pred = spec_cluster_pred
            miou = compute_miou(spec_cluster_pred, I_gt)
            loss_dict['miou'] = miou
            miou = compute_type_miou_abc(type_per_point, T_gt, cluster_pred, I_gt)
            loss_dict['type_miou'] = miou
 
        return total_loss, loss_dict
        

if __name__=='__main__':
    FLAGS = build_option()
    trainer = MyTrainer(FLAGS)
    trainer.train()
