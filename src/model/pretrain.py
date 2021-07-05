# modified from https://github.com/TengdaHan/CoCLR
# MoCo-related code is modified from https://github.com/facebookresearch/moco
# SwaV-related code is from https://github.com/facebookresearch/swav
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
sys.path.append('../')
from backbone.select_backbone import select_backbone


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ViCC(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    With SwaV clustering and without momentum encoder https://arxiv.org/abs/2006.09882  
    '''
    def __init__(self, 
                network='s3d', 
                dim=128, 
                K=1920, 
                m=0.999, 
                T=0.1,
                ################
                nmb_prototypes=0,
                nmb_views=0,
                world_size=0,
                epsilon=0.05,
                sinkhorn_iterations=3,
                views_for_assign=0,
                improve_numerical_stability=False,
                #################
                ):
        '''
        dim (D): feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048 CoCLR / 3840 SwaV / 1920 ViCC)
        m: moco momentum of updating key encoder (default: 0.999), only used for MoCo/CoCLR
        T: softmax temperature (default: 0.07 CoCLR / 0.1 Swav,ViCC)

        nmb_prototypes (C) (default: 300)
        nmb_views: amount of views used in a list, e.g. [2] or [2,2]
        epsilon: regularization parameter for Sinkhorn-Knopp algorithm
        sinkhorn_iterations: number of iterations in Sinkhorn-Knopp algorithm
        views_for_assign: list of views id used for computing assignments
        improve_numerical_stability: improves numerical stability in Sinkhorn-Knopp algorithm
        '''
        super(ViCC, self).__init__()

        self.dim = dim 
        self.K = K
        self.m = m
        self.T = T
        
        #######################
        self.nmb_prototypes = nmb_prototypes
        self.nmb_views = nmb_views
        self.world_size = world_size
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.views_for_assign = views_for_assign
        self.improve_numerical_stability = improve_numerical_stability
        print("=> viewsfa:", self.views_for_assign, "nmb views:", self.nmb_views)

        # create the encoder (including non-linear projection head: 2 FC layers)
        backbone, self.param = select_backbone(network)
        feature_size = self.param['feature_size']
        self.encoder_q = nn.Sequential(
                            backbone, 
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(dim, nmb_prototypes, bias=False) # Should be dim (D) x nmb_prototypes (C)
        
        self.softmax = nn.Softmax(dim=1).cuda() 
        # self.cos = nn.CosineSimilarity(dim=1)
        self.use_the_queue = False 
        self.start_queue = False
        self.queue = None
        self.batch_shuffle = None

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_head(self, x):
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, block):
        '''Output: logits, targets'''
        (B, N, *_) = block.shape # [B,N,C,T,H,W] e.g. [16, 2, 3, 32, 128, 128]
        assert N == 2
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()

        # compute features of x1
        s = self.encoder_q(x1)  # queries: B,C,1,1,1
        s = nn.functional.normalize(s, dim=1)
        s = s.view(B, self.dim) # To B, C e.g. 16, 128

        # compute features of x2
        if self.batch_shuffle:
            with torch.no_grad():
                # shuffle for making use of BN
                x2, idx_unshuffle = self._batch_shuffle_ddp(x2)
                
                t = self.encoder_q(x2)  # keys: B,C,1,1,1
                t = nn.functional.normalize(t, dim=1)

                # undo shuffle
                t = self._batch_unshuffle_ddp(t, idx_unshuffle)
        else:
            t = self.encoder_q(x2)  # keys: B,C,1,1,1
            t = nn.functional.normalize(t, dim=1)
            
        t = t.view(B, self.dim) # To B, C e.g. 16, 128

        ########### SWAV ######################
        # ============ multi-res forward passes ... ============
        # Embedding: Projhead(x): 2B x D. e.g. 32 x 128. Output: Prototype scores: Prototypes(Projhead(x)): 2B x C, e.g. 32 x 300
        embedding, output = self.forward_head(torch.cat((s, t))) # B, K # Positive examples
        embedding = embedding.detach() # Detach embedding: we dont need gradients, this is only used to fill the queue.
        bs = B

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.views_for_assign): # views for assign: [0, 1]
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)] # B x K, e.g. 16 x 300

                # time to use the queue
                if self.queue is not None:
                    if self.start_queue and (self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0)):
                        use_the_queue = True
                        # Queue size must be divisible by batch size, queue is Queue_Size x dim, e.g. 3480, 128.
                        # SWAV queue is a tensor of [N, L, Feat_dim], Coclr queue is a tensor of [Feat dim, L] 
                        # prototypes are dim x K, e.g. 128 x 300
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.prototypes.weight.t()
                        ), out)) # out is 16 x 300 (for current feature), cat this with 2048 x 300, prototype scores of queue
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q_swav = out / self.epsilon
                if self.improve_numerical_stability:
                    M = torch.max(q_swav)
                    dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q_swav -= M
                q_swav = torch.exp(q_swav).t()
                q_swav = self.distributed_sinkhorn(q_swav, self.sinkhorn_iterations)[-bs:]
                # q_swav are now soft assignments B x C e.g. 16 x 300

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_views)), crop_id): # Use crop ids except current crop id 
                p_swav = self.softmax(output[bs * v: bs * (v + 1)] / self.T) # B x 300
                
                # swap prediction problem
                subloss -= torch.mean(torch.sum(q_swav * torch.log(p_swav), dim=1))
            loss += subloss / (np.sum(self.nmb_views) - 1)
        loss /= len(self.views_for_assign)
        
        return embedding, output, loss

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            Q = self.shoot_infs(Q)
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.world_size * Q.shape[1])
            for it in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                dist.all_reduce(u)
                u = r / u
                u = self.shoot_infs(u)
                Q *= u.unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


    def shoot_infs(self, inp_tensor):
        """Replaces inf by maximum of tensor"""
        mask_inf = torch.isinf(inp_tensor)
        ind_inf = torch.nonzero(mask_inf)
        if len(ind_inf) > 0:
            for ind in ind_inf:
                if len(ind) == 2:
                    inp_tensor[ind[0], ind[1]] = 0
                elif len(ind) == 1:
                    inp_tensor[ind[0]] = 0
            m = torch.max(inp_tensor)
            for ind in ind_inf:
                if len(ind) == 2:
                    inp_tensor[ind[0], ind[1]] = m
                elif len(ind) == 1:
                    inp_tensor[ind[0]] = m
        return inp_tensor


class InfoNCE(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(InfoNCE, self).__init__()

        self.dim = dim 
        self.K = K
        self.m = m
        self.T = T

        # create the encoders (including non-linear projection head: 2 FC layers)
        backbone, self.param = select_backbone(network)
        feature_size = self.param['feature_size']
        self.encoder_q = nn.Sequential(
                            backbone, 
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        backbone, _ = select_backbone(network)
        self.encoder_k = nn.Sequential(
                            backbone, 
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Notes: for handling sibling videos, e.g. for UCF101 dataset


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, block):
        '''Output: logits, targets'''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]
        assert N == 2
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k)

        return logits, labels


class UberNCE(InfoNCE):
    '''
    UberNCE is a supervised version of InfoNCE,
    it uses labels to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(UberNCE, self).__init__(network, dim, K, m, T)
        # extra queue to store label
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, block, k_label):
        '''Output: logits, binary mask for positive pairs
        '''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]
        assert N == 2
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        mask = k_label.unsqueeze(1) == self.queue_label.unsqueeze(0) # B,K
        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1) # B,(1+K)
                
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, k_label)

        return logits, mask

class ViCC2(ViCC):
    '''
    CoCLR: using another view of the data to define positives https://github.com/TengdaHan/CoCLR
    With SwaV clustering and without momentum encoder https://arxiv.org/abs/2006.09882 
    '''
    def __init__(self, 
                network='s3d', 
                dim=128, 
                K=1920, 
                m=0.999, 
                T=0.1, 
                #################
                nmb_prototypes=0,
                nmb_views=0,
                world_size=0,
                epsilon=0.05,
                sinkhorn_iterations=3,
                views_for_assign=0,
                improve_numerical_stability=False,
                #################
                reverse=False,
                predict_only_mod=False):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048 CoCLR / 3840 SwaV / 1920 ViCC)
        m: moco momentum of updating key encoder (default: 0.999), only used for MoCo/CoCLR
        T: softmax temperature (default: 0.07 CoCLR / 0.1 Swav,ViCC)

        nmb_prototypes (C) (default: 300)
        nmb_views: amount of views used in a list, e.g. [2] or [2,2]
        epsilon: regularization parameter for Sinkhorn-Knopp algorithm
        sinkhorn_iterations: number of iterations in Sinkhorn-Knopp algorithm
        views_for_assign: list of views id used for computing assignments
        improve_numerical_stability: improves numerical stability in Sinkhorn-Knopp algorithm

        reverse: if true, we optimize flow instead of rgb
        '''
        super(ViCC2, self).__init__(network, dim, K, m, T, nmb_prototypes, nmb_views, world_size, epsilon, 
        sinkhorn_iterations, views_for_assign, improve_numerical_stability)

        # create another encoder, for the second view of the data 
        backbone, _ = select_backbone(network)
        feature_size = self.param['feature_size']
        self.second = nn.Sequential(
                            backbone,
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))
        for param_s in self.second.parameters():
            param_s.requires_grad = False  # not update by gradient

        self.reverse = reverse
        self.predict_only_mod = predict_only_mod 

    def forward(self, block1, block2, k_vsource):
        '''Output: logits, targets'''
        (B, N, *_) = block1.shape # B,N,C,T,H,W e.g. [16, 2, 3, 32, 128, 128] 
        assert N == 2
        x1 = block1[:,0,:].contiguous()
        f1 = block1[:,1,:].contiguous() # flow
        x2 = block2[:,0,:].contiguous()
        f2 = block2[:,1,:].contiguous() # flow

        if self.reverse: # reverse means main feature becomes flow instead of rgb
            x1, f1 = f1, x1
            x2, f2 = f2, x2

        # compute features of x1
        s = self.encoder_q(x1)  # queries: B,C,1,1,1
        s = nn.functional.normalize(s, dim=1)
        s = s.view(B, self.dim) # To B, C e.g. 16, 128

        # compute features of x2
        if self.batch_shuffle:
            with torch.no_grad():  
                # shuffle for making use of BN 
                x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

                t = self.encoder_q(x2)  # keys: B,C,1,1,1
                t = nn.functional.normalize(t, dim=1)

                # undo shuffle
                t = self._batch_unshuffle_ddp(t, idx_unshuffle)
        else:
            t = self.encoder_q(x2)  # keys: B,C,1,1,1
            t = nn.functional.normalize(t, dim=1)

        t = t.view(B, self.dim) # To B, C e.g. 16, 128

        with torch.no_grad():
            # compute feature for second view
            sf = self.second(f2) # keys: B,C,1,1,1
            sf = nn.functional.normalize(sf, dim=1)
            sf = sf.view(B, self.dim)

            # compute feature for second view
            tf = self.second(f1) # keys: B,C,1,1,1
            tf = nn.functional.normalize(tf, dim=1)
            tf = tf.view(B, self.dim)

        ########### SWAV ######################
        # ============ multi-res forward passes ... ============
        # Embedding: Projhead(x): 2B x D. e.g. 32 x 128. Output: Prototype scores: Prototypes(Projhead(x)): 2B x C, e.g. 32 x 300
        embedding, output = self.forward_head(torch.cat((s, t)))
        embeddingf, outputf = self.forward_head(torch.cat((sf, tf))) # augmentations of second view through main prototypes
        embedding = embedding.detach() # Detach embedding: we dont need gradients, this is only used to fill the queue.
        embeddingf = embeddingf.detach()

        output_both = torch.cat((output, outputf))
        embedding_both = torch.cat((embedding, embeddingf))
        bs = B

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.views_for_assign): # e.g. [0, 1] (use main space) or [0,1,2,3] (use other feature space for q computation)
            with torch.no_grad():
                out = output_both[bs * crop_id: bs * (crop_id + 1)] # B x K, e.g. 16 x 300
                
                # time to use the queue
                if self.queue is not None:
                    if self.start_queue and (self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0)): # queue full
                        use_the_queue = True
                        # Queue size must be divisible by batch size, queue is Queue_Size x dim, e.g. 3480, 128.
                        # SWAV queue is a tensor of [N, L, Feat_dim], Coclr queue is a tensor of [Feat dim, L] 
                        # prototypes are dim x K, e.g. 128 x 300
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.prototypes.weight.t()
                        ), out)) # out is 16 x 300 (for current feature), cat this with 2048 x 300, prototype scores of queue
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding_both[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q_swav = out / self.epsilon
                if self.improve_numerical_stability:
                    M = torch.max(q_swav)
                    dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q_swav -= M
                q_swav = torch.exp(q_swav).t()
                q_swav = self.distributed_sinkhorn(q_swav, self.sinkhorn_iterations)[-bs:]
                # q_swav are now soft assignments B x C e.g. 16 x 300

            # cluster assignment prediction 
            ################# Use output of both here with nmb_views [2, 2] for q prediction #################
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_views)), crop_id): # Use crop ids except current crop id 
                p_swav = self.softmax(output_both[bs * v: bs * (v + 1)] / self.T) # B x 300

                # swap prediction problem
                subloss -= torch.mean(torch.sum(q_swav * torch.log(p_swav), dim=1))
            
            loss += subloss / (np.sum(self.nmb_views) - 1)

        loss /= len(self.views_for_assign)

        return embedding_both, output_both, loss
        
class CoCLR(InfoNCE):
    '''
    CoCLR: using another view of the data to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, 
                network='s3d', 
                dim=128, 
                K=2048, 
                m=0.999, 
                T=0.07, 
                topk=5, 
                reverse=False):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(CoCLR, self).__init__(network, dim, K, m, T)

        self.topk = topk

        # create another encoder, for the second view of the data 
        backbone, _ = select_backbone(network)
        feature_size = self.param['feature_size']
        self.sampler = nn.Sequential(
                            backbone,
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))
        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", torch.randn(dim, K))
        self.queue_second = nn.functional.normalize(self.queue_second, dim=0)
        
        # for handling sibling videos, e.g. for UCF101 dataset
        self.register_buffer("queue_vname", torch.ones(K, dtype=torch.long) * -1) 
        # for monitoring purpose only
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)
        
        self.queue_is_full = False
        self.reverse = reverse 

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second, vnames):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        keys_second = concat_all_gather(keys_second)
        vnames = concat_all_gather(vnames)
        # labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        self.queue_vname[ptr:ptr + batch_size] = vnames
        self.queue_label[ptr:ptr + batch_size] = torch.ones_like(vnames)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, block1, block2, k_vsource):
        '''Output: logits, targets'''
        (B, N, *_) = block1.shape # B,N,C,T,H,W
        assert N == 2
        x1 = block1[:,0,:].contiguous()
        f1 = block1[:,1,:].contiguous()
        x2 = block2[:,0,:].contiguous()
        f2 = block2[:,1,:].contiguous()

        if self.reverse:
            x1, f1 = f1, x1
            x2, f2 = f2, x2 

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = k.view(B, self.dim)

            # compute key feature for second view
            kf = self.sampler(f2) # keys: B,C,1,1,1
            kf = nn.functional.normalize(kf, dim=1)
            kf = kf.view(B, self.dim)

        # if queue_second is full: compute mask & train CoCLR, else: train InfoNCE

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        # handle sibling videos, e.g. for UCF101. It has no effect on K400
        mask_source = k_vsource.unsqueeze(1) == self.queue_vname.unsqueeze(0) # B,K
        mask = mask_source.clone()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_label != -1)
            
        if self.queue_is_full: 
            print('\n===== queue is full now =====')

        if self.queue_is_full and (self.topk != 0):
            mask_sim = kf.matmul(self.queue_second.clone().detach())
            mask_sim[mask_source] = - np.inf # mask out self (and sibling videos)
            _, topkidx = torch.topk(mask_sim, self.topk, dim=1)
            topk_onehot = torch.zeros_like(mask_sim)
            topk_onehot.scatter_(1, topkidx, 1)
            mask[topk_onehot.bool()] = True

        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1)

        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, kf, k_vsource)

        return logits, mask.detach()


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out
