import torch

class GaLoreProjector:
    def __init__(self, rank = 128, update_proj_gap=None):
        self.rank = rank
        if update_proj_gap is None:
            update_proj_gap = int(32 * rank / 256) # assert rank >= 8
        self.update_proj_gap = update_proj_gap
        self.lowrank_svd = None
        self.scale = None

    def project(self, param, iter):
        if self.lowrank_svd is None or iter % self.update_proj_gap == 0:
            self.lowrank_svd = self.get_lowrank_svd(param, self.rank)
        low_rank_grad = torch.matmul(self.lowrank_svd[0].t().to(param.grad.device.type), param.grad) @ self.lowrank_svd[2].to(param.grad.device.type) / self.lowrank_svd[1].to(param.grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        restored_grad = torch.matmul(self.lowrank_svd[0].to(low_rank_grad.device.type), low_rank_grad * self.lowrank_svd[1].to(low_rank_grad.device.type)) @ self.lowrank_svd[2].t().to(low_rank_grad.device.type)
        return restored_grad * self.scale


    # svd decomposition
    def get_lowrank_svd(self, weights, rank):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, V = torch.svd_lowrank(matrix, q=rank, niter=2)
        if not float_data:
            U = U.to(original_device).type(original_type)
            s = s.to(original_device).type(original_type)
            V = V.to(original_device).type(original_type)

        if not self.scale:
            dim = min(module_params.data.shape[0], module_params.data.shape[1])
            from math import log2
            self.scale = dim * log2(dim) / (rank * log2(rank))
            # from math import sqrt
            # self.scale = dim * sqrt(dim) / (rank * sqrt(rank))

        return U, s, V