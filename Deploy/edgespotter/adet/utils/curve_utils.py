# borrow from https://github.com/voldemortX/pytorch-auto-drive/blob/master/utils/curve_utils.py

import torch

def upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    # https://github.com/pytorch/vision/pull/3383
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

class CatromSampler(torch.nn.Module):
    # Fast Batch Catmull-Rom sampler
    def __init__(self, num_sample_points):
        super().__init__()
        self.num_control_points = 4
        self.num_sample_points = num_sample_points
        self.control_points = []

        t_values = torch.linspace(0, 1, self.num_sample_points)  # 生成 t 值
        self.basis_matrix = torch.stack([self.catmull_rom_basis(t) for t in t_values], dim=0)  # 形状 (num_samples, 4)

    def catmull_rom_basis(self, t):
        t2 = t ** 2
        t3 = t ** 3
        return torch.stack([
            -0.5 * t3 + t2 - 0.5 * t,  
            1.5 * t3 - 2.5 * t2 + 1.0, 
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,  
            0.5 * t3 - 0.5 * t2 
        ], dim=-1)
    

    def get_sample_points(self, control_points_matrix):
        if control_points_matrix.numel() == 0:
            return control_points_matrix  # Looks better than a torch.Tensor
        if self.basis_matrix.device != control_points_matrix.device:
            self.basis_matrix = self.basis_matrix.to(control_points_matrix.device)

        return upcast(self.basis_matrix).matmul(upcast(control_points_matrix))

