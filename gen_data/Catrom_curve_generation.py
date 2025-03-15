import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from shapely.geometry import Polygon, Point
import glob
import os

class CatmullRom(nn.Module):
    def __init__(self, ps, ctps):
        super(CatmullRom, self).__init__()
        self.ctps = nn.Parameter(torch.as_tensor(ctps, dtype=torch.float64))
        self.ps = torch.as_tensor(ps, dtype=torch.float64)  
        self.t = torch.as_tensor(np.linspace(0, 1, 81), dtype=torch.float64)  

    def forward(self):
        catmull_rom_points = self.catmull_rom_interpolate(self.ctps, self.t)
        
        diffs = catmull_rom_points.unsqueeze(0) - self.ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        
        return min_dists.sum()

    def catmull_rom_interpolate(self, ctps, t):
        interpolated_points = []
        p0, p1, p2, p3 = ctps 
        
        for ti in t:
            ti2 = ti * ti
            ti3 = ti2 * ti
            a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
            b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
            c = -0.5 * p0 + 0.5 * p2
            d = p1
            point = a * ti3 + b * ti2 + c * ti + d
            interpolated_points.append(point)
        
        return torch.stack(interpolated_points)

def train(x, y, ctps):
    x, y = np.array(x), np.array(y)
    ps = np.vstack((x, y)).transpose()
    catmull_rom = CatmullRom(ps, ctps)

    return catmull_rom.ctps.detach().numpy()

def catmull_rom_fit(x, y):

    n = len(x)

    t = np.linspace(0, 1, n)

    coeff_matrix = np.array([catmull_rom_coeff(ti) for ti in t])  

    pseudoinverse = np.linalg.pinv(coeff_matrix)  

    data = np.column_stack((x, y))  
    control_points = pseudoinverse.dot(data) 

    return control_points
def catmull_rom_coeff(t):

    t2 = t * t
    t3 = t2 * t
    return np.array([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2
    ])

def adjust_control_points(control_points, scale_factor=1):

    P0, P1, P2, P3 = control_points

    M = (P1 + P2) / 2

    v0 = P0 - M
    v3 = P3 - M

    v0_prime = scale_factor * v0
    v3_prime = scale_factor * v3

    P0_prime = M + v0_prime
    P3_prime = M + v3_prime

    return np.array([P0_prime, P1, P2, P3_prime])


Basis_matrix = lambda ts: [catmull_rom_basis(t) for t in ts]

def catmull_rom_basis(t):
    t2 = t ** 2
    t3 = t ** 3
    return np.stack([
        -0.5 * t3 + t2 - 0.5 * t,  
        1.5 * t3 - 2.5 * t2 + 1.0, 
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,  
        0.5 * t3 - 0.5 * t2 
    ])

def process_file(label, target_folder, vis_folder):
    imgdir = label.replace('IPM2025_train\\gt_ctw1500\\', 'IPM2025_train\\images\\').replace('.txt', '.jpg')
    txt_file = label.replace('IPM2025_train\\gt_ctw1500\\', target_folder)
    dist_file = os.path.join(txt_file)

    data = []
    cts = []
    polys = []
    fin = open(label, 'r').readlines()
    for line in fin:
        line = line.strip().split('####')
        ct = line[-1]
        if ct == '#': continue
        coords = [(float(line[0].strip().split(',')[:-1][ix]), float(line[0].strip().split(',')[:-1][ix+1])) for ix in range(0, len(line[0].strip().split(',')[:-1]), 2)]
        poly = Polygon(coords)
        data.append(np.array([float(x) for x in line[0].strip().split(',')[:-1]]))
        cts.append(ct)
        polys.append(poly)

    img = plt.imread(imgdir)
    with open(dist_file, 'w', encoding='utf-8') as file:
        for iid, ddata in enumerate(data):
            lh = len(data[iid])
            assert(lh % 4 == 0)
            lhc2 = int(lh/2)
            lhc4 = int(lh/4)

            curve_data_top = data[iid][0:lhc2].reshape(lhc4, 2)
            curve_data_bottom = data[iid][lhc2:].reshape(lhc4, 2)

            x_data = curve_data_top[:, 0]
            y_data = curve_data_top[:, 1]

            init_ctps = catmull_rom_fit(x_data, y_data)
            control_points = train(x_data, y_data, init_ctps)
            control_points = adjust_control_points(control_points)

            x_data_b = curve_data_bottom[:, 0]
            y_data_b = curve_data_bottom[:, 1]

            init_ctps_b = catmull_rom_fit(x_data_b, y_data_b)

            control_points_b = train(x_data_b, y_data_b, init_ctps_b)
            control_points_b = adjust_control_points(control_points_b)

            outstr = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}####{}\n'.format(
                control_points[0, 0], control_points[0, 1],
                control_points[1, 0], control_points[1, 1],
                control_points[2, 0], control_points[2, 1],
                control_points[3, 0], control_points[3, 1],
                control_points_b[0, 0], control_points_b[0, 1],
                control_points_b[1, 0], control_points_b[1, 1],
                control_points_b[2, 0], control_points_b[2, 1],
                control_points_b[3, 0], control_points_b[3, 1],
                cts[iid]
            )
            file.write(outstr)

            t_plot = np.linspace(0, 1, 81)
            CatmullRom_top = np.array(Basis_matrix(t_plot)).dot(control_points)

            CatmullRom_bottom = np.array(Basis_matrix(t_plot)).dot(control_points_b)

            plt.plot(CatmullRom_top[:, 0], CatmullRom_top[:, 1], 'g-', label='fit', linewidth=1.0)
            plt.plot(CatmullRom_bottom[:, 0], CatmullRom_bottom[:, 1], 'g-', label='fit', linewidth=1.0)
            plt.plot(control_points[:, 0], control_points[:, 1], 'r.:', label='fit', linewidth=1.0)
            plt.plot(control_points_b[:, 0], control_points_b[:, 1], 'r.:', label='fit', linewidth=1.0)

        os.makedirs(vis_folder, exist_ok=True)
        vis_file = os.path.join(vis_folder, f'{os.path.basename(imgdir)}.png')
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(vis_file, bbox_inches='tight', dpi=100)
        plt.close()

def main():
    labels = glob.glob('IPM2025_train\\gt_ctw1500\\*.txt')
    labels.sort()
    target_folder = 'IPM2025_train\\gt_catmull_rom\\'
    vis_folder = 'IPM2025_train\\vis_results\\'
    os.makedirs(target_folder, exist_ok=True)
    for label in labels:
        process_file(label, target_folder, vis_folder)

if __name__ == '__main__':
    main()