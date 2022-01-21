import numpy as np
import random
import os
import h5py
import argparse
from fitting_func import *

def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U

def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='/path/to/parsenet-codebase/data/shapes')
parser.add_argument('--save_path', type=str, default='/path/to/saved/dir')

args = parser.parse_args()

data_path = os.path.join(args.data_path, 'train_data.h5')

with h5py.File(data_path, 'r') as hf:
    gt_points = np.array(hf.get("points")).astype(np.float64)
    gt_labels = np.array(hf.get("labels"))
    gt_normals = np.array(hf.get("normals")).astype(np.float64)
    gt_primitives = np.array(hf.get("prim"))
    
    means = np.mean(gt_points, 1)
    means = np.expand_dims(means, 1)
    new_gt_points = (gt_points - means)


len_ = len(gt_points)

for i in range(len_):
    print(i) 

    filename = '%05d.h5' % i

    P = new_gt_points[i]

    # align
    S, U = pca_numpy(P)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # rotate input points such that the minor principal
    # axis aligns with x axis.
    P = (R @ P.T).T

    # normalize
    std = np.max(P, 0) - np.min(P, 0)
    P = P / (np.max(std) + EPS)
    
    I_gt = gt_labels[i]
    normal_gt = gt_normals[i]
    normal_gt = (R @ normal_gt.T).T
    
    n_instances = I_gt.max() + 1

    instances = []
    T_gt = gt_primitives[i]
   
    primitive_param = np.zeros([gt_points.shape[1], 22])
    # render shape parameters
    # 0:sphere 4dim, 1:plane 4dim, 2:cylinder 7dim, 3:cone 7dim
    # 4 + 4 + 7 + 7 = 22

    for j in range(n_instances):
        indx = I_gt == j
        if indx.sum() == 0:
            continue
        
        primitive_type = T_gt[indx][0]
        
        if (T_gt[indx] != primitive_type).sum() != 0:
            import ipdb; ipdb.set_trace()

        points =  P[indx]
        normals = normal_gt[indx]
        weights = np.ones([points.shape[0], 1])

        N = points.shape[0]

        if points.shape[0] < 100: # skip small instance
            continue

        points = torch.from_numpy(points)
        normals = torch.from_numpy(normals)
        weights = torch.from_numpy(weights)
        
        if primitive_type in [0, 2, 6, 7, 8, 9]: # didn't process spline
            continue

        if primitive_type == 5: # fitting sphere
            center, radius = fit_sphere_torch(points, normals, weights)
            if radius > 10:
                print(filename + ': sphere radius:' + str(radius.cpu().numpy()) + 'point number:' + str(N))
                continue

            center = np.tile(center.cpu().numpy(), [N, 1])
            radius = np.tile(radius.cpu().numpy(), N)

            primitive_param[indx, :3] = center
            primitive_param[indx, 3] = radius

        elif primitive_type == 1: # fitting plane
            a, d = fit_plane_torch(points, normals, weights)
            
            a = np.tile(a.cpu().numpy(), [N, 1])
            d = np.tile(d.cpu().numpy(), N)

            primitive_param[indx, 4:7] = a
            primitive_param[indx, 7] = d


        elif primitive_type == 4: # fitting cylinder
            try:
                a, center, radius = fit_cylinder_torch(points, normals, weights)
            except:
                print('raise error', 'point number:' + str(N) + '\n')
                continue

            if radius > 10 or center[0][0] > 10 or center[0][1] > 10 or center[0][2] > 10:
                print(filename + ': cylinder radius:' + str(radius) +'point number:' + str(N))
                continue
            
            a = np.tile(a.T.cpu().numpy(), [N, 1])
            center = np.tile(center, [N, 1])
            radius = np.tile(radius, N)

            primitive_param[indx, 8:11] = a
            primitive_param[indx, 11:14] = center
            primitive_param[indx, 14] = radius


        elif primitive_type == 3:
            center, a, theta = fit_cone_torch(points, normals, weights)
            if center[0][0] > 10 or center[1][0] > 10 or center[2][0] > 10:
                tmp = center.cpu().numpy()
                print(filename + ': cone center:' + str(tmp[0][0]) + ' '+ str(tmp[1][0]) + ' ' + str(tmp[2][0]) +'point number:' + str(N))
                continue
            
            a = np.tile(a.cpu().numpy(), [N, 1])
            center = np.tile(center.T.cpu().numpy(), [N, 1])
            theta = np.tile(theta.cpu().numpy(), N)

            primitive_param[indx, 15:18] = a
            primitive_param[indx, 18:21] = center
            primitive_param[indx, 21] = theta
    
    wf = h5py.File(os.path.join(args.save_path, filename), 'w')
    wf.create_dataset('labels', data=I_gt)
    wf.create_dataset('prim', data=T_gt)
    wf.create_dataset('points', data=P)
    wf.create_dataset('normals',  data=normal_gt)
    wf.create_dataset('T_param',  data=primitive_param)
