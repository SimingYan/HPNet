import os
import random
import importlib
from scipy.sparse import csc_matrix
import scipy.sparse as sparse
import scipy
import numpy as np
import open3d as o3d
from functools import wraps
import time
from scipy.optimize import linear_sum_assignment
import torch
from torch.autograd import Variable
from lapsolver import solve_dense

DIVISION_EPS = 1e-10


def parameter_count(model):
    print('parameters number:',
          sum(param.numel() for param in model.parameters())/1e6, ' M')


def cuda_time():
    torch.cuda.synchronize()
    return time.time()


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(), volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var), volatile=volatile)
    if cuda:
        res = res.cuda()
    return res


def npy(var):
    return var.data.cpu().numpy()


def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module(model_version)


def write_ply(fn, point, normal=None, color=None):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)

    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)

    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)

    o3d.io.write_point_cloud(fn, ply)

    return


def write_xyz_files(output_path, point, normal=None):

    fout = open(output_path, "w")

    if normal is not None:
        for i in range(point.shape[0]):
            fout.write("%f %f %f %f %f %f\n" %
                       (point[i][0], point[i][1], point[i][2], normal[i][0],
                        normal[i][1], normal[i][2]))
    else:
        for i in range(point.shape[0]):
            fout.write("%f %f %f\n" % (point[i][0], point[i][1], point[i][2]))

    fout.close()

    return


def read_xyz_files(filename, normal=True):
    with open(filename, 'r') as f:
        lines = f.readlines()

        num_points = len(lines)
        pc_pos = []
        pc_norm = []
        i = 0
        for line in lines:
            line = line.split()
            line = [float(i) for i in line]
            pc_pos.append(line[:3])
            if normal:
                pc_norm.append(line[3:6])

    pc_pos = np.array(pc_pos)
    pc_norm = np.array(pc_norm)

    if normal:
        return pc_pos, pc_norm

    return pc_pos


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def pcloud_line(prev, cur, color=None):
    alpha = np.linspace(0, 1, 100)
    pcloud = prev[None, :] + alpha[:, None] * (cur - prev)[None, :]

    if color is None:
        pcolor = np.tile(np.array([0, 1, 0])[None, :], [pcloud.shape[0], 1])
    else:
        assert (len(color) == 3)
        pcolor = np.tile(np.array(color)[None, :], [pcloud.shape[0], 1])

    return pcloud, pcolor


def ComputeBasis(nor):
    signN = 1 if nor[2] > 0 else -1
    a = -1 / (signN + nor[2])
    b = nor[0] * nor[1] * a
    b1 = np.array(
        [1 + signN * nor[0] * nor[0] * a, signN * b, -signN * nor[0]])
    b2 = np.array([b, signN + nor[1] * nor[1] * a, -nor[1]])
    return b1, b2


def draw_plane_bbox(plane_c, plane_n, filename):

    b1, b2 = ComputeBasis(plane_n)

    verts = []
    verts_c = []

    bbox_size = 60

    corners = [
        plane_c + bbox_size * b1 + bbox_size * b2,
        plane_c + bbox_size * b1 - bbox_size * b2,
        plane_c - bbox_size * b1 - bbox_size * b2,
        plane_c - bbox_size * b1 + bbox_size * b2
    ]

    vv, v_c = pcloud_line(corners[0], corners[1])

    verts.append(vv)
    verts_c.append(v_c)

    vv, v_c = pcloud_line(corners[1], corners[2])
    verts.append(vv)
    verts_c.append(v_c)

    vv, v_c = pcloud_line(corners[2], corners[3])
    verts.append(vv)
    verts_c.append(v_c)

    vv, v_c = pcloud_line(corners[3], corners[0])
    verts.append(vv)
    verts_c.append(v_c)

    verts = np.concatenate(verts)
    verts_c = np.concatenate(verts_c)

    write_ply(filename, verts, color=verts_c)

    return verts, verts_c

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets, dtype=np.int8).reshape(-1)]
    # check none-type
    if targets.min() == -1:
        idx = np.argwhere(targets == -1)
        res[idx] = 0
    return res.reshape(list(targets.shape) + [nb_classes])


def hungarian_matching(W_pred, I_gt):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - BxNxK
    # I_gt - BxN, may contain -1's
    # Output: matching_indices - BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
    #   where only n_gt_labels entries on each row have meaning. The matching does not include gt background instance
    # calculate RIoU
    batch_size = I_gt.shape[0]
    n_points = I_gt.shape[1]
    n_max_labels = W_pred.shape[2]

    matching_indices = np.zeros([batch_size, n_max_labels], dtype=np.int32)
    for b in range(batch_size):
        # assuming I_gt[b] does not have gap
        n_gt_labels = np.max(I_gt[b]) + 1  # this is K'
        W_gt = np.zeros([n_points, n_gt_labels + 1
                         ])  # HACK: add an extra column to contain -1's
        W_gt[np.arange(n_points), I_gt[b]] = 1.0  # NxK'

        dot = np.sum(np.expand_dims(W_gt, axis=2) *
                     np.expand_dims(W_pred[b], axis=1),
                     axis=0)  # K'xK
        denominator = np.expand_dims(np.sum(
            W_gt, axis=0), axis=1) + np.expand_dims(np.sum(W_pred[b], axis=0),
                                                    axis=0) - dot
        cost = dot / np.maximum(denominator, DIVISION_EPS)  # K'xK
        cost = cost[:
                    n_gt_labels, :]  # remove last row, corresponding to matching gt background instance

        _, col_ind = linear_sum_assignment(-cost)  # want max solution
        matching_indices[b, :n_gt_labels] = col_ind

    return matching_indices

def to_one_hot(target, maxx=50, device_id=0):
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target.astype(np.int64)).cuda(device_id)
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))

    target_one_hot = target_one_hot.cuda(device_id)
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot

def relaxed_iou_fast(pred, gt, max_clusters=50):
    batch_size, N, K = pred.shape
    normalize = torch.nn.functional.normalize
    one = torch.ones(1).cuda()

    norms_p = torch.unsqueeze(torch.sum(pred, 1), 2)
    norms_g = torch.unsqueeze(torch.sum(gt, 1), 1)
    cost = []

    for b in range(batch_size):
        p = pred[b]
        g = gt[b]
        c_batch = []
        dots = p.transpose(1, 0) @ g
        r_iou = dots
        r_iou = r_iou / (norms_p[b] + norms_g[b] - dots + 1e-7)
        cost.append(r_iou)
    cost = torch.stack(cost, 0)
    return cost

def match(target, pred_labels):
    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(pred_labels)

    # cost = relaxed_iou(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
    # cost_ = 1.0 - torch.as_tensor(cost)
    cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())

    # cost_ = 1.0 - torch.as_tensor(cost)
    cost_ = 1.0 - cost.data.cpu().numpy()
    rids, cids = solve_dense(cost_[0])

    unique_target = np.unique(target)
    unique_pred = np.unique(pred_labels)
    return rids, cids, unique_target, unique_pred

def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)

