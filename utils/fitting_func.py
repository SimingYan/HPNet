import numpy as np
import torch
from torch.autograd import Function
import open3d as o3d
from fit_circle import fit_circle_numpy

read_point_cloud = o3d.io.read_point_cloud
write_point_cloud = o3d.io.write_point_cloud
PointCloud = o3d.geometry.PointCloud
Vector3dVector = o3d.utility.Vector3dVector

def write_ply(fn, point, normal=None, color=None):
  ply = PointCloud()
  ply.points = Vector3dVector(point)
  if color is not None:
    ply.colors = Vector3dVector(color)
  if normal is not None:
    ply.normals = Vector3dVector(normal)
  write_point_cloud(fn, ply)


EPS = np.finfo(np.float32).eps

def best_lambda(A):
    """
    Takes an under determined system and small lambda value,
    and comes up with lambda that makes the matrix A + lambda I
    invertible. Assuming A to be square matrix.
    """
    lamb = 1e-6
    cols = A.shape[0]

    for i in range(7):
        A_dash = A + lamb * torch.eye(cols)
        if cols == torch.matrix_rank(A_dash):
            # we achieved the required rank
            break
        else:
            # factor by which to increase the lambda. Choosing 10 for performance.
            lamb *= 10
    return lamb

class LeastSquares:
    def __init__(self):
        pass

    def lstsq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        cols = A.shape[1]
        if np.isinf(A.data.cpu().numpy()).any():
            import ipdb;
            ipdb.set_trace()

        # Assuming A to be full column rank
        if cols == torch.matrix_rank(A):
            # Full column rank
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.transpose(1, 0) @ Y
        else:
            # rank(A) < n, do regularized least square.
            AtA = A.transpose(1, 0) @ A

            # get the smallest lambda that suits our purpose, so that error in
            # results minimized.
            with torch.no_grad():
                lamb = best_lambda(AtA)
            A_dash = AtA + lamb * torch.eye(cols)
            Y_dash = A.transpose(1, 0) @ Y

            # if it still doesn't work, just set the lamb to be very high value.
            x = self.lstsq(A_dash, Y_dash, 1)
        return x

LS = LeastSquares()
lstsq = LS.lstsq

class CustomSVD(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    future work.
    """

    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is voilated, the gradients
        # will be wrong.
        try:
            U, S, V = torch.svd(input, some=True)
        except:
            import ipdb;
            ipdb.set_trace()

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input


customsvd = CustomSVD.apply

def guard_sqrt(x, minimum=1e-5):
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)

def fit_plane_torch(points, normals, weights, show_warning=False):
    """
    Fits plane
    :param points: points with size N x 3
    :param weights: weights with size N x 1
    """
    weights_sum = torch.sum(weights) + EPS

    X = points - torch.sum(weights * points, 0).reshape((1, 3)) / weights_sum

    weighted_X = weights * X
    np_weighted_X = weighted_X.data.cpu().numpy()
    if np.linalg.cond(np_weighted_X) > 1e5:
        if show_warning:
            print("condition number is large in plane!", np.sum(np_weighted_X))
            print(torch.sum(points), torch.sum(weights))

    U, s, V = customsvd(weighted_X)
    a = V[:, -1]
    a = torch.reshape(a, (1, 3))
    d = torch.sum(weights * (a @ points.permute(1, 0)).permute(1, 0)) / weights_sum
    return a, d

def fit_sphere_numpy(points, normals, weights):
    dimension = points.shape[1]
    N = weights.shape[0]
    sum_weights = np.sum(weights)
    A = 2 * (- points + np.sum(points * weights, 0) / sum_weights)
    dot_points = np.sum(points * points, 1)
    normalization = np.sum(dot_points * weights) / sum_weights
    Y = dot_points - normalization
    Y = Y.reshape((N, 1))
    A = weights * A
    Y = weights * Y
    center = -np.linalg.lstsq(A, Y)[0].reshape((1, dimension))
    radius = np.sqrt(np.sum(weights[:, 0] * np.sum((points - center) ** 2, 1)) / sum_weights)
    return center, radius

def fit_sphere_torch(points, normals, weights, show_warning=False):

    N = weights.shape[0]
    sum_weights = torch.sum(weights) + EPS
    A = 2 * (- points + torch.sum(points * weights, 0) / sum_weights)

    dot_points = weights * torch.sum(points * points, 1, keepdim=True)

    normalization = torch.sum(dot_points) / sum_weights

    Y = dot_points - normalization
    Y = Y.reshape((N, 1))
    A = weights * A
    Y = weights * Y

    if np.linalg.cond(A.data.cpu().numpy()) > 1e8:
        if show_warning:
            print("condition number is large in sphere!")

    center = -lstsq(A, Y, 0.01).reshape((1, 3))
    radius_square = torch.sum(weights[:, 0] * torch.sum((points - center) ** 2, 1)) / sum_weights
    radius_square = torch.clamp(radius_square, min=1e-3)
    radius = guard_sqrt(radius_square)
    return center, radius

def fit_cylinder_numpy(points, normals, weights):
    _, s, V = np.linalg.svd(weights * normals, compute_uv=True)
    a = V.T[:, np.argmin(s)]
    a = np.reshape(a, (1, 3))

    # find the projection onto a plane perpendicular to the axis
    a = a.reshape((3, 1))
    a = a / (np.linalg.norm(a, ord=2) + EPS)

    prj_circle = points - ((points @ a).T * a).T
    center, radius = fit_sphere_numpy(prj_circle, normals, weights)
    return a, center, radius



def fit_cylinder_torch(points, normals, weights, show_warning=False):
    # compute
    # U, s, V = torch.svd(weights * normals)
    weighted_normals = weights * normals

    if np.linalg.cond(weighted_normals.data.cpu().numpy()) > 1e5:
        if show_warning:
            print("condition number is large in cylinder")
            print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())

    U, s, V = customsvd(weighted_normals)
    a = V[:, -1]
    a = torch.reshape(a, (1, 3))

    # find the projection onto a plane perpendicular to the axis
    a = a.reshape((3, 1))
    a = a / (torch.norm(a, 2) + EPS)

    prj_circle = points - ((points @ a).permute(1, 0) * a).permute(1, 0)
    # torch doesn't have least square for

    #center, radius = fit_sphere_torch(prj_circle, normals, weights)
    center, radius = fit_circle_numpy(prj_circle.data.cpu().numpy())
    if 0:
        colors = np.array([[1, 1, 0]])  # Cylinder, yellow
        colors = np.tile(colors, [points.shape[0], 1])
        write_ply('test_cylinder.ply', points.data.numpy(), normal=normals.data.numpy(), color=colors)
        write_ply('test_cylinder_circle.ply', prj_circle.data.numpy(), normal=normals.data.numpy(), color=colors)
        write_ply('test_circle_center.ply', center1, color=colors[0:1])
    
    return a, center, radius

def fit_cone_torch(points, normals, weights, show_warning=False):
    """ Need to incorporate the cholesky decomposition based
    least square fitting because it is stable and faster."""

    N = points.shape[0]
    A = weights * normals
    Y = torch.sum(normals * points, 1).reshape((N, 1))
    Y = weights * Y

    # if condition number is too large, return a very zero cone.
    if np.linalg.cond(A.data.cpu().numpy()) > 1e5:
        if show_warning:
            print("condition number is large, cone")
            print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())
        return torch.zeros((3, 1)), torch.Tensor([[1.0, 0.0, 0.0]]), torch.zeros(1)

    c = lstsq(A, Y, lamb=1e-3)

    a, _ = fit_plane_torch(normals, None, weights)
    if torch.sum(normals @ a.transpose(1, 0)) > 0:
        # we want normals to be pointing outside and axis to
        # be pointing inside the cone.
        a = - 1 * a

    diff = points - c.transpose(1, 0)
    diff = torch.nn.functional.normalize(diff, p=2, dim=1)
    diff = diff @ a.transpose(1, 0)

    # This is done to avoid the numerical issue when diff = 1 or -1
    # the derivative of acos becomes inf
    diff = torch.abs(diff)
    diff = torch.clamp(diff, max=0.999)
    theta = torch.sum(weights * torch.acos(diff)) / (torch.sum(weights) + EPS)
    theta = torch.clamp(theta, min=1e-3, max=3.142 / 2 - 1e-3)
    return c, a, theta


