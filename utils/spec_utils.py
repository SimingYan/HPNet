import torch
import numpy as np
from src.approximation import fit_bezier_surface_fit_kronecker, BSpline, uniform_knot_bspline_
from src.curve_utils import DrawSurfs
from lapsolver import solve_dense

draw_surf = DrawSurfs()
EPS = np.finfo(np.float32).eps
torch.manual_seed(2)
np.random.seed(2)
draw_surf = DrawSurfs()
regular_parameters = draw_surf.regular_parameterization(30, 30)

def up_sample_points_torch(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    Takes two neareast neighbors and finds the centroid
    and that becomes the new point.
    :param points: N x 3
    """
    for t in range(times):
        dist = torch.unsqueeze(points, 1) - torch.unsqueeze(points, 0)
        dist = torch.sum(dist ** 2, 2)
        _, indices = torch.topk(dist, 5, 1, largest=False)
        neighbors = points[indices[:, 1:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points


def up_sample_points_torch_in_range(points, a_min, a_max):
    N = points.shape[0]
    if N > a_max:
        N = points.shape[0]
        L = np.random.choice(np.arange(N), a_max, replace=False)
        points = points[L]
        return points
    else:
        while True:
            points = up_sample_points_torch(points)
            if points.shape[0] >= a_max:
                break
    N = points.shape[0]
    L = np.random.choice(np.arange(N), a_max, replace=False)
    points = points[L]
    return points


def basis_function_one(degree, knot_vector, span, knot):
    """ Computes the value of a basis function for a single parameter.

    Implementation of Algorithm 2.4 from The NURBS Book by Piegl & Tiller.
    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis function, :math:`N_{i,p}`
    :rtype: float
    """
    # Special case at boundaries
    if (
            (span == 0 and knot == knot_vector[0])
            or (span == len(knot_vector) - degree - 2)
            and knot == knot_vector[len(knot_vector) - 1]
    ):
        return 1.0

    # Knot is outside of span range
    if knot < knot_vector[span] or knot >= knot_vector[span + degree + 1]:
        return 0.0

    N = [0.0 for _ in range(degree + span + 1)]

    # Initialize the zeroth degree basis functions
    for j in range(0, degree + 1):
        if knot_vector[span + j] <= knot < knot_vector[span + j + 1]:
            N[j] = 1.0

    # Computing triangular table of basis functions
    for k in range(1, degree + 1):
        # Detecting zeros saves computations
        saved = 0.0
        if N[0] != 0.0:
            saved = ((knot - knot_vector[span]) * N[0]) / (
                    knot_vector[span + k] - knot_vector[span]
            )

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[span + j + 1]
            Uright = knot_vector[span + j + k + 1]

            # Zero detection
            if N[j + 1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j + 1] / (Uright - Uleft)
                N[j] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp
    return N[0]


def uniform_knot_bspline(control_points_u, control_points_v, degree_u, degree_v, grid_size=30):
    """
    Returns uniform knots, given the number of control points in u and v directions and 
    their degrees.
    """
    u = v = np.arange(0., 1, 1 / grid_size)

    knots_u = [0.0] * degree_u + np.arange(0, 1.01, 1 / (control_points_u - degree_u)).tolist() + [1.0] * degree_u
    knots_v = [0.0] * degree_v + np.arange(0, 1.01, 1 / (control_points_v - degree_v)).tolist() + [1.0] * degree_v

    nu = []
    nu = np.zeros((u.shape[0], control_points_u))
    for i in range(u.shape[0]):
        for j in range(0, control_points_u):
            nu[i, j] = basis_function_one(degree_u, knots_u, j, u[i])

    nv = np.zeros((v.shape[0], control_points_v))
    for i in range(v.shape[0]):
        for j in range(0, control_points_v):
            nv[i, j] = basis_function_one(degree_v, knots_v, j, v[i])
    return nu, nv

def standardize_points_torch(points, weights):
    Points = []
    stds = []
    Rs = []
    means = []
    batch_size = points.shape[0]

    for i in range(batch_size):
        point, std, mean, R = standardize_point_torch(points[i], weights)

        Points.append(point)
        stds.append(std)
        means.append(mean)
        Rs.append(R)

    Points = torch.stack(Points, 0)
    return Points, stds, means, Rs

def pca_torch(X):
    # TODO 2Change this to do SVD, because it is stable and computationally
    # less intensive.
    covariance = torch.transpose(X, 1, 0) @ X
    S, U = torch.eig(covariance, eigenvectors=True)
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
    G = np.array([[cos, -sin, 0],
                  [sin, cos, 0],
                  [0, 0, 1]])
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R

def standardize_point_torch(point, weights):
    # TODO: not back propagation through rotation matrix and scaling yet.
    # Change this 0.8 to 0 to include all points.

    higher_indices = weights[:, 0] > 0.8

    # some heuristic
    if torch.sum(higher_indices) < 400:
        if weights.shape[0] >= 7500:
            _, higher_indices = torch.topk(weights[:, 0], weights.shape[0] // 4)
        else:
            _, higher_indices = torch.topk(weights[:, 0], weights.shape[0] // 2)

    weighted_points = point[higher_indices] * weights[higher_indices]

    # Note: gradients throught means, force the network to produce correct means.
    mean = torch.sum(weighted_points, 0) / (torch.sum(weights[higher_indices]) + EPS)

    point = point - mean

    # take only very confident points to compute PCA direction.
    S, U = pca_torch(point[higher_indices])
    smallest_ev = U[:, torch.min(S[:, 0], 0)[1]].data.cpu().numpy()

    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))

    # axis aligns with x axis.
    R = R.astype(np.float32)

    R = torch.from_numpy(R).cuda(point.get_device()).detach()

    point = R @ torch.transpose(point, 1, 0)
    point = torch.transpose(point, 1, 0)

    weighted_points = point[higher_indices] * weights[higher_indices]
    try:
        std = torch.abs(torch.max(weighted_points, 0)[0] - torch.min(weighted_points, 0)[0])
    except:
        import ipdb;
        ipdb.set_trace()
    std = std.reshape((1, 3)).detach()
    point = point / (std + EPS)
    return point, std, mean, R

def sample_points_from_control_points_(nu, nv, outputs, batch_size, input_size_u=20, input_size_v=20):
    batch_size = outputs.shape[0]
    grid_size = nu.shape[0]
    reconst_points = []
    outputs = outputs.reshape((batch_size, input_size_u, input_size_v, 3))
    for b in range(batch_size):
        point = []
        for i in range(3):
            # cloning because it is giving error in back ward pass.
            point.append(torch.matmul(torch.matmul(nu, outputs[b, :, :, i].clone()), torch.transpose(nv, 1, 0)))
        reconst_points.append(torch.stack(point, 2))
    reconst_points = torch.stack(reconst_points, 0)
    reconst_points = reconst_points.view(batch_size, grid_size ** 2, 3)
    return reconst_points

def optimize_open_spline_kronecker(reconstructed_points, input_points_, control_points, new_cp_size=10, new_degree=2,
                                   deform=False):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    from src.fitting_optimization import Arap
    bspline = BSpline()
    N = input_points_.shape[1]
    control_points = control_points[0].data.cpu().numpy()
    if deform:
        arap = Arap(30, 30)
        new_mesh = arap.deform(reconstructed_points[0].data.cpu().numpy(),
                               input_points_[0].data.cpu().numpy(), viz=False)
        reconstructed_points = torch.from_numpy(np.array(new_mesh.vertices)).cuda()
        reconstructed_points = torch.unsqueeze(reconstructed_points, 0)

    new_cp_size = new_cp_size
    new_degree = new_degree

    # Note that boundary parameterization is necessary for the fitting
    # otherwise you 
    parameters = draw_surf.boundary_parameterization(20)
    parameters = np.concatenate([np.random.random((1600 - parameters.shape[0], 2)), parameters], 0)

    _, _, ku, kv = uniform_knot_bspline_(20, 20, 3, 3, 2)
    spline_surf = bspline.create_geomdl_surface(control_points.reshape((20, 20, 3)),
                                                np.array(ku),
                                                np.array(kv),
                                                3, 3)

    # these are randomly sampled points on the surface of the predicted spline
    points = np.array(spline_surf.evaluate_list(parameters))

    input = up_sample_points_torch_in_range(input_points_[0], 1600, 2000)

    L = np.random.choice(np.arange(input.shape[0]), 1600, replace=False)
    input = input[L].data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(points, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]

    _, _, ku, kv = uniform_knot_bspline_(new_cp_size, new_cp_size, new_degree, new_degree, 2)

    NU = []
    NV = []
    for index in range(parameters.shape[0]):
        nu, nv = bspline.basis_functions(parameters[index], new_cp_size, new_cp_size, ku, kv, new_degree, new_degree)
        NU.append(nu)
        NV.append(nv)
    NU = np.concatenate(NU, 1).T
    NV = np.concatenate(NV, 1).T

    new_control_points = fit_bezier_surface_fit_kronecker(matched, NU, NV)
    new_spline_surf = bspline.create_geomdl_surface(new_control_points,
                                                    np.array(ku),
                                                    np.array(kv),
                                                    new_degree, new_degree)

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = new_spline_surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points




def optimize_close_spline_kronecker(reconstructed_points,
                                    input_points_,
                                    control_points,
                                    new_cp_size=10,
                                    new_degree=3,
                                    deform=True):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    if deform:
        from src.fitting_optimization import Arap
        arap = Arap()
        new_mesh = arap.deform(reconstructed_points[0].data.cpu().numpy(),
                               input_points_[0].data.cpu().numpy(), viz=False)
        reconstructed_points = torch.from_numpy(np.array(new_mesh.vertices)).cuda()
        reconstructed_points = torch.unsqueeze(reconstructed_points, 0)

    bspline = BSpline()
    N = input_points_.shape[1]
    control_points = control_points[0].data.cpu().numpy()

    new_cp_size = new_cp_size
    new_degree = new_degree

    # Note that boundary parameterization is necessary for the fitting
    parameters = draw_surf.boundary_parameterization(30)
    parameters = np.concatenate([np.random.random((1600 - parameters.shape[0], 2)), parameters], 0)

    _, _, ku, kv = uniform_knot_bspline_(21, 20, 3, 3, 2)

    spline_surf = bspline.create_geomdl_surface(control_points.reshape((21, 20, 3)),
                                                np.array(ku),
                                                np.array(kv),
                                                3, 3)

    # these are randomly sampled points on the surface of the predicted spline
    points = np.array(spline_surf.evaluate_list(parameters))

    input = up_sample_points_torch_in_range(input_points_[0], 2000, 2100)
    input = input.data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(points, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]

    _, _, ku, kv = uniform_knot_bspline_(new_cp_size, new_cp_size, new_degree, new_degree, 2)

    NU = []
    NV = []
    for index in range(parameters.shape[0]):
        nu, nv = bspline.basis_functions(parameters[index], new_cp_size, new_cp_size, ku, kv, new_degree, new_degree)
        NU.append(nu)
        NV.append(nv)
    NU = np.concatenate(NU, 1).T
    NV = np.concatenate(NV, 1).T

    new_control_points = fit_bezier_surface_fit_kronecker(matched, NU, NV)
    new_spline_surf = bspline.create_geomdl_surface(new_control_points,
                                                    np.array(ku),
                                                    np.array(kv),
                                                    new_degree, new_degree)

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = new_spline_surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = optimized_points.reshape((30, 30, 3))
    optimized_points = torch.cat([optimized_points, optimized_points[0:1]], 0)
    optimized_points = optimized_points.reshape((930, 3))
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points

