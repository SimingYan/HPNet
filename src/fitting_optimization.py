"""
This script describes all fitting modules like bspline fitting, geometric 
primitives. The idea is to call each module with required input parameters
and get as an output the parameters of fitting.
"""
import numpy as np
import open3d
import scipy
import torch
from lapsolver import solve_dense
from open3d import *

from src.VisUtils import tessalate_points
from src.curve_utils import DrawSurfs
#from src.utils import visualize_point_cloud

Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector

draw_surf = DrawSurfs()
regular_parameters = draw_surf.regular_parameterization(30, 30)
EPS = np.finfo(np.float32).eps


class Arap:
    def __init__(self, size_u=31, size_v=30):
        """
        As rigid as possible transformation of mesh,
        """
        self.size_u = size_u
        self.size_v = size_v
        l = np.array(self.get_boundary_indices(size_u, size_v))

        indices = []
        for i in range(l.shape[0]):
            indices.append(np.unravel_index(np.ravel_multi_index(l[i],
                                                                 [size_u, size_v]),
                                            [size_u * size_v])[0])
        self.indices = indices

    def deform(self, recon_points, gt_points, viz=False):
        """
        ARAP, given recon_points, that are in grid, we first create a mesh out of
        it, then we do max matching to find correspondance between gt and boundary
        points. Then we do ARAP over the mesh, making the boundary points go closer
        to the matched points. Note that this is taking only the points 
        TODO: better way to do it is do maximal matching between all points and use
        only the boundary points as the pivot points.
        """
        new_recon_points = recon_points.reshape((self.size_u, self.size_v, 3))
        mesh = tessalate_points(recon_points, self.size_u, self.size_v)

        new_recon_points = recon_points.reshape((self.size_u, self.size_v, 3))

        mesh_ = mesh
        for i in range(1):
            mesh, constraint_ids, constraint_pos = self.generate_handles(mesh_,
                                                                         self.indices,
                                                                         gt_points,
                                                                         np.array(mesh_.vertices))
            constraint_ids = np.array(constraint_ids, dtype=np.int32)
            constraint_pos = open3d.utility.Vector3dVector(constraint_pos)
            mesh_prime = mesh.deform_as_rigid_as_possible(
                open3d.utility.IntVector(constraint_ids), constraint_pos, max_iter=500)
            mesh_ = mesh_prime

        if viz:
            pcd = visualize_point_cloud(gt_points)
            mesh_prime.compute_vertex_normals()
            mesh.paint_uniform_color((1, 0, 0))
            handles = open3d.geometry.PointCloud()
            handles.points = constraint_pos
            handles.paint_uniform_color((0, 1, 0))
            open3d.visualization.draw_geometries([mesh, mesh_prime, handles, pcd])
        return mesh_prime

    def get_boundary_indices(self, m, n):
        l = []
        for i in range(m):
            for j in range(n):
                if (j == 0):
                    l.append((i, j))
                elif (j == n - 1):
                    l.append((i, j))
        return l

    def generate_handles(self, mesh, indices, input_points, recon_points):
        matched_points = self.define_matching(input_points, recon_points)
        dist = matched_points - recon_points
        vertices = np.asarray(mesh.vertices)

        handle_ids = indices
        handle_positions = []
        for i in indices:
            handle_positions.append(vertices[i] + dist[i])
        return mesh, handle_ids, handle_positions

    def define_matching(self, input, out):
        # Input points need to at least 1.2 times more than output points
        #L = np.random.choice(np.arange(input.shape[0]), int(1.2 * out.shape[0]), replace=False)
        L = np.random.choice(np.arange(input.shape[0]), int(1.2 * out.shape[0]), replace=True)
        input = input[L]

        dist = scipy.spatial.distance.cdist(out, input)
        rids, cids = solve_dense(dist)
        matched = input[cids]
        return matched

