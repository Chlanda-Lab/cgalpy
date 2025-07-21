import pathlib
from typing import Tuple, Optional
from ctypes import CDLL, c_size_t, c_void_p, c_double, c_uint32, c_ssize_t, c_bool
import numpy as np

NDPOINTER = lambda ndim, dtype: np.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, flags="C")

lib = CDLL(str(pathlib.Path(__file__).with_name('libcgalpy.so')))

# Mesh
# Constructors
lib.Mesh_new.restype = c_void_p
# Add
lib.Mesh_add.argtypes = [c_void_p, c_void_p]
# Set indices/vertices
lib.Mesh_set_indices.argtypes = [c_void_p, NDPOINTER(2, np.int64), c_size_t]
lib.Mesh_set_vertices.argtypes = [c_void_p, NDPOINTER(2, np.double), c_size_t]
# Number of indices/faces
lib.Mesh_number_of_faces.argtypes = [c_void_p]
lib.Mesh_number_of_faces.restype = c_size_t
lib.Mesh_number_of_vertices.argtypes = [c_void_p]
lib.Mesh_number_of_vertices.restype = c_size_t
# Get indices/faces
lib.Mesh_get_vertices.argtypes = [c_void_p, NDPOINTER(2, np.double)]
lib.Mesh_get_indices.argtypes = [c_void_p, NDPOINTER(2, np.intp)]
lib.Mesh_isotropic_remeshing.argtypes = [c_void_p, c_double, c_uint32]
# Get faces
lib.Mesh_get_faces.argtypes = [c_void_p, NDPOINTER(3, np.double)]
# Get normals
lib.Mesh_compute_vertex_normals.argtypes = [c_void_p, NDPOINTER(2, np.double)]
lib.Mesh_compute_face_normals.argtypes = [c_void_p, NDPOINTER(2, np.double)]
# Repair
lib.Mesh_remove_isolated_vertices.argtypes = [c_void_p]
lib.Mesh_collect_garbage.argtypes = [c_void_p]
# Curvature
#lib.Mesh_compute_curvature.argtypes = [c_void_p, NDPOINTER(1, np.double), NDPOINTER(1, np.double)]
# Smoothing
lib.Mesh_angle_and_area_smoothing.argtypes = [c_void_p]
# Face centroids
lib.Mesh_get_face_centroids.argtypes = [c_void_p, NDPOINTER(2, np.double)]

# AABBTree
# Constructor
lib.Tree_new.restype = c_void_p
lib.Tree_new.argtypes = [c_void_p]
# Squared distance
lib.Tree_squared_distances.argtypes = [c_void_p, NDPOINTER(2, np.double), c_ssize_t, NDPOINTER(1, np.double)]
# Closest_points_and_primitives
lib.Tree_closest_points_and_primitives.argtypes = [c_void_p, NDPOINTER(2, np.double), c_ssize_t, NDPOINTER(2, np.double), NDPOINTER(1, np.intp)]
# Intersection
lib.Tree_first_intersections.argtypes = [c_void_p, NDPOINTER(2, np.double), NDPOINTER(2, np.double), c_size_t, NDPOINTER(2, np.double), NDPOINTER(1, np.intp)]

# Util functions
lib.free_obj.argtypes = [c_void_p]

class Mesh:
    def __init__(self, vertices: np.ndarray, indices: np.ndarray):
        assert vertices.ndim == 2
        assert indices.ndim == 2
        self._mesh = lib.Mesh_new()
        self._aabbtree = None
        lib.Mesh_set_vertices(self._mesh, vertices, len(vertices))
        lib.Mesh_set_indices(self._mesh, indices, len(indices))
        
    def __del__(self):
        lib.free_obj(self._mesh)
        if self._aabbtree is not None:
            lib.free_obj(self._aabbtree)
            
    def add(self, other: "Mesh"):
        lib.Mesh_add(self._mesh, other._mesh)

    def number_of_vertices(self) -> int:
        return lib.Mesh_number_of_vertices(self._mesh)

    def number_of_faces(self) -> int:
        return lib.Mesh_number_of_faces(self._mesh)

    def get_vertices(self) -> np.ndarray:
        arr = np.zeros(shape=(self.number_of_vertices(), 3), dtype=float)
        lib.Mesh_get_vertices(self._mesh, arr)
        return arr

    def get_indices(self) -> np.ndarray:
        arr = np.zeros(shape=(self.number_of_faces(), 3), dtype=int)
        lib.Mesh_get_indices(self._mesh, arr)
        return arr
    
    def get_faces(self) -> np.ndarray:
        arr = np.zeros(shape=(self.number_of_faces(), 3, 3), dtype=float)
        lib.Mesh_get_faces(self._mesh, arr)
        return arr
    
    def compute_vertex_normals(self) -> np.ndarray:
        arr = np.zeros((self.number_of_vertices(), 3), dtype=float)
        lib.Mesh_compute_vertex_normals(self._mesh, arr)
        return arr
    
    def compute_face_normals(self) -> np.ndarray:
        arr = np.zeros((self.number_of_faces(), 3), dtype=float)
        lib.Mesh_compute_face_normals(self._mesh, arr)
        return arr

    def get_face_centroids(self) -> np.ndarray:
        arr = np.empty((self.number_of_faces(), 3), dtype=float)
        lib.Mesh_get_face_centroids(self._mesh, arr)
        return arr
    
    def remove_isolated_vertices(self):
        lib.Mesh_remove_isolated_vertices(self._mesh)

    def collect_garbage(self):
        lib.Mesh_collect_garbage(self._mesh)

    def isotropic_remeshing(self, target_edge_length: float, nb_iter: int):
        lib.Mesh_isotropic_remeshing(self._mesh, target_edge_length, nb_iter)
        
    def compute_curvature(self) -> Tuple[np.ndarray, np.ndarray]:
        mean_curvatures = np.zeros(self.number_of_vertices(), dtype=float)
        gaussian_curvatures = np.zeros(self.number_of_vertices(), dtype=float)
        lib.Mesh_compute_curvature(self._mesh, mean_curvatures, gaussian_curvatures)
        return mean_curvatures, gaussian_curvatures
    
    def angle_and_area_smoothing(self):
        lib.Mesh_angle_and_area_smoothing(self._mesh)
        
    def closest_points_and_primitives(self, query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert query_points.ndim == 2
        assert query_points.dtype == float
        if self._aabbtree is None:
            self._aabbtree = lib.Tree_new(self._mesh)
        closest_points = np.empty_like(query_points)
        closest_primitives = np.empty(len(query_points), dtype=int)
        lib.Tree_closest_points_and_primitives(self._aabbtree, query_points, len(query_points), closest_points, closest_primitives)
        return closest_points, closest_primitives

    def squared_distances(self, points: np.ndarray) -> np.ndarray:
        assert points.ndim == 2
        assert points.dtype == float
        if self._aabbtree is None:
            self._aabbtree = lib.Tree_new(self._mesh)
        arr = np.empty(len(points), dtype=float)
        lib.Tree_squared_distances(self._aabbtree, points, len(points), arr)
        return arr
    
    def first_intersections(self, ray_start_points: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._aabbtree is None:
            self._aabbtree = lib.Tree_new(self._mesh)
        ray_start_points = ray_start_points.reshape((-1, 3))
        normals = normals.reshape((-1, 3))
        assert ray_start_points.shape == normals.shape
        assert ray_start_points.dtype == float
        assert normals.dtype == float
        intersection_points = np.full_like(ray_start_points, fill_value=np.inf)
        faces = np.full(len(intersection_points), fill_value=-1, dtype=np.intp)
        lib.Tree_first_intersections(
            self._aabbtree,
            ray_start_points,
            ray_start_points + normals,
            len(intersection_points),
            intersection_points,
            faces,
        )
        return intersection_points, faces