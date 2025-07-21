#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <algorithm>
#include <list>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <boost/variant/variant.hpp>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
//#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>
#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/Named_function_parameters.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef K::Vector_3 Vector;
typedef K::Ray_3 Ray;

typedef CGAL::Surface_mesh<Point> Mesh;
typedef Mesh::Vertex_index vertex_descriptor;
typedef Mesh::Face_index face_descriptor;
typedef Mesh::Halfedge_index halfedge_descriptor;

typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

namespace PMP = CGAL::Polygon_mesh_processing;

extern "C" {

  Mesh * Mesh_new() {
    return new Mesh();
  }
  
  void Mesh_set_vertices(Mesh * mesh, double * vertices, size_t n_vertices) {
    for (size_t v = 0; v < n_vertices; v++) {
        auto i = mesh->add_vertex(Point(vertices[v*3 + 0], vertices[v*3 + 1], vertices[v*3 + 2]));
    }
  }

  void Mesh_set_indices(Mesh * mesh, unsigned long * indices, size_t n_indices) {
    for (size_t i = 0; i < n_indices; i++) {
         mesh->add_face(
          vertex_descriptor(indices[i*3 + 0]),
          vertex_descriptor(indices[i*3 + 1]),
          vertex_descriptor(indices[i*3 + 2]));
    }
  }
  
  void Mesh_add(Mesh * mesh, Mesh * other) {
    *mesh+=*other;
  }

  size_t Mesh_number_of_vertices(Mesh * mesh) {
    return mesh->number_of_vertices();
  }

  size_t Mesh_number_of_faces(Mesh * mesh) {
    return mesh->number_of_faces();
  }

  void Mesh_get_vertices(Mesh * mesh, double * vertices) {
    size_t offset = 0;
    for (vertex_descriptor v : mesh->vertices()) {
      auto point = mesh->point(v);
      vertices[offset++] = point[0];
      vertices[offset++] = point[1];
      vertices[offset++] = point[2];
    }
  }
  
  void Mesh_get_indices(Mesh * mesh, size_t * indices) {
    size_t offset = 0;
    for (face_descriptor face : mesh->faces()) {
      halfedge_descriptor halfedge1 = mesh->halfedge(face);
      halfedge_descriptor halfedge2 = mesh->next(halfedge1);
      halfedge_descriptor halfedge3 = mesh->next(halfedge2);
      vertex_descriptor vertex1 = mesh->target(halfedge1);
      vertex_descriptor vertex2 = mesh->target(halfedge2);
      vertex_descriptor vertex3 = mesh->target(halfedge3);
      indices[offset++] = (size_t) vertex1;
      indices[offset++] = (size_t) vertex2;
      indices[offset++] = (size_t) vertex3;
    }
  }
  
  void Mesh_remove_isolated_vertices(Mesh * mesh) {
    PMP::remove_isolated_vertices(*mesh);
  }
  
  void Mesh_collect_garbage(Mesh * mesh) {
    mesh->collect_garbage();
  }

  void Mesh_isotropic_remeshing(Mesh * mesh, double target_edge_length, unsigned int nb_iter) {
    CGAL::Polygon_mesh_processing::isotropic_remeshing(
      faces(*mesh),
      target_edge_length,
      *mesh,
      CGAL::parameters::number_of_iterations(nb_iter).protect_constraints(true)
    );
  }
  
  void Mesh_get_faces(Mesh * mesh, double * faces) {
    size_t offset = 0;
    for (face_descriptor face : mesh->faces()) {
      halfedge_descriptor halfedge1 = mesh->halfedge(face);
      halfedge_descriptor halfedge2 = mesh->next(halfedge1);
      halfedge_descriptor halfedge3 = mesh->next(halfedge2);
      vertex_descriptor vertex1 = mesh->target(halfedge1);
      vertex_descriptor vertex2 = mesh->target(halfedge2);
      vertex_descriptor vertex3 = mesh->target(halfedge3);
      auto point = mesh->point(vertex1);
      faces[offset++] = point[0];
      faces[offset++] = point[1];
      faces[offset++] = point[2];
      point = mesh->point(vertex2);
      faces[offset++] = point[0];
      faces[offset++] = point[1];
      faces[offset++] = point[2];
      point = mesh->point(vertex3);
      faces[offset++] = point[0];
      faces[offset++] = point[1];
      faces[offset++] = point[2];
    }
  }
  
  void Mesh_compute_vertex_normals(Mesh * mesh, double * normals) {
    size_t offset = 0;
    auto vertex_normals = mesh->add_property_map<vertex_descriptor, Vector>("v:normal", CGAL::NULL_VECTOR).first;
    PMP::compute_vertex_normals(*mesh, vertex_normals);
    for (vertex_descriptor v : mesh->vertices()) {
        normals[offset++] = vertex_normals[v].x();
        normals[offset++] = vertex_normals[v].y();
        normals[offset++] = vertex_normals[v].z();
    }
  }
  void Mesh_compute_face_normals(Mesh * mesh, double * normals) {
    size_t offset = 0;
    auto face_normals = mesh->add_property_map<face_descriptor, Vector>("f:normal", CGAL::NULL_VECTOR).first;
    PMP::compute_face_normals(*mesh, face_normals);
    for (face_descriptor v : mesh->faces()) {
        normals[offset++] = face_normals[v].x();
        normals[offset++] = face_normals[v].y();
        normals[offset++] = face_normals[v].z();
    }
  }
  
  void Mesh_get_face_centroids(Mesh * mesh, double * vertices) {
    size_t offset = 0;
    for (face_descriptor f : mesh->faces()) {
      const halfedge_descriptor halfedge1 = mesh->halfedge(f);
      const halfedge_descriptor halfedge2 = mesh->next(halfedge1);
      const halfedge_descriptor halfedge3 = mesh->next(halfedge2);
      const vertex_descriptor vertex1 = mesh->target(halfedge1);
      const vertex_descriptor vertex2 = mesh->target(halfedge2);
      const vertex_descriptor vertex3 = mesh->target(halfedge3);
      const Point p1 = mesh->point(vertex1);
      const Point p2 = mesh->point(vertex2);
      const Point p3 = mesh->point(vertex3);
      const Point centroid = Point(
        (p1.x() + p2.x() + p3.x())/3.0,
        (p1.y() + p2.y() + p3.y())/3.0,
        (p1.z() + p2.z() + p3.z())/3.0
      );
      vertices[offset++] = centroid.x();
      vertices[offset++] = centroid.y();
      vertices[offset++] = centroid.z();
    }
  }
  /*
  void Mesh_compute_curvature(Mesh * mesh, double * mean_curvatures, double * gaussian_curvatures) {
    auto mean_curvature_map = mesh->add_property_map<vertex_descriptor, double>("v:mean_curvature", 0).first;
    auto gaussian_curvature_map = mesh->add_property_map<vertex_descriptor, double>("v:gaussian_curvature", 0).first;
    PMP::interpolated_corrected_curvatures(
      *mesh,
      CGAL::parameters::vertex_mean_curvature_map(mean_curvature_map).vertex_Gaussian_curvature_map(gaussian_curvature_map)
    );
    size_t offset = 0;
    for (vertex_descriptor v : mesh->vertices()) {
        mean_curvatures[offset] = mean_curvature_map[v];
        gaussian_curvatures[offset++] = gaussian_curvature_map[v];
    }
  }
  */
  
  void Mesh_angle_and_area_smoothing(Mesh * mesh) {
    PMP::angle_and_area_smoothing(*mesh);
  }
  
  Tree * Tree_new(Mesh * mesh) {
    return new Tree(mesh->faces().first, mesh->faces().second, *mesh);
  }
  
  void Tree_closest_points_and_primitives(Tree * tree, double * query_points, size_t n_query_points, double * closest_points, size_t * closest_primitives) {
    for (size_t p = 0; p<n_query_points; p++) {
      const size_t offset = p*3;
      const Point query(
        query_points[offset+0],
        query_points[offset+1],
        query_points[offset+2]);
      const auto pair = tree->closest_point_and_primitive(query);
      for (size_t i = 0; i<3; i++)
        closest_points[offset+i] = pair.first[i];
      closest_primitives[p] = (size_t) pair.second;
    }
  }

  void Tree_squared_distances(Tree * tree, double * query_points, size_t n_query_points, double * sq_distances) {
    for (size_t p=0; p<n_query_points; p++) {
      const size_t offset = p*3;
      const Point query(
        query_points[offset+0],
        query_points[offset+1],
        query_points[offset+2]
      );
      sq_distances[p] = tree->squared_distance(query);
    }
  }
  
  void Tree_first_intersections(Tree * tree, double * ray_start_points, double * normals, size_t n_rays, double * intersection_points, size_t * faces) {
    for (size_t ray_idx; ray_idx<n_rays; ray_idx++) {
      size_t offset = ray_idx * 3;
      Ray ray(
        Point(ray_start_points[offset], ray_start_points[offset+1], ray_start_points[offset+2]),
        Point(normals[offset], normals[offset+1], normals[offset+2])
      );
      auto intersection = tree->first_intersection(ray);
      if (intersection) {
        if (const Point* point = boost::get<Point>(&(intersection->first))) {
          for (size_t i=0; i<3; i++)
            intersection_points[offset+i] = (*point)[i];
          faces[ray_idx] = (size_t) intersection.value().second;
        }
      }
    }
  }
  
  void free_obj(void * obj) {
    free(obj);
  }
}
