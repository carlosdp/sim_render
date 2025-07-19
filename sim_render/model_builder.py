"""Core model building utilities for 3D scene construction."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RawMesh:
    """Raw mesh data."""

    vertices: np.ndarray  # Shape: (N, 3)
    normals: np.ndarray  # Shape: (N, 3)
    indices: np.ndarray  # Shape: (M,) flat array
    colors: Optional[np.ndarray] = None  # Shape: (N, 4) RGBA
    material: Optional[Dict[str, Any]] = None  # Material and texture info


@dataclass
class BodyGeometry:
    """Geometry data for a body."""

    body_id: int
    meshes: List[RawMesh]
    transform: np.ndarray  # 4x4 transformation matrix
    is_plane: bool = False
    cameras: Optional[List[Dict[str, Any]]] = None


@dataclass
class CameraData:
    """Camera configuration."""

    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float  # Field of view in radians


class ModelBuilder:
    """Builder for 3D models with mesh encoding and animation support."""

    def __init__(self):
        self.bodies: Dict[int, BodyGeometry] = {}
        self.camera: Optional[CameraData] = None
        self.animation_frames: List[Tuple[float, Dict[int, Dict[str, Any]]]] = []
        self._animation_started = False
        # Compatibility with old GlbSceneBuilder interface
        self.individual_meshes: List[Dict[str, Any]] = []

    def add_body(
        self,
        body_id: int,
        meshes: List[RawMesh],
        transform: Optional[np.ndarray] = None,
        is_plane: bool = False,
    ):
        """Add a body with meshes to the model.

        Args:
            body_id: Unique identifier for the body
            meshes: List of meshes for this body
            transform: 4x4 transformation matrix (defaults to identity)
            is_plane: Whether this body represents a plane
        """
        if transform is None:
            transform = np.eye(4)

        self.bodies[body_id] = BodyGeometry(
            body_id=body_id,
            meshes=meshes,
            transform=transform.copy(),
            is_plane=is_plane,
        )

    def update_body_transform(
        self, body_id: int, position: np.ndarray, rotation: Optional[np.ndarray] = None
    ):
        """Update the transform of a body.

        Args:
            body_id: Body identifier
            position: 3D position
            rotation: Optional quaternion (w, x, y, z)
        """
        if body_id not in self.bodies:
            return

        transform = np.eye(4)
        transform[:3, 3] = position

        if rotation is not None:
            # Convert quaternion to rotation matrix
            w, x, y, z = rotation
            transform[:3, :3] = quaternion_to_matrix(rotation)

        self.bodies[body_id].transform = transform

    def add_mesh(
        self,
        body_id: int,
        position: List[float],
        rotation: List[float],
        vertices: List[List[float]],
        normals: List[List[float]],
        indices: List[int],
        color: List[float],
        material: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a mesh in the old GlbSceneBuilder format for compatibility."""
        mesh = RawMesh(
            vertices=np.array(vertices, dtype=np.float32),
            normals=np.array(normals, dtype=np.float32),
            indices=np.array(indices, dtype=np.uint32),
            colors=np.tile(color, (len(vertices), 1)).astype(np.float32),
            material=material,
        )

        self.individual_meshes.append(
            {
                "body_id": body_id,
                "position": position,
                "rotation": rotation,
                "mesh": mesh,
            }
        )

    def add_camera(
        self,
        body_id: int,
        position: List[float],
        rotation: List[float],
        camera_data: CameraData,
    ) -> None:
        """Add a camera to a specific body."""
        # Add to the individual_cameras list for compatibility with existing export
        if not hasattr(self, "individual_cameras"):
            self.individual_cameras = []

        self.individual_cameras.append(
            {
                "body_id": body_id,
                "position": position,
                "rotation": rotation,
                "camera": camera_data,
            }
        )

    def set_camera(
        self, position: np.ndarray, target: np.ndarray, up: np.ndarray, fov: float
    ):
        """Set the camera configuration.

        Args:
            position: Camera position
            target: Look-at target
            up: Up vector
            fov: Field of view in radians
        """
        self.camera = CameraData(
            position=position.copy(), target=target.copy(), up=up.copy(), fov=fov
        )

    def start_animation(self):
        """Start recording animation frames."""
        self.animation_frames = []
        self._animation_started = True

    def add_animation_frame(self, time: float, transforms: Dict[int, Dict[str, Any]]):
        """Add an animation frame.

        Args:
            time: Time in seconds
            transforms: Dict mapping body_id to transform data with keys:
                - 'translation': [x, y, z]
                - 'rotation': [w, x, y, z] quaternion
        """
        if not self._animation_started:
            self.start_animation()

        self.animation_frames.append((time, transforms))

    def encode_mesh(self, mesh_data) -> Dict[str, Any]:
        """Encode a mesh into a format suitable for export.

        Returns:
            Dict with encoded mesh data
        """
        if isinstance(mesh_data, dict) and "mesh" in mesh_data:
            # New format with transform
            mesh = mesh_data["mesh"]
            return {
                "vertices": mesh.vertices,
                "normals": mesh.normals,
                "indices": mesh.indices,
                "colors": mesh.colors,
                "material": mesh.material,
                "transform": mesh_data.get("transform", np.eye(4)),
                "is_plane": mesh_data.get("is_plane", False),
            }
        else:
            # Old format - direct RawMesh
            return {
                "vertices": mesh_data.vertices,
                "normals": mesh_data.normals,
                "indices": mesh_data.indices,
                "colors": mesh_data.colors,
                "material": mesh_data.material,
                "transform": np.eye(4),
                "is_plane": False,
            }

    def get_encoded_bodies(self) -> List[Dict[str, Any]]:
        """Get all bodies with encoded meshes.

        Returns:
            List of dicts with body data
        """
        # If we have individual meshes (old format), use those
        if self.individual_meshes:
            # Group individual meshes by body_id
            body_groups = {}
            for mesh_data in self.individual_meshes:
                body_id = mesh_data["body_id"]
                if body_id not in body_groups:
                    body_groups[body_id] = []
                body_groups[body_id].append(mesh_data)

            encoded = []
            for body_id, meshes in body_groups.items():
                encoded_meshes = []
                for mesh_data in meshes:
                    encoded_meshes.append(
                        {
                            "vertices": mesh_data["mesh"].vertices,
                            "normals": mesh_data["mesh"].normals,
                            "indices": mesh_data["mesh"].indices,
                            "colors": mesh_data["mesh"].colors,
                            "material": mesh_data["mesh"].material,
                            "position": mesh_data["position"],
                            "rotation": mesh_data["rotation"],
                            "is_plane": self._is_plane_mesh(mesh_data["mesh"]),
                        }
                    )

                encoded.append(
                    {
                        "body_id": body_id,
                        "meshes": encoded_meshes,
                        "transform": np.eye(4),  # Body transform is identity
                        "is_plane": any(
                            m.get("is_plane", False) for m in encoded_meshes
                        ),
                    }
                )
            return encoded

        # Otherwise use the new format
        encoded = []
        for body_id, body in self.bodies.items():
            encoded_meshes = [self.encode_mesh(mesh) for mesh in body.meshes]
            encoded.append(
                {
                    "body_id": body_id,
                    "meshes": encoded_meshes,
                    "transform": body.transform,
                    "is_plane": body.is_plane,
                }
            )
        return encoded

    def _is_plane_mesh(self, mesh: RawMesh) -> bool:
        """Check if a mesh is a plane (all Z coordinates near 0)."""
        return all(abs(v[2]) < 0.001 for v in mesh.vertices)

    def save_to_glb(self, filename: str):
        """Save the model to a GLB file.

        Args:
            filename: Output filename
        """
        from .export.glb import export_to_glb

        export_to_glb(self, filename)

    # Static mesh generation methods
    @staticmethod
    def generate_sphere_mesh(
        radius: float = 1.0,
        rings: int = 16,
        sectors: int = 32,
        color: Optional[np.ndarray] = None,
    ) -> RawMesh:
        """Generate a sphere mesh."""
        vertices = []
        normals = []
        indices = []

        for r in range(rings + 1):
            phi = np.pi * r / rings
            for s in range(sectors + 1):
                theta = 2 * np.pi * s / sectors

                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)

                vertices.append([x, y, z])
                normals.append([x / radius, y / radius, z / radius])

        for r in range(rings):
            for s in range(sectors):
                first = r * (sectors + 1) + s
                second = first + sectors + 1

                indices.extend([first, first + 1, second])
                indices.extend([first + 1, second + 1, second])

        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)

        if color is not None:
            colors = np.tile(color, (len(vertices), 1))
        else:
            colors = np.ones((len(vertices), 4), dtype=np.float32)

        return RawMesh(
            vertices=vertices, normals=normals, indices=indices, colors=colors
        )

    @staticmethod
    def generate_box_mesh(
        half_sizes: np.ndarray, color: Optional[np.ndarray] = None
    ) -> RawMesh:
        """Generate a box mesh."""
        hx, hy, hz = half_sizes

        # 8 vertices of the box
        vertices = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],  # Back face
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],  # Front face
            ],
            dtype=np.float32,
        )

        # Expand vertices for each face (6 faces * 4 vertices = 24 vertices)
        face_vertices = []
        face_normals = []
        face_indices = []

        # Normal vectors for each face
        normals_per_face = [
            [0, 0, -1],  # Back
            [0, 0, 1],  # Front
            [0, -1, 0],  # Bottom
            [0, 1, 0],  # Top
            [-1, 0, 0],  # Left
            [1, 0, 0],  # Right
        ]

        vertex_count = 0
        for face_idx in range(6):
            # Get the 4 vertices for this face
            if face_idx == 0:  # Back
                face_verts = [0, 1, 2, 3]
            elif face_idx == 1:  # Front
                face_verts = [4, 5, 6, 7]
            elif face_idx == 2:  # Bottom
                face_verts = [0, 1, 5, 4]
            elif face_idx == 3:  # Top
                face_verts = [2, 3, 7, 6]
            elif face_idx == 4:  # Left
                face_verts = [0, 3, 7, 4]
            else:  # Right
                face_verts = [1, 2, 6, 5]

            # Add vertices and normals
            for v_idx in face_verts:
                face_vertices.append(vertices[v_idx])
                face_normals.append(normals_per_face[face_idx])

            # Add indices
            face_indices.extend(
                [
                    vertex_count,
                    vertex_count + 1,
                    vertex_count + 2,
                    vertex_count,
                    vertex_count + 2,
                    vertex_count + 3,
                ]
            )
            vertex_count += 4

        vertices = np.array(face_vertices, dtype=np.float32)
        normals = np.array(face_normals, dtype=np.float32)
        indices = np.array(face_indices, dtype=np.uint32)

        if color is not None:
            colors = np.tile(color, (len(vertices), 1))
        else:
            colors = np.ones((len(vertices), 4), dtype=np.float32)

        return RawMesh(
            vertices=vertices, normals=normals, indices=indices, colors=colors
        )

    @staticmethod
    def generate_plane_mesh(
        size: float = 1.0, color: Optional[np.ndarray] = None
    ) -> RawMesh:
        """Generate a plane mesh."""
        half_size = size / 2

        vertices = np.array(
            [
                [-half_size, 0, -half_size],
                [half_size, 0, -half_size],
                [half_size, 0, half_size],
                [-half_size, 0, half_size],
            ],
            dtype=np.float32,
        )

        normals = np.array(
            [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32
        )

        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        if color is not None:
            colors = np.tile(color, (len(vertices), 1))
        else:
            colors = np.ones((len(vertices), 4), dtype=np.float32)

        return RawMesh(
            vertices=vertices, normals=normals, indices=indices, colors=colors
        )

    @staticmethod
    def generate_capsule_mesh(
        radius: float = 1.0,
        half_length: float = 1.0,
        rings: int = 8,
        sectors: int = 16,
        color: Optional[np.ndarray] = None,
    ) -> RawMesh:
        """Generate a capsule mesh (cylinder with hemispherical caps).

        Args:
            radius: Radius of the capsule
            half_length: Half the length of the cylindrical part
            rings: Number of horizontal rings for spherical caps
            sectors: Number of sectors around the capsule
            color: Optional RGBA color

        Returns:
            RawMesh: The generated capsule mesh
        """
        vertices = []
        normals = []
        indices = []

        # Generate top hemisphere
        for r in range(rings + 1):
            phi = (np.pi / 2) * r / rings  # Only go from 0 to pi/2
            z = half_length + radius * np.sin(phi)
            ring_radius = radius * np.cos(phi)

            for s in range(sectors + 1):
                theta = 2 * np.pi * s / sectors

                x = ring_radius * np.cos(theta)
                y = ring_radius * np.sin(theta)

                vertices.append([x, y, z])

                # Normal for hemisphere
                nz = np.sin(phi)
                nr = np.cos(phi)
                normals.append([nr * np.cos(theta), nr * np.sin(theta), nz])

        # Generate cylinder
        cylinder_rings = 2  # Just top and bottom of cylinder
        for r in range(cylinder_rings):
            z = half_length if r == 0 else -half_length

            for s in range(sectors + 1):
                theta = 2 * np.pi * s / sectors

                x = radius * np.cos(theta)
                y = radius * np.sin(theta)

                vertices.append([x, y, z])
                normals.append([np.cos(theta), np.sin(theta), 0])

        # Generate bottom hemisphere
        for r in range(rings + 1):
            phi = (np.pi / 2) * r / rings
            z = -half_length - radius * np.sin(phi)
            ring_radius = radius * np.cos(phi)

            for s in range(sectors + 1):
                theta = 2 * np.pi * s / sectors

                x = ring_radius * np.cos(theta)
                y = ring_radius * np.sin(theta)

                vertices.append([x, y, z])

                # Normal for hemisphere
                nz = -np.sin(phi)
                nr = np.cos(phi)
                normals.append([nr * np.cos(theta), nr * np.sin(theta), nz])

        # Generate indices
        # Top hemisphere - flip winding for Z-axis orientation
        for r in range(rings):
            for s in range(sectors):
                first = r * (sectors + 1) + s
                second = first + sectors + 1

                indices.extend([first, first + 1, second])
                indices.extend([first + 1, second + 1, second])

        # Cylinder
        top_cap_end = (rings + 1) * (sectors + 1)
        for s in range(sectors):
            # Connect top hemisphere to cylinder - corrected winding
            hem_idx = rings * (sectors + 1) + s
            cyl_idx = top_cap_end + s

            indices.extend([hem_idx, hem_idx + 1, cyl_idx])
            indices.extend([hem_idx + 1, cyl_idx + 1, cyl_idx])

            # Connect cylinder sides - flip winding for Z-axis orientation
            top_idx = top_cap_end + s
            bot_idx = top_cap_end + (sectors + 1) + s

            indices.extend([top_idx, bot_idx, top_idx + 1])
            indices.extend([top_idx + 1, bot_idx, bot_idx + 1])

            # Connect cylinder to bottom hemisphere - corrected winding
            cyl_bot_idx = top_cap_end + (sectors + 1) + s
            hem_bot_idx = top_cap_end + 2 * (sectors + 1) + s

            indices.extend([cyl_bot_idx, cyl_bot_idx + 1, hem_bot_idx])
            indices.extend([cyl_bot_idx + 1, hem_bot_idx + 1, hem_bot_idx])

        # Bottom hemisphere - flip winding for Z-axis orientation
        bot_hem_start = top_cap_end + 2 * (sectors + 1)
        for r in range(rings):
            for s in range(sectors):
                first = bot_hem_start + r * (sectors + 1) + s
                second = first + sectors + 1

                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)

        if color is not None:
            colors = np.tile(color, (len(vertices), 1))
        else:
            colors = np.ones((len(vertices), 4), dtype=np.float32)

        return RawMesh(
            vertices=vertices, normals=normals, indices=indices, colors=colors
        )


def triangulate_and_compute_normals(
    vertices: List[List[float]], indices: List[int]
) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    """
    Triangulate mesh and compute per-vertex normals.
    Returns new vertices, normals, and indices with proper triangulation.
    """
    assert len(indices) % 3 == 0, "Indices must be divisible by 3 for triangles"

    tri_count = len(indices) // 3
    new_vertices = []
    new_normals = []
    new_indices = []

    for t in range(tri_count):
        # Get triangle vertices
        i0, i1, i2 = indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]
        v0 = np.array(vertices[i0])
        v1 = np.array(vertices[i1])
        v2 = np.array(vertices[i2])

        # Compute normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len
        else:
            normal = np.array([0.0, 0.0, 1.0])

        # Add vertices and normals
        base_idx = len(new_vertices)
        new_vertices.extend([v0.tolist(), v1.tolist(), v2.tolist()])
        new_normals.extend([normal.tolist()] * 3)
        new_indices.extend([base_idx, base_idx + 1, base_idx + 2])

    return new_vertices, new_normals, new_indices


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    Args:
        quat: Quaternion as [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    # Normalize quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm > 0:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Build rotation matrix
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )
