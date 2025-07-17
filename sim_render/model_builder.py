"""Core model building utilities for 3D scene construction."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RawMesh:
    """Raw mesh data."""

    vertices: np.ndarray  # Shape: (N, 3)
    normals: np.ndarray  # Shape: (N, 3)
    indices: np.ndarray  # Shape: (M,) flat array
    colors: Optional[np.ndarray] = None  # Shape: (N, 4) RGBA


@dataclass
class BodyGeometry:
    """Geometry data for a body."""

    body_id: int
    meshes: List[RawMesh]
    transform: np.ndarray  # 4x4 transformation matrix
    is_plane: bool = False


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

    def encode_mesh(self, mesh: RawMesh) -> Dict[str, Any]:
        """Encode a mesh into a format suitable for export.

        Returns:
            Dict with encoded mesh data
        """
        return {
            "vertices": mesh.vertices,
            "normals": mesh.normals,
            "indices": mesh.indices,
            "colors": mesh.colors,
        }

    def get_encoded_bodies(self) -> List[Dict[str, Any]]:
        """Get all bodies with encoded meshes.

        Returns:
            List of dicts with body data
        """
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

        # Face indices (two triangles per face)
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Back
                [4, 6, 5],
                [4, 7, 6],  # Front
                [0, 4, 5],
                [0, 5, 1],  # Bottom
                [2, 6, 7],
                [2, 7, 3],  # Top
                [0, 3, 7],
                [0, 7, 4],  # Left
                [1, 5, 6],
                [1, 6, 2],  # Right
            ],
            dtype=np.uint32,
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
