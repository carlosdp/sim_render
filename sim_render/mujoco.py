"""MuJoCo model building and rendering utilities."""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    warnings.warn("MuJoCo not available. MuJoCo-specific features will be disabled.")

from .model_builder import ModelBuilder, RawMesh, quaternion_to_matrix


def extract_mujoco_geometry(model) -> Tuple[List[Any], Optional[Any]]:
    """Extract geometry data from a MuJoCo model.

    Returns:
        Tuple of (bodies, camera_data)
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo is required for this functionality")

    bodies = []
    camera_data = None

    # Extract camera if present
    if model.ncam > 0:
        cam_id = 0
        cam_pos = model.cam_pos[cam_id]
        cam_quat = model.cam_quat[cam_id]
        cam_fovy = model.cam_fovy[cam_id]

        # Convert quaternion to rotation matrix
        cam_mat = quaternion_to_matrix(cam_quat)

        # Camera forward is -Z in camera space
        cam_forward = -cam_mat[2]
        cam_up = cam_mat[1]

        from .model_builder import CameraData

        camera_data = CameraData(
            position=cam_pos.copy(),
            target=cam_pos + cam_forward,
            up=cam_up.copy(),
            fov=np.radians(cam_fovy),
        )

    # Process each geom
    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        geom_size = model.geom_size[geom_id]
        geom_pos = model.geom_pos[geom_id]
        geom_quat = model.geom_quat[geom_id]
        geom_rgba = model.geom_rgba[geom_id]
        geom_group = model.geom_group[geom_id]
        geom_body = model.geom_bodyid[geom_id]

        # Only include geoms from groups 0, 1, and 2
        if not any(geom_group & (1 << i) for i in range(3)):
            continue

        mesh = None
        transform = np.eye(4)
        # Convert quaternion to rotation matrix
        transform[:3, :3] = quaternion_to_matrix(geom_quat)
        transform[:3, 3] = geom_pos

        # Generate mesh based on geom type
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = ModelBuilder.generate_sphere_mesh(
                radius=geom_size[0], color=geom_rgba
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = ModelBuilder.generate_box_mesh(
                half_sizes=geom_size[:3], color=geom_rgba
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            # Planes in MuJoCo are infinite, we'll make a large finite plane
            mesh = ModelBuilder.generate_plane_mesh(size=10.0, color=geom_rgba)
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            geom_dataid = model.geom_dataid[geom_id]
            if geom_dataid >= 0:
                mesh_id = geom_dataid
                vert_start = model.mesh_vertadr[mesh_id]
                vert_count = model.mesh_vertnum[mesh_id]
                face_start = model.mesh_faceadr[mesh_id]
                face_count = model.mesh_facenum[mesh_id]

                vertices = model.mesh_vert[
                    vert_start : vert_start + vert_count
                ].reshape(-1, 3)
                faces = model.mesh_face[face_start : face_start + face_count].reshape(
                    -1, 3
                )

                # Compute normals
                normals = np.zeros_like(vertices)
                for face in faces:
                    v0, v1, v2 = vertices[face]
                    normal = np.cross(v1 - v0, v2 - v0)
                    normal = normal / (np.linalg.norm(normal) + 1e-8)
                    normals[face] += normal

                # Normalize
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / (norms + 1e-8)

                # Apply color
                colors = np.tile(geom_rgba, (len(vertices), 1))

                mesh = RawMesh(
                    vertices=vertices,
                    normals=normals,
                    indices=faces.flatten(),
                    colors=colors,
                )

        if mesh is not None:
            from .model_builder import BodyGeometry

            bodies.append(
                BodyGeometry(
                    body_id=geom_body,
                    meshes=[mesh],
                    transform=transform,
                    is_plane=geom_type == mujoco.mjtGeom.mjGEOM_PLANE,
                )
            )

    return bodies, camera_data


def convert_mujoco_to_gltf_position(pos: np.ndarray) -> np.ndarray:
    """Convert position from MuJoCo to glTF coordinate system."""
    # MuJoCo: Y-up, Z-forward
    # glTF: Y-up, Z-backward
    return np.array([pos[0], pos[1], -pos[2]])


def convert_mujoco_to_gltf_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from MuJoCo to glTF coordinate system."""
    # MuJoCo quaternion: (w, x, y, z)
    # Need to adjust for coordinate system change
    w, x, y, z = quat
    # Flip Z axis rotation
    return np.array([w, x, y, -z])


class MujocoRender:
    """Render MuJoCo models with animation support."""

    def __init__(self, model):
        """Initialize with a MuJoCo model.

        Args:
            model: MuJoCo MjModel instance
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required for MujocoRender")

        self.model = model
        self.model_builder = ModelBuilder()
        self._animation_context = None
        self._prepared = False

    def prepare(self):
        """Extract static geometry from the MuJoCo model."""
        if self._prepared:
            return

        bodies, camera_data = extract_mujoco_geometry(self.model)

        # Add bodies to model builder
        for body in bodies:
            self.model_builder.add_body(
                body_id=body.body_id,
                meshes=body.meshes,
                transform=body.transform,
                is_plane=body.is_plane,
            )

        # Set camera if present
        if camera_data:
            self.model_builder.set_camera(
                position=camera_data.position,
                target=camera_data.target,
                up=camera_data.up,
                fov=camera_data.fov,
            )

        self._prepared = True

    def render(self, data, time: Optional[float] = None):
        """Render a frame with the given MjData.

        Args:
            data: MuJoCo MjData instance
            time: Optional time for the frame (defaults to data.time)
        """
        if not self._prepared:
            self.prepare()

        if self._animation_context is None:
            # Single frame render - just update transforms
            for body_id in range(self.model.nbody):
                if body_id == 0:  # Skip world body
                    continue

                pos = data.xpos[body_id]
                quat = data.xquat[body_id]

                # Convert to glTF coordinate system
                gltf_pos = convert_mujoco_to_gltf_position(pos)
                gltf_quat = convert_mujoco_to_gltf_quaternion(quat)

                # Update body transform
                self.model_builder.update_body_transform(body_id, gltf_pos, gltf_quat)
        else:
            # Animation render - add frame
            frame_time = time if time is not None else data.time
            self._add_animation_frame(data, frame_time)

    def _add_animation_frame(self, data, time: float):
        """Add an animation frame."""
        transforms = {}

        for body_id in range(self.model.nbody):
            if body_id == 0:  # Skip world body
                continue

            pos = data.xpos[body_id]
            quat = data.xquat[body_id]

            # Convert to glTF coordinate system
            gltf_pos = convert_mujoco_to_gltf_position(pos)
            gltf_quat = convert_mujoco_to_gltf_quaternion(quat)

            transforms[body_id] = {
                "translation": gltf_pos.tolist(),
                "rotation": gltf_quat.tolist(),
            }

        self.model_builder.add_animation_frame(time, transforms)

    def animation(self, fps: int = 30):
        """Context manager for recording animations.

        Args:
            fps: Frames per second for the animation

        Usage:
            with render.animation(fps=30):
                for _ in range(100):
                    render.render(data)
        """
        return _AnimationContext(self, fps)

    def save(self, filename: str):
        """Save the rendered model to a GLB file.

        Args:
            filename: Output filename
        """
        if not self._prepared:
            self.prepare()

        self.model_builder.save_to_glb(filename)


class _AnimationContext:
    """Context manager for animation recording."""

    def __init__(self, render: MujocoRender, fps: int):
        self.render = render
        self.fps = fps
        self.frame_time = 0.0
        self.frame_duration = 1.0 / fps

    def __enter__(self):
        self.render._animation_context = self
        self.render.model_builder.start_animation()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.render._animation_context = None
