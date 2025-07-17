import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np
import mujoco
import gymnasium as gym

from .glb_builder import (
    GlbSceneBuilder,
    generate_box_mesh,
    generate_sphere_mesh,
    generate_plane_mesh,
)


logger = logging.getLogger(__name__)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


SQRT2_OVER_2 = np.sqrt(2.0) * 0.5
R_MAP = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
Q_MAP = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float64)
EPS = 1e-8


def quat_safe_normalize(q):
    n = np.linalg.norm(q)
    return q if n < EPS else q / n


def pos_mj2gltf(v3):
    return R_MAP @ np.asarray(v3, dtype=np.float64)


def quat_mj2gltf(q_mj):
    q_mj = quat_safe_normalize(np.asarray(q_mj, dtype=np.float64))
    q_comb = quat_mul(Q_MAP, q_mj)
    q_comb = quat_safe_normalize(q_comb)
    w, x, y, z = q_comb
    return np.array([x, y, z, w], dtype=np.float64)


def quat_wxyz_to_xyzw(q):
    w, x, y, z = q
    return np.array([x, y, z, w], dtype=np.float64)


class Environment(gym.Wrapper):
    def __init__(self, env: gym.Env, *, fps: int = 30):
        super().__init__(env)
        self.builder = GlbSceneBuilder()
        self._prepared = False
        self._is_recording = False
        self._target_fps = fps
        self._sim_time = 0.0
        self._last_frame_time = 0.0
        self._frame_interval = 1.0 / fps
        self._last_data_time = None
        # Get simulation timestep if available
        self._sim_dt = getattr(env.unwrapped.model, "opt", None)
        if self._sim_dt:
            self._sim_dt = self._sim_dt.timestep
        else:
            self._sim_dt = 0.002  # Default MuJoCo timestep

    @contextmanager
    def render_settings(self, fps: int = 30):
        old_fps = self._target_fps
        self._target_fps = fps
        self._frame_interval = 1.0 / fps
        self._is_recording = True
        self._sim_time = 0.0
        self._last_frame_time = 0.0
        self._last_data_time = None
        self.builder.start_animation()
        try:
            yield
        finally:
            self.builder.finish_animation()
            self._is_recording = False
            self._target_fps = old_fps
            self._frame_interval = 1.0 / old_fps

    def prepare(self):
        if self._prepared:
            return
        if not self.env.unwrapped.model:
            return
        model = self.env.unwrapped.model
        # textures ignored for now
        for body_id in range(model.nbody):
            geom_start = model.body_geomadr[body_id]
            for geom_id in range(geom_start, geom_start + model.body_geomnum[body_id]):
                # Include more geometry groups to find the floor plane
                if model.geom_group[geom_id] not in (0, 1, 2):
                    continue
                geom_type = model.geom_type[geom_id]
                geom_pos = model.geom_pos[geom_id]
                geom_quat = quat_wxyz_to_xyzw(model.geom_quat[geom_id])
                color = model.geom_rgba[geom_id].tolist()
                if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    radius = model.geom_size[geom_id][0]
                    vertices, normals, indices = generate_sphere_mesh(radius)
                elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    half_extents = model.geom_size[geom_id].tolist()
                    vertices, normals, indices = generate_box_mesh(half_extents)
                elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                    half_extents = model.geom_size[geom_id][:2].tolist()
                    vertices, normals, indices = generate_plane_mesh(half_extents)
                elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = model.geom_dataid[geom_id]
                    vert_start = model.mesh_vertadr[mesh_id]
                    vert_count = model.mesh_vertnum[mesh_id]
                    face_start = model.mesh_faceadr[mesh_id]
                    face_count = model.mesh_facenum[mesh_id]
                    indices = (
                        model.mesh_face[face_start : face_start + face_count]
                        .ravel()
                        .astype(np.uint32)
                        .tolist()
                    )
                    verts = model.mesh_vert[vert_start : vert_start + vert_count]
                    vertices = verts.tolist()

                    # Check if model has normals
                    if model.nmeshnormal > 0:
                        normals = model.mesh_normal[
                            vert_start : vert_start + vert_count
                        ].tolist()
                    else:
                        # No normals provided, we'll compute them during triangulation
                        normals = []
                        # Triangulate and compute normals
                        from .glb_builder import triangulate_and_compute_normals

                        vertices, normals, indices = triangulate_and_compute_normals(
                            vertices, indices
                        )
                else:
                    continue
                self.builder.add_mesh(
                    body_id,
                    geom_pos.tolist(),
                    geom_quat.tolist(),
                    vertices,
                    normals,
                    indices,
                    color,
                )
        for cam_id in range(model.ncam):
            cam_pos = pos_mj2gltf(model.cam_pos[cam_id])
            cam_quat = quat_wxyz_to_xyzw(model.cam_quat[cam_id])
            yfov = float(np.deg2rad(model.cam_fovy[cam_id]))
            self.builder.add_camera(
                f"camera_{cam_id}",
                cam_pos.tolist(),
                cam_quat.tolist(),
                yfov,
                1.0,
                0.01,
                100.0,
            )
        self._prepared = True

    def render(self):
        if not self._prepared:
            self.prepare()
        if not self.env.unwrapped.model:
            return
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data

        if self._is_recording:
            # Get current simulation time from MuJoCo data
            current_data_time = data.time

            # Only process if this is a new timestep
            if self._last_data_time is None or current_data_time > self._last_data_time:
                self._sim_time = current_data_time
                self._last_data_time = current_data_time

                # Check if enough time has passed for next frame
                if self._sim_time >= self._last_frame_time + self._frame_interval:
                    # Add a new frame at current simulation time
                    self.builder.add_frame(self._sim_time)
                    for body_id in range(model.nbody):
                        pos = pos_mj2gltf(data.xpos[body_id])
                        quat = quat_mj2gltf(data.xquat[body_id])
                        self.builder.add_frame_transform(
                            body_id, pos.tolist(), quat.tolist()
                        )
                    self._last_frame_time = self._sim_time
        else:
            # Single frame capture
            self.builder.start_animation()
            self.builder.add_frame(0.0)
            for body_id in range(model.nbody):
                pos = pos_mj2gltf(data.xpos[body_id])
                quat = quat_mj2gltf(data.xquat[body_id])
                self.builder.add_frame_transform(body_id, pos.tolist(), quat.tolist())
            self.builder.finish_animation()

    def save(self, path: str):
        self.builder.save(path)

    def close(self):
        return self.env.close()
