"""MuJoCo model building and rendering utilities."""

import numpy as np
from typing import Optional, List, Any

from .model_builder import ModelBuilder, RawMesh, quaternion_to_matrix
from .viewer import GLBViewer


def extract_mujoco_geometry(model) -> List[Any]:
    """Extract geometry data from a MuJoCo model.

    Returns:
        List of bodies with their geometries and cameras
    """
    try:
        import mujoco
    except ImportError:
        raise ImportError("MuJoCo is required for extract_mujoco_geometry")

    # Create a dict to group geometries and cameras by body
    body_data = {}

    # Process each body to collect geometries
    for body_id in range(model.nbody):
        geom_start = model.body_geomadr[body_id]
        geom_count = model.body_geomnum[body_id]

        if geom_count == 0:
            continue

        meshes = []

        for geom_idx in range(geom_count):
            geom_id = geom_start + geom_idx
            geom_type = model.geom_type[geom_id]
            geom_size = model.geom_size[geom_id]
            geom_pos = model.geom_pos[geom_id]
            geom_quat = model.geom_quat[geom_id]
            geom_rgba = model.geom_rgba[geom_id]
            geom_group = model.geom_group[geom_id]
            geom_matid = model.geom_matid[geom_id]

            # Only include geoms from groups 0, 1, and 2
            if geom_group > 2:
                continue

            mesh = None
            material_info = None

            # Extract material and texture info for all geometries
            material_info = _extract_material_info(model, geom_matid)

            # Keep geometry in MuJoCo coordinates - root transform handles conversion
            mj_pos = geom_pos
            mj_quat = quat_wxyz_to_xyzw(geom_quat)  # Only format conversion

            # Create transform matrix for this geometry
            transform = np.eye(4)
            transform[:3, :3] = quaternion_to_matrix(mj_quat)
            transform[:3, 3] = mj_pos

            # Generate mesh based on geom type
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                mesh = ModelBuilder.generate_sphere_mesh(
                    radius=geom_size[0], color=geom_rgba
                )
                if mesh and material_info:
                    mesh.material = material_info
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                mesh = ModelBuilder.generate_box_mesh(
                    half_sizes=geom_size[:3], color=geom_rgba
                )
                if mesh and material_info:
                    mesh.material = material_info
            elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                # Planes in MuJoCo are infinite, we'll make a large finite plane
                plane_size = (
                    max(geom_size[0], geom_size[1])
                    if geom_size[0] > 0 and geom_size[1] > 0
                    else 10.0
                )
                mesh = ModelBuilder.generate_plane_mesh(
                    size=plane_size * 2, color=geom_rgba
                )
                if mesh and material_info:
                    mesh.material = material_info
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # Capsule in MuJoCo: geom_size[0] is radius, geom_size[1] is half-length
                mesh = ModelBuilder.generate_capsule_mesh(
                    radius=geom_size[0], half_length=geom_size[1], color=geom_rgba
                )
                if mesh and material_info:
                    mesh.material = material_info
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # Cylinder in MuJoCo: geom_size[0] is radius, geom_size[1] is half-length
                mesh = ModelBuilder.generate_cylinder_mesh(
                    radius=geom_size[0], half_length=geom_size[1], color=geom_rgba
                )
                if mesh and material_info:
                    mesh.material = material_info
            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                # Ellipsoid in MuJoCo: geom_size[0:3] are the three radii
                mesh = ModelBuilder.generate_ellipsoid_mesh(
                    radii=geom_size[:3], color=geom_rgba
                )
                if mesh and material_info:
                    mesh.material = material_info
            elif geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
                # Height field in MuJoCo
                hfield_id = model.geom_dataid[geom_id]
                if hfield_id >= 0:
                    mesh = _generate_hfield_mesh(model, hfield_id, geom_size, geom_rgba)
                    if mesh and material_info:
                        mesh.material = material_info
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
                    faces = model.mesh_face[
                        face_start : face_start + face_count
                    ].reshape(-1, 3)

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
                        vertices=vertices.astype(np.float32),
                        normals=normals.astype(np.float32),
                        indices=faces.flatten().astype(np.uint32),
                        colors=colors.astype(np.float32),
                        material=material_info,
                    )

            if mesh is not None:
                # Store the mesh with its transform and material info
                mesh_with_transform = {
                    "mesh": mesh,
                    "transform": transform,
                    "is_plane": geom_type == mujoco.mjtGeom.mjGEOM_PLANE,
                    "material": material_info,
                }
                meshes.append(mesh_with_transform)

        if meshes:
            if body_id not in body_data:
                body_data[body_id] = {"meshes": [], "cameras": []}
            body_data[body_id]["meshes"].extend(meshes)

    # Process cameras and assign them to bodies
    for cam_id in range(model.ncam):
        cam_body_id = model.cam_bodyid[cam_id]
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

        # Create transform matrix for camera
        transform = np.eye(4)
        transform[:3, :3] = quaternion_to_matrix(quat_wxyz_to_xyzw(cam_quat))
        transform[:3, 3] = cam_pos

        camera_with_transform = {
            "camera": camera_data,
            "transform": transform,
        }

        if cam_body_id not in body_data:
            body_data[cam_body_id] = {"meshes": [], "cameras": []}
        body_data[cam_body_id]["cameras"].append(camera_with_transform)

    # Convert to list format
    bodies = []
    for body_id, data in body_data.items():
        from .model_builder import BodyGeometry

        bodies.append(
            BodyGeometry(
                body_id=body_id,
                meshes=data["meshes"],  # Store meshes with their individual transforms
                cameras=data["cameras"],  # Store cameras with their transforms
                transform=np.eye(4),  # Body-level transform is identity
                is_plane=any(m["is_plane"] for m in data["meshes"]),
            )
        )

    return bodies


def quat_mul(q1, q2):
    """Multiply two quaternions."""
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


def quat_safe_normalize(q):
    """Safely normalize a quaternion."""
    n = np.linalg.norm(q)
    EPS = 1e-8
    return q if n < EPS else q / n


def convert_mujoco_to_gltf_position(pos: np.ndarray) -> np.ndarray:
    """Convert position from MuJoCo to glTF coordinate system."""
    # MuJoCo: Z-up, Y-forward, X-right
    # glTF: Y-up, Z-backward, X-right
    # Transformation matrix from MuJoCo to glTF
    R_MAP = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return R_MAP @ np.asarray(pos, dtype=np.float64)


def convert_mujoco_to_gltf_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from MuJoCo to glTF coordinate system."""
    # MuJoCo quaternion: (w, x, y, z)
    # Apply coordinate system transformation
    Q_MAP = np.array([0.5, -0.5, -0.5, -0.5])

    q_mj = quat_safe_normalize(np.asarray(quat, dtype=np.float64))
    q_comb = quat_mul(Q_MAP, q_mj)
    q_comb = quat_safe_normalize(q_comb)

    # Return in glTF format [x, y, z, w]
    w, x, y, z = q_comb
    return np.array([x, y, z, w])


def quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from (w, x, y, z) to (x, y, z, w) format."""
    w, x, y, z = quat
    return np.array([x, y, z, w])


class MujocoRender:
    """Render MuJoCo models with animation support."""

    def __init__(self, model):
        """Initialize with a MuJoCo model.

        Args:
            model: MuJoCo MjModel instance
        """

        self.model = model
        self.model_builder = ModelBuilder()
        self._animation_context = None
        self._prepared = False

    def prepare(self):
        """Extract static geometry from the MuJoCo model."""
        if self._prepared:
            return

        try:
            import mujoco
        except ImportError:
            raise ImportError("MuJoCo is required for MujocoRender")

        model = self.model

        # Process each body to collect geometries (matching old approach)
        for body_id in range(model.nbody):
            geom_start = model.body_geomadr[body_id]
            geom_count = model.body_geomnum[body_id]

            for geom_idx in range(geom_count):
                geom_id = geom_start + geom_idx

                # Filter by geometry group (match old code)
                if model.geom_group[geom_id] not in (0, 1, 2):
                    continue

                geom_type = model.geom_type[geom_id]
                geom_pos = model.geom_pos[geom_id]
                geom_quat = model.geom_quat[geom_id]
                geom_size = model.geom_size[geom_id]
                color = model.geom_rgba[geom_id].tolist()
                geom_matid = model.geom_matid[geom_id]

                # Extract material info
                material_info = _extract_material_info(model, geom_matid)

                # Only format conversion for quaternions (NOT coordinate conversion)
                formatted_quat = quat_wxyz_to_xyzw(geom_quat)

                # Generate mesh based on geometry type
                vertices, normals, indices = None, None, None

                if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    vertices, normals, indices = self._generate_sphere_mesh(
                        geom_size[0]
                    )
                elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    vertices, normals, indices = self._generate_box_mesh(geom_size[:3])
                elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                    vertices, normals, indices = self._generate_plane_mesh(
                        geom_size[:2]
                    )
                elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    vertices, normals, indices = self._generate_capsule_mesh(
                        geom_size[0], geom_size[1]
                    )
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    vertices, normals, indices = self._generate_cylinder_mesh(
                        geom_size[0], geom_size[1]
                    )
                elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                    vertices, normals, indices = self._generate_ellipsoid_mesh(
                        geom_size[:3]
                    )
                elif geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
                    vertices, normals, indices = self._generate_hfield_mesh_data(
                        model, model.geom_dataid[geom_id], geom_size
                    )
                elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    vertices, normals, indices = self._extract_mesh_data(model, geom_id)

                if vertices is not None:
                    # Add individual mesh - keep everything in MuJoCo coordinates
                    self.model_builder.add_mesh(
                        body_id,
                        geom_pos.tolist(),  # Keep MuJoCo coordinates
                        formatted_quat.tolist(),  # Keep MuJoCo coordinates, just format conversion
                        vertices,
                        normals,
                        indices,
                        color,
                        material_info,  # Pass material info
                    )

        # Add cameras to their respective bodies
        for cam_id in range(model.ncam):
            cam_body_id = model.cam_bodyid[cam_id]
            cam_pos = model.cam_pos[cam_id]  # Keep MuJoCo coordinates
            cam_quat = quat_wxyz_to_xyzw(
                model.cam_quat[cam_id]
            )  # Only format conversion
            cam_fovy = model.cam_fovy[cam_id]

            # Convert quaternion to rotation matrix for camera orientation
            cam_mat = quaternion_to_matrix(cam_quat)
            cam_forward = -cam_mat[2]
            cam_up = cam_mat[1]

            from .model_builder import CameraData

            camera_data = CameraData(
                position=cam_pos.copy(),
                target=cam_pos + cam_forward,
                up=cam_up.copy(),
                fov=np.radians(cam_fovy),
            )

            # Create transform matrix for camera
            transform = np.eye(4)
            transform[:3, :3] = quaternion_to_matrix(cam_quat)
            transform[:3, 3] = cam_pos

            # Add camera to the body by creating a mesh entry for it
            self.model_builder.add_camera(
                cam_body_id,
                cam_pos.tolist(),
                cam_quat.tolist(),
                camera_data,
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

                # Keep in MuJoCo coordinates - root transform handles conversion
                formatted_quat = quat_wxyz_to_xyzw(quat)

                # Update body transform
                self.model_builder.update_body_transform(body_id, pos, formatted_quat)
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

            # Keep in MuJoCo coordinates - root transform handles conversion
            formatted_quat = quat_wxyz_to_xyzw(quat)

            transforms[body_id] = {
                "translation": pos.tolist(),
                "rotation": formatted_quat.tolist(),
            }

        self.model_builder.add_animation_frame(time, transforms)

    def _generate_sphere_mesh(self, radius: float):
        """Generate sphere mesh matching old format."""
        vertices = []
        normals = []
        indices = []
        longitude_segments = 16
        latitude_segments = 16

        for lat in range(latitude_segments + 1):
            theta = lat * np.pi / latitude_segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for lon in range(longitude_segments + 1):
                phi = lon * 2.0 * np.pi / longitude_segments
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
                x = cos_phi * sin_theta
                y = sin_phi * sin_theta
                z = cos_theta
                vertices.append([x * radius, y * radius, z * radius])
                normals.append([x, y, z])

        for lat in range(latitude_segments):
            for lon in range(longitude_segments):
                first = lat * (longitude_segments + 1) + lon
                second = first + longitude_segments + 1
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        return vertices, normals, indices

    def _generate_box_mesh(self, half_extents):
        """Generate box mesh matching old format."""
        hx, hy, hz = half_extents
        corners = [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ]
        faces = [
            ([0, 1, 2, 3], [0, 0, -1]),
            ([5, 4, 7, 6], [0, 0, 1]),
            ([4, 0, 3, 7], [-1, 0, 0]),
            ([1, 5, 6, 2], [1, 0, 0]),
            ([0, 4, 5, 1], [0, -1, 0]),
            ([3, 2, 6, 7], [0, 1, 0]),
        ]
        vertices = []
        normals = []
        indices = []
        idx = 0
        for face_indices, normal in faces:
            for i in face_indices:
                vertices.append(corners[i])
                normals.append(normal)
            indices.extend([idx, idx + 2, idx + 1])
            indices.extend([idx, idx + 3, idx + 2])
            idx += 4
        return vertices, normals, indices

    def _generate_plane_mesh(self, half_extents):
        """Generate plane mesh matching old format."""
        hx, hy = half_extents
        if hx <= 0:
            hx = 1.0
        if hy <= 0:
            hy = 1.0
        vertices = [
            [-hx, -hy, 0.0],
            [hx, -hy, 0.0],
            [hx, hy, 0.0],
            [-hx, hy, 0.0],
        ]
        normals = [[0.0, 0.0, 1.0]] * 4
        indices = [0, 2, 1, 0, 3, 2]
        return vertices, normals, indices

    def _extract_mesh_data(self, model, geom_id):
        """Extract mesh data from MuJoCo model."""
        geom_dataid = model.geom_dataid[geom_id]
        if geom_dataid < 0:
            return None, None, None

        mesh_id = geom_dataid
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

        # Compute normals using triangulation
        if model.nmeshnormal > 0:
            normals = model.mesh_normal[vert_start : vert_start + vert_count].tolist()
        else:
            # Use the triangulation function from old code
            from .model_builder import triangulate_and_compute_normals

            vertices, normals, indices = triangulate_and_compute_normals(
                vertices, indices
            )

        return vertices, normals, indices

    def _generate_capsule_mesh(self, radius: float, half_length: float):
        """Generate capsule mesh matching old format."""
        vertices = []
        normals = []
        indices = []
        rings = 8
        sectors = 16

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
            # Connect top hemisphere to cylinder
            hem_idx = rings * (sectors + 1) + s
            cyl_idx = top_cap_end + s

            indices.extend([hem_idx, hem_idx + 1, cyl_idx])
            indices.extend([hem_idx + 1, cyl_idx + 1, cyl_idx])

            # Connect cylinder sides - flip winding for Z-axis orientation
            top_idx = top_cap_end + s
            bot_idx = top_cap_end + (sectors + 1) + s

            indices.extend([top_idx, bot_idx, top_idx + 1])
            indices.extend([top_idx + 1, bot_idx, bot_idx + 1])

            # Connect cylinder to bottom hemisphere
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

        return vertices, normals, indices

    def _generate_cylinder_mesh(self, radius: float, half_length: float):
        """Generate cylinder mesh."""
        vertices = []
        normals = []
        indices = []
        sectors = 32

        # Generate cylinder body
        for z_idx, z in enumerate([half_length, -half_length]):
            for s in range(sectors + 1):
                theta = 2 * np.pi * s / sectors
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), np.sin(theta), 0])

        # Generate cylinder side indices
        for s in range(sectors):
            # Top ring vertex index
            top = s
            # Bottom ring vertex index
            bottom = (sectors + 1) + s

            # Two triangles per quad
            indices.extend([top, bottom, top + 1])
            indices.extend([top + 1, bottom, bottom + 1])

        # Generate top cap
        center_top = len(vertices)
        vertices.append([0, 0, half_length])
        normals.append([0, 0, 1])

        for s in range(sectors):
            theta = 2 * np.pi * s / sectors
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, half_length])
            normals.append([0, 0, 1])

        # Top cap indices
        for s in range(sectors):
            indices.extend(
                [center_top, center_top + s + 1, center_top + ((s + 1) % sectors) + 1]
            )

        # Generate bottom cap
        center_bottom = len(vertices)
        vertices.append([0, 0, -half_length])
        normals.append([0, 0, -1])

        for s in range(sectors):
            theta = 2 * np.pi * s / sectors
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, -half_length])
            normals.append([0, 0, -1])

        # Bottom cap indices (reversed winding)
        for s in range(sectors):
            indices.extend(
                [
                    center_bottom,
                    center_bottom + ((s + 1) % sectors) + 1,
                    center_bottom + s + 1,
                ]
            )

        return vertices, normals, indices

    def _generate_ellipsoid_mesh(self, radii):
        """Generate ellipsoid mesh."""
        vertices = []
        normals = []
        indices = []
        longitude_segments = 32
        latitude_segments = 16
        rx, ry, rz = radii

        # Generate vertices and normals
        for lat in range(latitude_segments + 1):
            theta = lat * np.pi / latitude_segments
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for lon in range(longitude_segments + 1):
                phi = lon * 2.0 * np.pi / longitude_segments
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                # Unit sphere coordinates
                x = cos_phi * sin_theta
                y = sin_phi * sin_theta
                z = cos_theta

                # Scale by radii to get ellipsoid
                vertices.append([x * rx, y * ry, z * rz])

                # Normal is the normalized gradient of the ellipsoid equation
                # For ellipsoid: (x/rx)^2 + (y/ry)^2 + (z/rz)^2 = 1
                # Gradient: [2x/rx^2, 2y/ry^2, 2z/rz^2]
                normal = np.array([x / (rx * rx), y / (ry * ry), z / (rz * rz)])
                normal = normal / np.linalg.norm(normal)
                normals.append(normal.tolist())

        # Generate indices
        for lat in range(latitude_segments):
            for lon in range(longitude_segments):
                first = lat * (longitude_segments + 1) + lon
                second = first + longitude_segments + 1

                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])

        return vertices, normals, indices

    def _generate_hfield_mesh_data(self, model, hfield_id, geom_size):
        """Generate height field mesh data."""
        if hfield_id < 0:
            return None, None, None

        # Get height field dimensions
        nrow = model.hfield_nrow[hfield_id]
        ncol = model.hfield_ncol[hfield_id]
        size = model.hfield_size[hfield_id]  # [x_size, y_size, z_base, z_height]
        data_adr = model.hfield_adr[hfield_id]

        # Get height data
        hfield_data = model.hfield_data[data_adr : data_adr + nrow * ncol]
        heights = hfield_data.reshape(nrow, ncol)

        # Generate vertices and normals
        vertices = []
        normals = []
        indices = []

        # Create vertex grid
        for i in range(nrow):
            for j in range(ncol):
                # Calculate position
                x = (j / (ncol - 1) - 0.5) * size[0] * 2
                y = (i / (nrow - 1) - 0.5) * size[1] * 2
                z = size[2] + heights[i, j] * size[3]
                vertices.append([x, y, z])

        # Calculate normals using central differences
        for i in range(nrow):
            for j in range(ncol):
                # Get neighboring heights for normal calculation
                h_left = heights[i, max(0, j - 1)]
                h_right = heights[i, min(ncol - 1, j + 1)]
                h_up = heights[max(0, i - 1), j]
                h_down = heights[min(nrow - 1, i + 1), j]

                # Calculate gradients
                dx = ((h_right - h_left) * size[3]) / (2 * size[0] / (ncol - 1))
                dy = ((h_down - h_up) * size[3]) / (2 * size[1] / (nrow - 1))

                # Normal is (-dx, -dy, 1) normalized
                normal = np.array([-dx, -dy, 1.0])
                normal = normal / np.linalg.norm(normal)
                normals.append(normal.tolist())

        # Generate indices for triangles
        for i in range(nrow - 1):
            for j in range(ncol - 1):
                # Get vertex indices
                v0 = i * ncol + j
                v1 = i * ncol + (j + 1)
                v2 = (i + 1) * ncol + j
                v3 = (i + 1) * ncol + (j + 1)

                # Create two triangles per quad
                indices.extend([v0, v1, v2])
                indices.extend([v1, v3, v2])

        return vertices, normals, indices

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
        self._last_saved_path = filename

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        # Save to a temporary file if not already saved
        if not hasattr(self, "_last_saved_path"):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                temp_path = f.name
            self.save(temp_path)
            self._last_saved_path = temp_path

        # Create a GLBViewer and return its HTML representation
        viewer = GLBViewer(self._last_saved_path)
        return viewer._repr_html_()


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


def _generate_hfield_mesh(model, hfield_id, geom_size, color):
    """Generate height field mesh for extract_mujoco_geometry."""
    if hfield_id < 0:
        return None

    # Get height field dimensions
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    size = model.hfield_size[hfield_id]  # [x_size, y_size, z_base, z_height]
    data_adr = model.hfield_adr[hfield_id]

    # Get height data
    hfield_data = model.hfield_data[data_adr : data_adr + nrow * ncol]
    heights = hfield_data.reshape(nrow, ncol)

    # Generate vertices and normals
    vertices = []
    normals = []
    indices = []

    # Create vertex grid
    for i in range(nrow):
        for j in range(ncol):
            # Calculate position
            x = (j / (ncol - 1) - 0.5) * size[0] * 2
            y = (i / (nrow - 1) - 0.5) * size[1] * 2
            z = size[2] + heights[i, j] * size[3]
            vertices.append([x, y, z])

    # Calculate normals using central differences
    for i in range(nrow):
        for j in range(ncol):
            # Get neighboring heights for normal calculation
            h_left = heights[i, max(0, j - 1)]
            h_right = heights[i, min(ncol - 1, j + 1)]
            h_up = heights[max(0, i - 1), j]
            h_down = heights[min(nrow - 1, i + 1), j]

            # Calculate gradients
            dx = ((h_right - h_left) * size[3]) / (2 * size[0] / (ncol - 1))
            dy = ((h_down - h_up) * size[3]) / (2 * size[1] / (nrow - 1))

            # Normal is (-dx, -dy, 1) normalized
            normal = np.array([-dx, -dy, 1.0])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal.tolist())

    # Generate indices for triangles
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            # Get vertex indices
            v0 = i * ncol + j
            v1 = i * ncol + (j + 1)
            v2 = (i + 1) * ncol + j
            v3 = (i + 1) * ncol + (j + 1)

            # Create two triangles per quad
            indices.extend([v0, v1, v2])
            indices.extend([v1, v3, v2])

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    colors = np.tile(color, (len(vertices), 1))

    return RawMesh(
        vertices=vertices,
        normals=normals,
        indices=indices,
        colors=colors,
    )


def _extract_material_info(model, geom_matid):
    """Extract material and texture information from MuJoCo model.

    Args:
        model: MuJoCo model
        geom_matid: Material ID for the geometry (-1 if no material)

    Returns:
        Dict with material info or None if no material
    """
    if geom_matid < 0 or geom_matid >= model.nmat:
        return None

    try:
        import mujoco
    except ImportError:
        return None

    material_info = {
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, geom_matid),
        "rgba": model.mat_rgba[geom_matid].copy(),
        "texrepeat": model.mat_texrepeat[geom_matid].copy(),
        "texuniform": bool(model.mat_texuniform[geom_matid]),
        "textures": {},
    }

    # Extract texture information for each texture type
    # mat_texid[matid] contains texture IDs for different texture types
    # Index 0 is usually the diffuse texture
    for tex_type, tex_id in enumerate(model.mat_texid[geom_matid]):
        if tex_id >= 0 and tex_id < model.ntex:
            tex_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_id)

            # Extract texture data
            tex_width = model.tex_width[tex_id]
            tex_height = model.tex_height[tex_id]
            tex_type_val = model.tex_type[tex_id]
            tex_adr = model.tex_adr[tex_id]
            tex_nchannel = model.tex_nchannel[tex_id]

            # Get texture data
            tex_size = tex_width * tex_height * tex_nchannel
            tex_data = model.tex_data[tex_adr : tex_adr + tex_size]

            # Reshape to proper format (height, width, channels)
            tex_data = tex_data.reshape(tex_height, tex_width, tex_nchannel)

            material_info["textures"][tex_type] = {
                "name": tex_name,
                "width": tex_width,
                "height": tex_height,
                "type": tex_type_val,
                "channels": tex_nchannel,
                "data": tex_data.copy(),
            }

    return material_info
