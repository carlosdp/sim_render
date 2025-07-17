import struct
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from pygltflib import (
    GLTF2,
    Scene,
    Node,
    Mesh,
    Primitive,
    Buffer,
    BufferView,
    Accessor,
    Asset,
    Material,
    PbrMetallicRoughness,
    Animation,
    AnimationSampler,
    AnimationChannel,
    AnimationChannelTarget,
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_INT,
    VEC3,
    VEC4,
    SCALAR,
)


@dataclass
class RawMesh:
    vertices: List[List[float]]
    normals: List[List[float]]
    indices: List[int]
    color: List[float]


@dataclass
class BodyGeometry:
    body_id: int
    position: List[float]
    rotation: List[float]
    mesh: RawMesh


@dataclass
class CameraData:
    name: str
    position: List[float]
    rotation: List[float]
    yfov: float
    aspect_ratio: float
    znear: float
    zfar: float


@dataclass
class GlbSceneBuilder:
    bodies: List[BodyGeometry] = field(default_factory=list)
    cameras: List[CameraData] = field(default_factory=list)
    frames: List[Dict[int, Tuple[List[float], List[float]]]] = field(
        default_factory=list
    )
    frame_times: List[float] = field(default_factory=list)
    recording: bool = False

    def add_mesh(
        self,
        body_id: int,
        position: List[float],
        rotation: List[float],
        vertices: List[List[float]],
        normals: List[List[float]],
        indices: List[int],
        color: List[float],
    ) -> None:
        mesh = RawMesh(vertices, normals, indices, color)
        self.bodies.append(BodyGeometry(body_id, position, rotation, mesh))

    def add_camera(
        self,
        name: str,
        position: List[float],
        rotation: List[float],
        yfov: float,
        aspect_ratio: float,
        znear: float,
        zfar: float,
    ) -> None:
        self.cameras.append(
            CameraData(name, position, rotation, yfov, aspect_ratio, znear, zfar)
        )

    def start_animation(self) -> None:
        self.frames = []
        self.frame_times = []
        self.recording = True

    def add_frame_transform(
        self, body_id: int, position: List[float], rotation: List[float]
    ) -> None:
        if not self.recording:
            return
        if not self.frames:
            return  # Must call add_frame first
        self.frames[-1][body_id] = (position, rotation)

    def add_frame(self, time: float) -> None:
        if self.recording:
            self.frames.append({})
            self.frame_times.append(time)

    def finish_animation(self) -> None:
        self.recording = False
        while self.frames and not self.frames[-1]:
            self.frames.pop()

    def _push_data(self, buffer: bytearray, data: bytes) -> Tuple[int, int]:
        offset = len(buffer)
        buffer.extend(data)
        while len(buffer) % 4:
            buffer.append(0)
        return offset, len(data)

    def _pack_f32(self, values: List[float]) -> bytes:
        return struct.pack("<%sf" % len(values), *values)

    def _pack_u32(self, values: List[int]) -> bytes:
        return struct.pack("<%sI" % len(values), *values)

    def build_gltf(self) -> GLTF2:
        gltf = GLTF2(asset=Asset(version="2.0"))
        scene = Scene(nodes=[])
        gltf.scenes = [scene]
        gltf.scene = 0

        gltf.nodes = []
        gltf.meshes = []
        gltf.materials = []
        gltf.buffers = [Buffer(byteLength=0)]
        gltf.bufferViews = []
        gltf.accessors = []
        gltf.animations = []

        buffer = bytearray()

        body_nodes: Dict[int, int] = {}

        # group geometries by body
        bodies_map: Dict[int, List[BodyGeometry]] = {}
        for geom in self.bodies:
            bodies_map.setdefault(geom.body_id, []).append(geom)

        for body_id, geoms in bodies_map.items():
            body_node = Node(
                name=f"body_{body_id}",
                translation=[0, 0, 0],
                rotation=[0, 0, 0, 1],
                children=[],
            )
            gltf.nodes.append(body_node)
            body_index = len(gltf.nodes) - 1
            scene.nodes.append(body_index)
            body_nodes[body_id] = body_index

            for geom in geoms:
                # create buffers
                vertices = [c for v in geom.mesh.vertices for c in v]
                normals = [c for n in geom.mesh.normals for c in n]
                indices = geom.mesh.indices

                pos_bytes = self._pack_f32(vertices)
                pos_off, pos_len = self._push_data(buffer, pos_bytes)
                pos_view = BufferView(
                    buffer=0,
                    byteOffset=pos_off,
                    byteLength=pos_len,
                    target=ARRAY_BUFFER,
                )
                gltf.bufferViews.append(pos_view)
                pos_accessor = Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    byteOffset=0,
                    componentType=FLOAT,
                    count=len(geom.mesh.vertices),
                    type=VEC3,
                    max=[float(max(c)) for c in zip(*geom.mesh.vertices)],
                    min=[float(min(c)) for c in zip(*geom.mesh.vertices)],
                )
                gltf.accessors.append(pos_accessor)
                pos_accessor_index = len(gltf.accessors) - 1

                if geom.mesh.normals:
                    norm_bytes = self._pack_f32(normals)
                    norm_off, norm_len = self._push_data(buffer, norm_bytes)
                    norm_view = BufferView(
                        buffer=0,
                        byteOffset=norm_off,
                        byteLength=norm_len,
                        target=ARRAY_BUFFER,
                    )
                    gltf.bufferViews.append(norm_view)
                    norm_accessor = Accessor(
                        bufferView=len(gltf.bufferViews) - 1,
                        byteOffset=0,
                        componentType=FLOAT,
                        count=len(geom.mesh.normals),
                        type=VEC3,
                    )
                    gltf.accessors.append(norm_accessor)
                    norm_accessor_index = len(gltf.accessors) - 1
                else:
                    norm_accessor_index = None

                idx_bytes = self._pack_u32(indices)
                idx_off, idx_len = self._push_data(buffer, idx_bytes)
                idx_view = BufferView(
                    buffer=0,
                    byteOffset=idx_off,
                    byteLength=idx_len,
                    target=ELEMENT_ARRAY_BUFFER,
                )
                gltf.bufferViews.append(idx_view)
                idx_accessor = Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    byteOffset=0,
                    componentType=UNSIGNED_INT,
                    count=len(indices),
                    type=SCALAR,
                )
                gltf.accessors.append(idx_accessor)
                idx_accessor_index = len(gltf.accessors) - 1

                # Check if this is a plane (all Z coordinates near 0)
                is_plane = all(abs(v[2]) < 0.001 for v in geom.mesh.vertices)

                material = Material(
                    pbrMetallicRoughness=PbrMetallicRoughness(
                        baseColorFactor=geom.mesh.color
                    ),
                    doubleSided=is_plane,  # Make planes double-sided
                )
                gltf.materials.append(material)
                mat_index = len(gltf.materials) - 1

                prim = Primitive(
                    attributes={"POSITION": pos_accessor_index},
                    indices=idx_accessor_index,
                    material=mat_index,
                )
                if norm_accessor_index is not None:
                    prim.attributes["NORMAL"] = norm_accessor_index

                mesh = Mesh(primitives=[prim])
                gltf.meshes.append(mesh)
                mesh_index = len(gltf.meshes) - 1

                child_node = Node(
                    mesh=mesh_index,
                    translation=geom.position,
                    rotation=geom.rotation,
                )
                gltf.nodes.append(child_node)
                child_index = len(gltf.nodes) - 1
                gltf.nodes[body_index].children.append(child_index)

        # Animations
        if self.frames and self.frame_times:
            time_bytes = self._pack_f32(self.frame_times)
            t_off, t_len = self._push_data(buffer, time_bytes)
            t_view = BufferView(
                buffer=0, byteOffset=t_off, byteLength=t_len, target=ARRAY_BUFFER
            )
            gltf.bufferViews.append(t_view)
            t_accessor = Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                byteOffset=0,
                componentType=FLOAT,
                count=len(self.frame_times),
                type=SCALAR,
                max=[float(self.frame_times[-1]) if self.frame_times else 0.0],
                min=[0.0],
            )
            gltf.accessors.append(t_accessor)
            time_index = len(gltf.accessors) - 1

            animation = Animation(name="Animation", channels=[], samplers=[])
            for body_id, node_index in body_nodes.items():
                translations = []
                rotations = []
                for frame in self.frames:
                    pos, rot = frame.get(body_id, ([0, 0, 0], [0, 0, 0, 1]))
                    translations.extend(pos)
                    rotations.extend(rot)

                trans_bytes = self._pack_f32(translations)
                trans_off, trans_len = self._push_data(buffer, trans_bytes)
                trans_view = BufferView(
                    buffer=0,
                    byteOffset=trans_off,
                    byteLength=trans_len,
                    target=ARRAY_BUFFER,
                )
                gltf.bufferViews.append(trans_view)
                trans_accessor = Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    byteOffset=0,
                    componentType=FLOAT,
                    count=len(self.frames),
                    type=VEC3,
                )
                gltf.accessors.append(trans_accessor)
                trans_index = len(gltf.accessors) - 1

                rot_bytes = self._pack_f32(rotations)
                rot_off, rot_len = self._push_data(buffer, rot_bytes)
                rot_view = BufferView(
                    buffer=0,
                    byteOffset=rot_off,
                    byteLength=rot_len,
                    target=ARRAY_BUFFER,
                )
                gltf.bufferViews.append(rot_view)
                rot_accessor = Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    byteOffset=0,
                    componentType=FLOAT,
                    count=len(self.frames),
                    type=VEC4,
                )
                gltf.accessors.append(rot_accessor)
                rot_index = len(gltf.accessors) - 1

                s_trans = AnimationSampler(
                    input=time_index, output=trans_index, interpolation="LINEAR"
                )
                animation.samplers.append(s_trans)
                sampler_index = len(animation.samplers) - 1
                animation.channels.append(
                    AnimationChannel(
                        sampler=sampler_index,
                        target=AnimationChannelTarget(
                            node=node_index, path="translation"
                        ),
                    )
                )
                s_rot = AnimationSampler(
                    input=time_index, output=rot_index, interpolation="LINEAR"
                )
                animation.samplers.append(s_rot)
                sampler_index = len(animation.samplers) - 1
                animation.channels.append(
                    AnimationChannel(
                        sampler=sampler_index,
                        target=AnimationChannelTarget(node=node_index, path="rotation"),
                    )
                )
            gltf.animations.append(animation)

        gltf.buffers[0].byteLength = len(buffer)
        gltf.set_binary_blob(bytes(buffer))
        return gltf

    def save(self, path: str) -> None:
        gltf = self.build_gltf()
        gltf.save_binary(path)


def generate_sphere_mesh(
    radius: float, longitude_segments: int = 16, latitude_segments: int = 16
):
    vertices = []
    normals = []
    indices = []
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


def generate_box_mesh(half_extents):
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
        indices.extend([idx, idx + 1, idx + 2])
        indices.extend([idx, idx + 2, idx + 3])
        idx += 4
    return vertices, normals, indices


def generate_plane_mesh(half_extents):
    hx, hy = half_extents
    # Ensure we have valid dimensions
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
