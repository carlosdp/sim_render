"""GLB export functionality."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pygltflib


def export_to_glb(model_builder, filename: str):
    """Export a ModelBuilder to GLB format.

    Args:
        model_builder: ModelBuilder instance with scene data
        filename: Output filename
    """
    gltf = pygltflib.GLTF2()
    gltf.scene = 0
    gltf.scenes = [pygltflib.Scene(nodes=[])]

    # Binary buffer for all data
    binary_data = bytearray()

    # Get encoded bodies
    bodies = model_builder.get_encoded_bodies()

    # Process each body
    for body_data in bodies:
        body_id = body_data["body_id"]
        meshes_data = body_data["meshes"]
        transform = body_data["transform"]
        is_plane = body_data.get("is_plane", False)

        # Create a node for this body
        node = pygltflib.Node(name=f"body_{body_id}")

        # Set transform
        position = transform[:3, 3]
        rotation_matrix = transform[:3, :3]

        # Convert rotation matrix to quaternion
        quaternion = matrix_to_quaternion(rotation_matrix)

        node.translation = position.tolist()
        node.rotation = quaternion.tolist()

        # Process meshes
        mesh_indices = []
        for mesh_data in meshes_data:
            mesh_index = _add_mesh_to_gltf(gltf, binary_data, mesh_data, is_plane)
            mesh_indices.append(mesh_index)

        # If multiple meshes, create child nodes
        if len(mesh_indices) == 1:
            node.mesh = mesh_indices[0]
        else:
            node.children = []
            for i, mesh_idx in enumerate(mesh_indices):
                child_node = pygltflib.Node(
                    name=f"body_{body_id}_mesh_{i}", mesh=mesh_idx
                )
                child_idx = len(gltf.nodes)
                gltf.nodes.append(child_node)
                node.children.append(child_idx)

        # Add node to scene
        node_idx = len(gltf.nodes)
        gltf.nodes.append(node)
        gltf.scenes[0].nodes.append(node_idx)

    # Add camera if present
    if model_builder.camera:
        _add_camera_to_gltf(gltf, model_builder.camera)

    # Handle animations
    if model_builder.animation_frames:
        _add_animation_to_gltf(
            gltf, binary_data, model_builder.animation_frames, bodies
        )

    # Set up the binary buffer
    if binary_data:
        gltf.buffers = [pygltflib.Buffer(byteLength=len(binary_data))]
        gltf.set_binary_blob(bytes(binary_data))

    # Save as GLB
    gltf.save_binary(filename)


def _add_mesh_to_gltf(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    mesh_data: Dict[str, Any],
    is_plane: bool,
) -> int:
    """Add a mesh to the glTF scene.

    Returns:
        Mesh index in the glTF
    """
    vertices = mesh_data["vertices"]
    normals = mesh_data["normals"]
    indices = mesh_data["indices"]
    colors = mesh_data.get("colors")

    # Create accessors for vertex data
    vertex_accessor = _add_accessor(
        gltf,
        binary_data,
        vertices,
        pygltflib.FLOAT,
        pygltflib.VEC3,
        vertices.min(axis=0).tolist(),
        vertices.max(axis=0).tolist(),
    )

    normal_accessor = _add_accessor(
        gltf, binary_data, normals, pygltflib.FLOAT, pygltflib.VEC3
    )

    index_accessor = _add_accessor(
        gltf, binary_data, indices, pygltflib.UNSIGNED_INT, pygltflib.SCALAR
    )

    attributes = {"POSITION": vertex_accessor, "NORMAL": normal_accessor}

    # Add colors if present
    if colors is not None:
        color_accessor = _add_accessor(
            gltf, binary_data, colors, pygltflib.FLOAT, pygltflib.VEC4
        )
        attributes["COLOR_0"] = color_accessor

    # Create primitive
    primitive = pygltflib.Primitive(
        attributes=attributes, indices=index_accessor, mode=pygltflib.TRIANGLES
    )

    # Set material
    if colors is not None:
        # Create material with vertex colors
        material = pygltflib.Material(
            name="vertex_color_material",
            doubleSided=is_plane,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.5,
            ),
        )
        material_idx = len(gltf.materials)
        gltf.materials.append(material)
        primitive.material = material_idx

    # Create mesh
    mesh = pygltflib.Mesh(primitives=[primitive])
    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(mesh)

    return mesh_idx


def _add_accessor(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    data: np.ndarray,
    component_type: int,
    type_: str,
    min_vals: Optional[List[float]] = None,
    max_vals: Optional[List[float]] = None,
) -> int:
    """Add an accessor for array data.

    Returns:
        Accessor index
    """
    # Ensure data is contiguous and has the right dtype
    if component_type == pygltflib.FLOAT:
        data = np.ascontiguousarray(data, dtype=np.float32)
    elif component_type == pygltflib.UNSIGNED_INT:
        data = np.ascontiguousarray(data, dtype=np.uint32)

    # Add to binary buffer
    byte_offset = len(binary_data)
    binary_data.extend(data.tobytes())

    # Create buffer view
    buffer_view = pygltflib.BufferView(
        buffer=0, byteOffset=byte_offset, byteLength=data.nbytes
    )
    view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(buffer_view)

    # Create accessor
    accessor = pygltflib.Accessor(
        bufferView=view_idx,
        byteOffset=0,
        componentType=component_type,
        count=len(data) if data.ndim == 1 else len(data),
        type=type_,
    )

    if min_vals is not None:
        accessor.min = min_vals
    if max_vals is not None:
        accessor.max = max_vals

    accessor_idx = len(gltf.accessors)
    gltf.accessors.append(accessor)

    return accessor_idx


def _add_camera_to_gltf(gltf: pygltflib.GLTF2, camera_data):
    """Add a camera to the glTF scene."""
    # Create camera
    camera = pygltflib.Camera(
        type="perspective",
        perspective=pygltflib.Perspective(yfov=camera_data.fov, znear=0.1),
    )
    camera_idx = len(gltf.cameras)
    gltf.cameras.append(camera)

    # Create camera node
    camera_node = pygltflib.Node(name="camera", camera=camera_idx)

    # Set camera transform using look-at
    position = camera_data.position
    target = camera_data.target
    up = camera_data.up

    # Calculate camera orientation
    forward = target - position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    # Build rotation matrix (camera looks down -Z)
    rotation_matrix = np.array(
        [
            [right[0], up[0], -forward[0]],
            [right[1], up[1], -forward[1]],
            [right[2], up[2], -forward[2]],
        ]
    )

    quaternion = matrix_to_quaternion(rotation_matrix)

    camera_node.translation = position.tolist()
    camera_node.rotation = quaternion.tolist()

    # Add to scene
    node_idx = len(gltf.nodes)
    gltf.nodes.append(camera_node)
    gltf.scenes[0].nodes.append(node_idx)


def _add_animation_to_gltf(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    animation_frames: List[Tuple[float, Dict]],
    bodies: List[Dict],
):
    """Add animation data to the glTF."""
    if not animation_frames:
        return

    animation = pygltflib.Animation(name="animation", channels=[], samplers=[])

    # Group frames by body
    body_frames = {}
    for time, transforms in animation_frames:
        for body_id, transform in transforms.items():
            if body_id not in body_frames:
                body_frames[body_id] = {
                    "times": [],
                    "translations": [],
                    "rotations": [],
                }

            body_frames[body_id]["times"].append(time)
            body_frames[body_id]["translations"].append(transform["translation"])
            body_frames[body_id]["rotations"].append(transform["rotation"])

    # Create channels and samplers for each animated body
    for body_id, frames in body_frames.items():
        # Find node index for this body
        node_idx = None
        for i, body_data in enumerate(bodies):
            if body_data["body_id"] == body_id:
                node_idx = i  # Assuming direct mapping
                break

        if node_idx is None:
            continue

        # Time accessor (shared for translation and rotation)
        times = np.array(frames["times"], dtype=np.float32)
        time_accessor = _add_accessor(
            gltf,
            binary_data,
            times,
            pygltflib.FLOAT,
            pygltflib.SCALAR,
            [float(times.min())],
            [float(times.max())],
        )

        # Translation
        if frames["translations"]:
            translations = np.array(frames["translations"], dtype=np.float32)
            trans_accessor = _add_accessor(
                gltf, binary_data, translations, pygltflib.FLOAT, pygltflib.VEC3
            )

            sampler_idx = len(animation.samplers)
            animation.samplers.append(
                pygltflib.AnimationSampler(
                    input=time_accessor,
                    output=trans_accessor,
                    interpolation=pygltflib.LINEAR,
                )
            )

            animation.channels.append(
                pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(
                        node=node_idx, path=pygltflib.TRANSLATION
                    ),
                )
            )

        # Rotation
        if frames["rotations"]:
            rotations = np.array(frames["rotations"], dtype=np.float32)
            rot_accessor = _add_accessor(
                gltf, binary_data, rotations, pygltflib.FLOAT, pygltflib.VEC4
            )

            sampler_idx = len(animation.samplers)
            animation.samplers.append(
                pygltflib.AnimationSampler(
                    input=time_accessor,
                    output=rot_accessor,
                    interpolation=pygltflib.LINEAR,
                )
            )

            animation.channels.append(
                pygltflib.AnimationChannel(
                    sampler=sampler_idx,
                    target=pygltflib.AnimationChannelTarget(
                        node=node_idx, path=pygltflib.ROTATION
                    ),
                )
            )

    if animation.channels:
        gltf.animations.append(animation)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion.

    Args:
        matrix: 3x3 rotation matrix

    Returns:
        Quaternion as [x, y, z, w] (glTF format)
    """
    # Based on "Converting a Rotation Matrix to a Quaternion" by Mike Day
    m = matrix

    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2], m[1, 2] - m[2, 1]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1], m[2, 0] - m[0, 2]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t, m[0, 1] - m[1, 0]]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0], t]

    q = np.array(q) * 0.5 / np.sqrt(t)

    # Return in glTF format [x, y, z, w]
    return np.array([q[0], q[1], q[2], q[3]])
