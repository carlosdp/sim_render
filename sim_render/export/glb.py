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

    # Create root transformation node for MuJoCo→GLB coordinate conversion
    # MuJoCo: X-right, Y-forward, Z-up
    # GLB: X-right, Y-up, Z-backward
    # This is equivalent to rotating 90° around X-axis, then 180° around Y-axis
    root_transform_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    root_quaternion = matrix_to_quaternion(root_transform_matrix)

    root_node = pygltflib.Node(
        name="mujoco_to_gltf_root", rotation=root_quaternion.tolist(), children=[]
    )

    gltf.nodes = [root_node]
    gltf.scenes[0].nodes = [0]  # Scene contains only the root node

    # Get encoded bodies
    bodies = model_builder.get_encoded_bodies()

    # Track body node indices for animation
    body_node_indices = {}

    # Process each body
    for body_data in bodies:
        body_id = body_data["body_id"]
        meshes_data = body_data["meshes"]
        body_transform = body_data["transform"]

        # Create a node for this body
        body_node = pygltflib.Node(name=f"body_{body_id}")

        # Set body transform
        position = body_transform[:3, 3]
        rotation_matrix = body_transform[:3, :3]

        # Convert rotation matrix to quaternion
        quaternion = matrix_to_quaternion(rotation_matrix)

        body_node.translation = position.tolist()
        body_node.rotation = quaternion.tolist()
        body_node.children = []

        # Process each mesh with its individual transform
        for i, mesh_data in enumerate(meshes_data):
            mesh_is_plane = mesh_data.get("is_plane", False)

            # Create mesh and add to glTF
            mesh_index = _add_mesh_to_gltf(gltf, binary_data, mesh_data, mesh_is_plane)

            # Handle mesh position and rotation (old format)
            if "position" in mesh_data and "rotation" in mesh_data:
                # Old format: position and rotation are separate
                mesh_position = mesh_data["position"]
                mesh_rotation = mesh_data["rotation"]  # Already in [x, y, z, w] format
            else:
                # New format: use transform matrix
                mesh_transform = mesh_data.get("transform", np.eye(4))
                mesh_position = mesh_transform[:3, 3].tolist()
                mesh_rotation_matrix = mesh_transform[:3, :3]
                mesh_rotation = matrix_to_quaternion(mesh_rotation_matrix).tolist()

            child_node = pygltflib.Node(
                name=f"body_{body_id}_mesh_{i}",
                mesh=mesh_index,
                translation=mesh_position,
                rotation=mesh_rotation,
            )

            child_idx = len(gltf.nodes)
            gltf.nodes.append(child_node)
            body_node.children.append(child_idx)

        # Add body node as child of root node
        node_idx = len(gltf.nodes)
        gltf.nodes.append(body_node)
        gltf.nodes[0].children.append(node_idx)  # Add to root node
        body_node_indices[body_id] = node_idx  # Track for animation

    # Add camera if present
    if model_builder.camera:
        _add_camera_to_gltf(gltf, model_builder.camera, root_node_idx=0)

    # Add individual cameras attached to bodies
    if hasattr(model_builder, "individual_cameras"):
        for camera_data in model_builder.individual_cameras:
            body_id = camera_data["body_id"]
            camera = camera_data["camera"]
            position = camera_data["position"]
            rotation = camera_data["rotation"]

            # Find the parent body node
            parent_node_idx = body_node_indices.get(
                body_id, 0
            )  # Default to root if body not found
            _add_camera_to_gltf(
                gltf,
                camera,
                root_node_idx=parent_node_idx,
                position=position,
                rotation=rotation,
            )

    # Handle animations
    if model_builder.animation_frames:
        _add_animation_to_gltf(
            gltf, binary_data, model_builder.animation_frames, body_node_indices
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
    material_info = mesh_data.get("material")

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

    # Set material
    material_idx = None
    if material_info and material_info.get("textures"):
        # Use MuJoCo material with textures
        material_idx = _add_material_with_textures(
            gltf, binary_data, material_info, vertices, attributes
        )
    elif colors is not None:
        # Check if this is a plane - add checker pattern for planes
        if is_plane:
            # Create checker pattern material
            material = pygltflib.Material(
                name="checker_material",
                doubleSided=True,
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorFactor=[0.9, 0.9, 0.9, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.8,
                ),
            )

            # Add UV coordinates for checker pattern
            if "TEXCOORD_0" not in attributes:
                # Generate UV coordinates based on vertex positions
                uvs = []
                for vertex in vertices:
                    # Use XZ coordinates for UV mapping on horizontal planes
                    u = vertex[0] * 0.1  # Scale factor for checker size
                    v = vertex[2] * 0.1
                    uvs.append([u, v])

                uvs = np.array(uvs, dtype=np.float32)
                uv_accessor = _add_accessor(
                    gltf, binary_data, uvs, pygltflib.FLOAT, pygltflib.VEC2
                )
                attributes["TEXCOORD_0"] = uv_accessor
        else:
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

    # Create primitive after material handling
    primitive = pygltflib.Primitive(
        attributes=attributes, indices=index_accessor, mode=pygltflib.TRIANGLES
    )

    # Set material if we created one
    if material_idx is not None:
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


def _add_camera_to_gltf(
    gltf: pygltflib.GLTF2, camera_data, root_node_idx=None, position=None, rotation=None
):
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

    # Set camera transform
    if position is not None and rotation is not None:
        # Use provided position and rotation (from body-attached cameras)
        camera_node.translation = position
        camera_node.rotation = rotation
    else:
        # Use camera_data look-at transform (for global cameras)
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

    # Add to scene or root node
    node_idx = len(gltf.nodes)
    gltf.nodes.append(camera_node)
    if root_node_idx is not None:
        gltf.nodes[root_node_idx].children.append(node_idx)
    else:
        gltf.scenes[0].nodes.append(node_idx)


def _add_animation_to_gltf(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    animation_frames: List[Tuple[float, Dict]],
    body_node_indices: Dict[int, int],
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
        node_idx = body_node_indices.get(body_id)
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


def _add_material_with_textures(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    material_info: Dict[str, Any],
    vertices: np.ndarray,
    attributes: Dict[str, int],
) -> int:
    """Add a material with textures to the glTF.

    Args:
        gltf: glTF object
        binary_data: Binary data buffer
        material_info: Material information from MuJoCo
        vertices: Vertex data for UV generation
        attributes: Vertex attributes dict (modified in place)

    Returns:
        Material index
    """
    material_name = material_info.get("name", "mujoco_material")
    material_rgba = material_info.get("rgba", [1.0, 1.0, 1.0, 1.0])
    texrepeat = material_info.get("texrepeat", [1.0, 1.0])
    textures = material_info.get("textures", {})

    # Create material
    material = pygltflib.Material(
        name=material_name,
        doubleSided=True,
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=material_rgba.tolist()
            if hasattr(material_rgba, "tolist")
            else material_rgba,
            metallicFactor=0.0,
            roughnessFactor=0.8,
        ),
    )

    # Add diffuse texture if available (check all texture slots for the first available texture)
    texture_info = None
    for slot in textures:
        texture_info = textures[slot]
        break  # Use the first available texture

    if texture_info:
        # Add texture to glTF
        texture_idx = _add_texture_to_gltf(gltf, binary_data, texture_info)

        # Set base color texture
        material.pbrMetallicRoughness.baseColorTexture = pygltflib.TextureInfo(
            index=texture_idx
        )

        # Generate UV coordinates based on vertex positions and material texrepeat
        if "TEXCOORD_0" not in attributes:
            # Calculate bounding box for proper UV mapping
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            size = max_coords - min_coords
            
            uvs = []
            for vertex in vertices:
                # Check if this is a plane by looking at which coordinate is constant
                if abs(vertex[2]) < 0.001:  # XY plane (Z near 0)
                    # Map XY to UV using actual plane dimensions
                    u = (vertex[0] - min_coords[0]) / size[0] * texrepeat[0]
                    v = (vertex[1] - min_coords[1]) / size[1] * texrepeat[1]
                elif abs(vertex[1]) < 0.001:  # XZ plane (Y near 0)
                    # Map XZ to UV using actual plane dimensions
                    u = (vertex[0] - min_coords[0]) / size[0] * texrepeat[0]
                    v = (vertex[2] - min_coords[2]) / size[2] * texrepeat[1]
                else:
                    # For non-planar surfaces, use a more conservative scaling
                    # Use the largest dimension to normalize
                    max_size = max(size[0], size[1], size[2])
                    u = (vertex[0] - min_coords[0]) / max_size * texrepeat[0]
                    v = (vertex[1] - min_coords[1]) / max_size * texrepeat[1]
                uvs.append([u, v])

            uvs = np.array(uvs, dtype=np.float32)
            uv_accessor = _add_accessor(
                gltf, binary_data, uvs, pygltflib.FLOAT, pygltflib.VEC2
            )
            attributes["TEXCOORD_0"] = uv_accessor

    # Add material to glTF
    material_idx = len(gltf.materials)
    gltf.materials.append(material)

    return material_idx


def _add_texture_to_gltf(
    gltf: pygltflib.GLTF2,
    binary_data: bytearray,
    texture_info: Dict[str, Any],
) -> int:
    """Add a texture to the glTF.

    Args:
        gltf: glTF object
        binary_data: Binary data buffer
        texture_info: Texture information from MuJoCo

    Returns:
        Texture index
    """
    tex_data = texture_info["data"]
    tex_width = texture_info["width"]
    tex_height = texture_info["height"]
    tex_channels = texture_info["channels"]

    # Convert texture data to proper format and encode as PNG
    if tex_channels == 3:
        # RGB data - need to add alpha channel
        rgba_data = np.zeros((tex_height, tex_width, 4), dtype=np.uint8)
        rgba_data[:, :, :3] = tex_data
        rgba_data[:, :, 3] = 255  # Full alpha
        image_data = rgba_data
    else:
        # Already has alpha or is grayscale
        image_data = tex_data.astype(np.uint8)

    # Convert to PNG bytes
    try:
        import io
        from PIL import Image

        # Create PIL image
        if image_data.shape[2] == 4:
            pil_image = Image.fromarray(image_data, "RGBA")
        elif image_data.shape[2] == 3:
            pil_image = Image.fromarray(image_data, "RGB")
        else:
            pil_image = Image.fromarray(image_data[:, :, 0], "L")

        # Convert to PNG bytes
        png_buffer = io.BytesIO()
        pil_image.save(png_buffer, format="PNG")
        image_bytes = png_buffer.getvalue()
        use_png_mime = True

    except ImportError:
        # Fallback to raw data if PIL not available
        image_bytes = image_data.flatten().tobytes()
        use_png_mime = False

    # Add image data to binary buffer
    image_byte_offset = len(binary_data)
    binary_data.extend(image_bytes)

    # Create buffer view for image
    image_buffer_view = pygltflib.BufferView(
        buffer=0,
        byteOffset=image_byte_offset,
        byteLength=len(image_bytes),
    )
    image_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(image_buffer_view)

    # Create image
    image = pygltflib.Image(
        name=texture_info.get("name", "texture"),
        bufferView=image_view_idx,
    )

    # Set mimeType if we encoded as PNG
    if use_png_mime:
        image.mimeType = "image/png"
    image_idx = len(gltf.images)
    gltf.images.append(image)

    # Create sampler
    sampler = pygltflib.Sampler(
        magFilter=pygltflib.LINEAR,
        minFilter=pygltflib.LINEAR,
        wrapS=pygltflib.REPEAT,
        wrapT=pygltflib.REPEAT,
    )
    sampler_idx = len(gltf.samplers)
    gltf.samplers.append(sampler)

    # Create texture
    texture = pygltflib.Texture(
        source=image_idx,
        sampler=sampler_idx,
        name=texture_info.get("name", "texture"),
    )
    texture_idx = len(gltf.textures)
    gltf.textures.append(texture)

    return texture_idx


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
