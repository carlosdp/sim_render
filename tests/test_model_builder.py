"""Test ModelBuilder functionality."""

import numpy as np
import os
import tempfile


class TestModelBuilder:
    """Test ModelBuilder functionality."""

    def test_model_builder_direct_usage(self):
        """Test ModelBuilder direct usage."""
        from sim_render import ModelBuilder, generate_sphere_mesh, generate_box_mesh

        builder = ModelBuilder()

        # Add a sphere
        sphere = generate_sphere_mesh(radius=0.5, color=np.array([1, 0, 0, 1]))
        builder.add_body(0, [sphere], transform=np.eye(4))

        # Add a box
        box = generate_box_mesh(
            half_sizes=np.array([0.2, 0.3, 0.1]), color=np.array([0, 1, 0, 1])
        )
        transform = np.eye(4)
        transform[:3, 3] = [1, 0, 0]  # Position at x=1
        builder.add_body(1, [box], transform=transform)

        # Add camera
        builder.set_camera(
            position=np.array([2, 2, 2]),
            target=np.array([0, 0, 0]),
            up=np.array([0, 1, 0]),
            fov=np.radians(45),
        )

        # Add animation
        builder.start_animation()
        for t in range(10):
            time = t * 0.1
            # Rotate the box
            angle = time * 2 * np.pi
            transforms = {
                1: {
                    "translation": [1, 0, 0],
                    "rotation": [
                        np.cos(angle / 2),
                        0,
                        np.sin(angle / 2),
                        0,
                    ],  # Y-axis rotation
                }
            }
            builder.add_animation_frame(time, transforms)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            builder.save_to_glb(tmp.name)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)

    def test_mesh_generation_functions(self):
        """Test mesh generation functions."""
        from sim_render import (
            generate_sphere_mesh,
            generate_box_mesh,
            generate_plane_mesh,
        )

        # Test sphere generation
        sphere = generate_sphere_mesh(radius=1.0, color=np.array([1, 0, 0, 1]))
        assert sphere.vertices.shape[1] == 3
        assert sphere.normals.shape[1] == 3
        assert sphere.colors.shape[1] == 4

        # Test box generation
        box = generate_box_mesh(
            half_sizes=np.array([0.5, 0.5, 0.5]), color=np.array([0, 1, 0, 1])
        )
        assert box.vertices.shape[1] == 3
        assert box.normals.shape[1] == 3
        assert box.colors.shape[1] == 4

        # Test plane generation
        plane = generate_plane_mesh(size=2.0, color=np.array([0, 0, 1, 1]))
        assert plane.vertices.shape[1] == 3
        assert plane.normals.shape[1] == 3
        assert plane.colors.shape[1] == 4

    def test_model_builder_encoding(self):
        """Test mesh encoding functionality."""
        from sim_render import ModelBuilder, generate_sphere_mesh

        builder = ModelBuilder()
        sphere = generate_sphere_mesh(radius=0.5)
        builder.add_body(0, [sphere])

        # Test encoding
        encoded = builder.get_encoded_bodies()
        assert len(encoded) == 1
        assert encoded[0]["body_id"] == 0
        assert len(encoded[0]["meshes"]) == 1

        mesh_data = encoded[0]["meshes"][0]
        assert "vertices" in mesh_data
        assert "normals" in mesh_data
        assert "indices" in mesh_data
        assert "colors" in mesh_data

    def test_model_builder_animation(self):
        """Test animation functionality."""
        from sim_render import ModelBuilder, generate_sphere_mesh

        builder = ModelBuilder()
        sphere = generate_sphere_mesh(radius=0.5)
        builder.add_body(0, [sphere])

        # Test animation
        builder.start_animation()
        assert builder._animation_started

        # Add some frames
        for i in range(3):
            transforms = {0: {"translation": [i * 0.1, 0, 0], "rotation": [1, 0, 0, 0]}}
            builder.add_animation_frame(i * 0.1, transforms)

        assert len(builder.animation_frames) == 3

    def test_model_builder_camera(self):
        """Test camera functionality."""
        from sim_render import ModelBuilder

        builder = ModelBuilder()

        # Set camera
        position = np.array([1, 2, 3])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        fov = np.radians(45)

        builder.set_camera(position, target, up, fov)

        assert builder.camera is not None
        assert np.allclose(builder.camera.position, position)
        assert np.allclose(builder.camera.target, target)
        assert np.allclose(builder.camera.up, up)
        assert builder.camera.fov == fov
