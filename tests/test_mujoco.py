"""Test MujocoRender functionality."""

import pytest
import os
import tempfile


class TestMujocoRender:
    """Test MujocoRender functionality."""

    def test_mujoco_render_direct_usage(self):
        """Test direct MujocoRender usage if mujoco available."""
        pytest.importorskip("mujoco")

        import mujoco
        from sim_render.mujoco import MujocoRender

        # Create a simple MuJoCo model
        xml = """
        <mujoco>
            <worldbody>
                <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
                <body pos="0.2 0 0">
                    <geom type="box" size="0.05 0.05 0.05" rgba="0 1 0 1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Create renderer
        render = MujocoRender(model)

        # Single frame render
        mujoco.mj_step(model, data)
        render.render(data)

        # Animation
        with render.animation(fps=10):
            for i in range(10):
                mujoco.mj_step(model, data)
                render.render(data)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            render.save(tmp.name)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)

