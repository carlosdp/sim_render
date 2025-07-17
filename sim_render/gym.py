"""Gymnasium environment wrapper for interactive rendering."""

import warnings
from typing import Optional
from contextlib import contextmanager

try:
    import gymnasium as gym
except ImportError:
    gym = None
    warnings.warn(
        "Gymnasium not available. InteractiveRenderWrapper will not be available."
    )

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


class InteractiveRenderWrapper(gym.Wrapper if gym else object):
    """Wrapper for Gymnasium environments that enables GLB export with animation.

    This wrapper intercepts render calls and builds a 3D model that can be
    exported to GLB format. It automatically detects MuJoCo environments and
    uses MujocoRender for rendering.
    """

    def __init__(self, env):
        """Initialize the wrapper.

        Args:
            env: Gymnasium environment to wrap
        """
        if gym is None:
            raise ImportError("Gymnasium is required for InteractiveRenderWrapper")

        super().__init__(env)

        self._mujoco_render = None
        self._animation_context = None
        self._check_mujoco_env()

    def _check_mujoco_env(self):
        """Check if the wrapped environment is MuJoCo-based and set up rendering."""
        if not MUJOCO_AVAILABLE:
            return

        # Check if the unwrapped environment has MuJoCo model and data
        unwrapped = self.unwrapped

        if hasattr(unwrapped, "model") and hasattr(unwrapped, "data"):
            # Check if they are MuJoCo types
            if (
                hasattr(unwrapped.model, "__class__")
                and unwrapped.model.__class__.__name__ == "MjModel"
                and hasattr(unwrapped.data, "__class__")
                and unwrapped.data.__class__.__name__ == "MjData"
            ):
                from .mujoco import MujocoRender

                self._mujoco_render = MujocoRender(unwrapped.model)

    def reset(self, **kwargs):
        """Reset the environment and prepare for rendering."""
        result = super().reset(**kwargs)

        # Prepare the MuJoCo render if available
        if self._mujoco_render is not None:
            self._mujoco_render.prepare()

        return result

    def render(self, mode: Optional[str] = None):
        """Render the current state.

        Args:
            mode: Rendering mode (ignored, uses internal rendering)
        """
        if self._mujoco_render is not None:
            # Get current MjData from the unwrapped environment
            unwrapped = self.unwrapped
            if hasattr(unwrapped, "data"):
                self._mujoco_render.render(unwrapped.data)
        else:
            # For non-MuJoCo environments, we might want to add support later
            warnings.warn(
                "Non-MuJoCo environments are not yet supported for GLB export"
            )

    @contextmanager
    def animation(self, fps: int = 30):
        """Context manager for recording animations.

        Args:
            fps: Frames per second for the animation

        Usage:
            with env.animation(fps=30):
                for _ in range(100):
                    env.step(action)
                    env.render()
        """
        if self._mujoco_render is not None:
            with self._mujoco_render.animation(fps=fps):
                yield
        else:
            # No animation support for non-MuJoCo environments yet
            yield

    def save(self, filename: str):
        """Save the rendered scene to a GLB file.

        Args:
            filename: Output filename
        """
        if self._mujoco_render is not None:
            self._mujoco_render.save(filename)
        else:
            raise RuntimeError(
                "No render data available. Make sure to call render() first."
            )

    # Additional methods to match expected interface
    def save_to_glb(self, filename: str):
        """Alias for save() to match the expected interface.

        Args:
            filename: Output filename
        """
        self.save(filename)
