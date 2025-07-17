"""sim_glb: A package for exporting simulations to GLB format."""

# Core model building
from .model_builder import ModelBuilder, RawMesh, BodyGeometry, CameraData

# MuJoCo integration
try:
    from .mujoco import MujocoRender
except ImportError:
    MujocoRender = None

# Gymnasium integration
try:
    from .gym import InteractiveRenderWrapper
except ImportError:
    InteractiveRenderWrapper = None

# Mesh generation utilities - imported as functions
generate_box_mesh = ModelBuilder.generate_box_mesh
generate_sphere_mesh = ModelBuilder.generate_sphere_mesh
generate_plane_mesh = ModelBuilder.generate_plane_mesh

__all__ = [
    # Core classes
    "ModelBuilder",
    "RawMesh",
    "BodyGeometry",
    "CameraData",
    # Integration classes
    "MujocoRender",
    "InteractiveRenderWrapper",
    # Utilities
    "generate_box_mesh",
    "generate_sphere_mesh",
    "generate_plane_mesh",
]

# Version
__version__ = "0.1.0"
