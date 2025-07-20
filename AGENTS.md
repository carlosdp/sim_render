# AGENTS.md - Developer Documentation

## Package Purpose

`simrender` is a Python package designed to easily render robotic simulation environments to GLB format, including animations for episodes. The package provides seamless integration with popular robotics simulation frameworks like Gymnasium and MuJoCo, enabling researchers and developers to export their simulation data for visualization, analysis, and sharing.

## Notes

- This codebase uses `uv`, so use `uv run` when running python commands and scripts and such

## Code Layout

```
simrender/
├── __init__.py                 # Package exports and imports
├── model_builder.py           # Core 3D scene construction (ModelBuilder class)
├── gym.py                     # Gymnasium environment wrapper (InteractiveRenderWrapper)
├── mujoco.py                  # MuJoCo integration (MujocoRender class)
├── glb_builder.py             # GLB format building utilities
├── viewer.py                  # GLBViewer class for notebook/web rendering with ThreeJS
└── export/
    ├── __init__.py
    └── glb.py                 # GLB export functionality using pygltflib

tests/
├── __init__.py
├── test_gym.py                # Tests for Gymnasium wrapper
├── test_mujoco.py             # Tests for MuJoCo rendering
└── test_model_builder.py      # Tests for core model building
```

## Core Components

### ModelBuilder (`model_builder.py`)
- **Purpose**: Core class for 3D scene construction and management
- **Key Features**: 
  - Mesh generation utilities (spheres, boxes, planes)
  - Transform management for bodies
  - Animation frame recording
  - Camera configuration
- **Main Classes**: `ModelBuilder`, `RawMesh`, `BodyGeometry`, `CameraData`

### MuJoCo Integration (`mujoco.py`)
- **Purpose**: Direct MuJoCo model rendering and animation
- **Key Features**:
  - MuJoCo model geometry extraction
  - Coordinate system conversion (MuJoCo ↔ glTF)
  - Animation context management
- **Main Classes**: `MujocoRender`, `_AnimationContext`

### Gymnasium Wrapper (`gym.py`)
- **Purpose**: Wrap Gymnasium environments for GLB export
- **Key Features**:
  - Automatic MuJoCo environment detection
  - Episode recording with animation support
  - Context manager for render settings
- **Main Classes**: `InteractiveRenderWrapper`

### GLB Export (`export/glb.py`)
- **Purpose**: Export 3D scenes to GLB format
- **Key Features**:
  - Binary GLB file generation
  - Animation data encoding
  - Material and camera support
- **Main Functions**: `export_to_glb`, mesh/accessor utilities

## Libraries Used

### Core Dependencies
- **numpy**: Numerical computing for 3D transformations and mesh data
- **pygltflib**: GLB/glTF file format handling and export

### Optional Dependencies
- **gymnasium**: Reinforcement learning environment framework integration
- **mujoco**: Physics simulation engine integration
- **imageio**: Image processing (development dependency)

### Development Dependencies
- **pytest**: Testing framework
- **ruff**: Code linting and formatting
- **ty**: Type checking utilities

## Development Workflow

### Code Formatting
Use `ruff` for code formatting and linting:

```bash
# Format code
uv run ruff format

# Check linting
uv run ruff check

# Fix auto-fixable linting issues
uv run ruff check --fix
```

### Testing
Run tests with pytest:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_mujoco.py

# Run tests with verbose output
uv run pytest -v
```

### Build and Install
The package uses modern Python packaging with `pyproject.toml`:

```bash
# Install in development mode
uv sync

# Install with optional dependencies
uv sync --extra dev
```

## Key Design Patterns

### Coordinate System Conversion
MuJoCo uses Y-up, Z-forward while glTF uses Y-up, Z-backward:
- Conversion utilities in `mujoco.py`
- Quaternion and position transformations handled automatically

### Mesh Generation
Static methods on `ModelBuilder` provide common mesh primitives:
- `generate_sphere_mesh()`: UV sphere generation
- `generate_box_mesh()`: Box with proper face normals
- `generate_plane_mesh()`: Simple ground plane

## Contributing Guidelines

1. **Code Style**: Always run `uv run ruff format` before committing
2. **Testing**: Add tests for new functionality in the appropriate test files
3. **Documentation**: Update docstrings and this AGENTS.md file for significant changes
4. **Optional Dependencies**: Ensure code gracefully handles missing dependencies
5. **Coordinate Systems**: Be mindful of MuJoCo ↔ glTF coordinate conversions

## Architecture Notes

- **Modular Design**: Each integration (MuJoCo, Gymnasium) is self-contained
- **Extensible**: New simulation frameworks can be added following the existing patterns
- **Performance**: Uses numpy for efficient mesh operations and binary GLB output
- **Standards Compliance**: Generates valid glTF 2.0 files with proper animation support
