import base64
from pathlib import Path


class GLBViewer:
    """A viewer for GLB files that renders them using Three.js in HTML."""

    def __init__(self, glb_path: str | Path):
        """Initialize the viewer with a path to a GLB file.

        Args:
            glb_path: Path to the GLB file to display
        """
        self.glb_path = Path(glb_path)
        if not self.glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        # Read the GLB file and encode as base64
        with open(self.glb_path, "rb") as f:
            glb_data = f.read()
        glb_base64 = base64.b64encode(glb_data).decode("utf-8")
        glb_data_url = f"data:model/gltf-binary;base64,{glb_base64}"

        # Create self-contained HTML with Three.js viewer
        html = f"""
        <div id="glb-viewer-{{id}}" style="width: 100%; height: 600px; position: relative;">
            <canvas id="canvas-{{id}}" style="width: 100%; height: 100%; display: block;"></canvas>
        </div>
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
            }}
        }}
        </script>
        <script type="module">
            // Import Three.js and GLTFLoader from CDN
            import * as THREE from 'three';
            import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
            
            // Generate unique ID for this viewer instance
            const viewerId = 'viewer-' + Math.random().toString(36).substr(2, 9);
            const containers = document.querySelectorAll('div[id^="glb-viewer-"]');
            const container = containers[containers.length - 1];
            container.id = container.id.replace('{{id}}', viewerId);
            const canvas = container.querySelector('canvas');
            canvas.id = canvas.id.replace('{{id}}', viewerId);
            
            // Setup scene
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);
            
            // Setup camera
            const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
            camera.position.set(5, 5, 5);
            
            // Setup renderer
            const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 5);
            directionalLight.castShadow = true;
            directionalLight.shadow.camera.near = 0.1;
            directionalLight.shadow.camera.far = 50;
            directionalLight.shadow.camera.left = -10;
            directionalLight.shadow.camera.right = 10;
            directionalLight.shadow.camera.top = 10;
            directionalLight.shadow.camera.bottom = -10;
            scene.add(directionalLight);
            
            // Add controls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Load GLB
            const loader = new GLTFLoader();
            let mixer = null;
            let clock = new THREE.Clock();
            
            // Convert base64 to blob
            const base64Data = '{glb_data_url}'.split(',')[1];
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            const blob = new Blob([bytes], {{ type: 'model/gltf-binary' }});
            const url = URL.createObjectURL(blob);
            
            loader.load(
                url,
                function (gltf) {{
                    scene.add(gltf.scene);
                    
                    // Center the model
                    const box = new THREE.Box3().setFromObject(gltf.scene);
                    const center = box.getCenter(new THREE.Vector3());
                    gltf.scene.position.sub(center);
                    
                    // Adjust camera to fit the model
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const fov = camera.fov * (Math.PI / 180);
                    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                    cameraZ *= 1.5; // Add some padding
                    camera.position.set(cameraZ, cameraZ * 0.5, cameraZ);
                    camera.lookAt(0, 0, 0);
                    
                    // Setup animations if any
                    if (gltf.animations && gltf.animations.length > 0) {{
                        mixer = new THREE.AnimationMixer(gltf.scene);
                        // Play all animations
                        gltf.animations.forEach((clip) => {{
                            mixer.clipAction(clip).play();
                        }});
                    }}
                    
                    // Handle camera if defined in GLB
                    if (gltf.cameras && gltf.cameras.length > 0) {{
                        // Use the first camera from the GLB
                        const gltfCamera = gltf.cameras[0];
                        camera.copy(gltfCamera);
                        camera.aspect = canvas.clientWidth / canvas.clientHeight;
                        camera.updateProjectionMatrix();
                    }}
                    
                    // Clean up blob URL
                    URL.revokeObjectURL(url);
                }},
                function (xhr) {{
                    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
                }},
                function (error) {{
                    console.error('Error loading GLB:', error);
                    URL.revokeObjectURL(url);
                }}
            );
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                
                if (mixer) {{
                    mixer.update(clock.getDelta());
                }}
                
                controls.update();
                renderer.render(scene, camera);
            }}
            
            // Handle window resize
            function handleResize() {{
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            }}
            
            // Observe canvas size changes
            const resizeObserver = new ResizeObserver(handleResize);
            resizeObserver.observe(canvas);
            
            // Start animation
            animate();
        </script>
        """

        return html
