import os
import numpy as np
import trimesh
import pyvista as pv

class HologramGenerator:
    def __init__(self):
        self.scene = None
        self.hologram_object = None
        self.holo_cube = None
        self.frames = 100

    def get_object_dimensions(self, mesh):
        """Get the dimensions of a mesh object"""
        bounds = mesh.bounds
        mins = bounds[0]
        maxs = bounds[1]
        dimensions = maxs - mins
        return dimensions, mins, maxs

    def fit_object_in_cube(self, mesh, cube_size=2.0, center_pos=(0, 0, 1)):
        """Scale and position the object to fit within a cube"""
        # Object dimensions measurements
        dimensions, mins, maxs = self.get_object_dimensions(mesh)
        
        # Scaling factor to fit in cube
        max_dimension = max(dimensions)
        scale_factor = cube_size / max_dimension
        
        # Application of scaling
        mesh.apply_scale(scale_factor)
        
        # Recalculation of dimensions after scaling
        dimensions, mins, maxs = self.get_object_dimensions(mesh)
        
        # Center of the object
        center = (mins + maxs) / 2
        
        # Object moved to center position
        translation = np.array([
            center_pos[0] - center[0],
            center_pos[1] - center[1],
            center_pos[2] - center[2]
        ])
        mesh.apply_translation(translation)
        
        return dimensions

    def create_hologram_cube(self, size, center):
        """Create a cube to contain the hologram"""
        cube = trimesh.creation.box(extents=[size, size, size])
        cube.apply_translation(center)
        cube.visual.face_colors = np.array([0, 128, 255, 77])  # RGBA: translucent blue
        return cube

    def create_hologram_object(self, obj_path, cube_size, cube_center):
        # 3D model importation 
        if not os.path.exists(obj_path):
            print(f"The file '{obj_path}' was not found!")
            return None
        
        try:
            mesh = trimesh.load(obj_path)
            print("Model imported successfully")
        except Exception as e:
            print(f"Could not import OBJ file: {e}")
            return None
        
        # Fit object in cube
        dimensions = self.fit_object_in_cube(mesh, cube_size * 0.8, cube_center)
        print(f"Model dimensions after scaling: {dimensions}")
        
        # Set the object to be glowing green
        mesh.visual.face_colors = np.array([0, 255, 0, 220])  # RGBA: green
        
        return mesh

    def generate_hologram(self, obj_path):
        """Create a hologram from the given 3D model"""
        # CUBE SETTINGS
        cube_size = 2.5  # Size of the cube container
        cube_center = np.array([0, 0, 1.5])  # Center position of the cube

        # Create the cube
        self.holo_cube = self.create_hologram_cube(cube_size, cube_center)
        
        # Create the hologram object
        self.hologram_object = self.create_hologram_object(obj_path, cube_size, cube_center)
        
        # Create a scene
        self.scene = trimesh.Scene()
        self.scene.add_geometry(self.holo_cube, node_name="cube")
        
        if self.hologram_object is not None:
            self.scene.add_geometry(self.hologram_object, node_name="object")
        
        return self.scene

    def visualize(self):
        """Visualize the hologram scene using PyVista"""
        if self.scene is None:
            print("No scene to visualize")
            return
        
        # Convert trimesh scene to pyvista
        plotter = pv.Plotter()
        
        # Add the cube
        cube_pv = pv.PolyData(
            self.holo_cube.vertices, 
            np.hstack([np.ones((self.holo_cube.faces.shape[0], 1), dtype=np.int64) * 3, 
                       self.holo_cube.faces])
        )
        plotter.add_mesh(cube_pv, color=[0, 0.5, 1], opacity=0.3)
        
        # Add the object
        if self.hologram_object is not None:
            obj_pv = pv.PolyData(
                self.hologram_object.vertices, 
                np.hstack([np.ones((self.hologram_object.faces.shape[0], 1), dtype=np.int64) * 3, 
                           self.hologram_object.faces])
            )
            plotter.add_mesh(obj_pv, color=[0, 1, 0], opacity=0.8)
        
        # Set up camera
        plotter.camera_position = [(8, -8, 8), (0, 0, 1.5), (0, 0, 1)]
        plotter.camera.zoom(1.2)
        
        # Add light
        plotter.add_light(pv.Light(position=(5, 5, 5), color=[0.2, 0.8, 1]))
        
        # Show the scene
        plotter.show()


def main():
    # Path to 3D model OBJ file
    obj_file_path = input("Enter the 3D path: ")
    obj_file_path = os.path.expanduser(obj_file_path) 
    
    # Create hologram generator
    generator = HologramGenerator()
    
    # Generate hologram
    scene = generator.generate_hologram(obj_file_path)
    
    # Visualize the scene
    generator.visualize()
    
    print("!")


if __name__ == "__main__":
    main()