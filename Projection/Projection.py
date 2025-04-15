import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh

class ProjectorDisplay:
    def __init__(self, model_path, fullscreen=True, projector_display=1):
        """
        Initialize projector display for 3D model
        
        Args:
            model_path (str): Path to the 3D model file
            fullscreen (bool): Whether to run in fullscreen mode
            projector_display (int): Display index for the projector (typically 1 for secondary display)
        """
        # Load the 3D model
        self.mesh = trimesh.load(model_path)
        print(f"Model loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Display settings
        self.fullscreen = fullscreen
        self.projector_display = projector_display
        
        # Ensure we have faces and vertices as numpy arrays
        self.vertices = np.array(self.mesh.vertices)
        self.faces = np.array(self.mesh.faces)
        
        # Fit the model within projection bounds
        self.fit_object_in_cube()
        
        # Projection settings
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.zoom = -5
        
        # Initialize pygame and OpenGL
        self.init_display()
    
    def get_object_dimensions(self):
        """Calculate the dimensions of the model"""
        mins = np.min(self.vertices, axis=0)
        maxs = np.max(self.vertices, axis=0)
        dimensions = maxs - mins
        return dimensions, mins, maxs
    
    def fit_object_in_cube(self, cube_size=2.0, center_pos=(0, 0, 0)):
        """Scale and position the object to fit within a cube"""
        # Object dimensions measurements
        dimensions, mins, maxs = self.get_object_dimensions()
        
        # Scaling factor to fit in cube
        max_dimension = max(dimensions)
        scale_factor = cube_size / max_dimension if max_dimension > 0 else 1.0
        
        # Application of scaling
        self.vertices *= scale_factor
        
        # Recalculation of dimensions after scaling
        dimensions, mins, maxs = self.get_object_dimensions()
        
        # Center of the object
        center = (mins + maxs) / 2
        
        # Object moved to center position
        translation = np.array([
            center_pos[0] - center[0],
            center_pos[1] - center[1],
            center_pos[2] - center[2]
        ])
        self.vertices += translation
        
        print(f"Model dimensions after scaling: {dimensions}")
        print(f"Model centered at: {center_pos}")
        
        return dimensions
    
    def init_display(self):
        """Initialize pygame display with OpenGL context"""
        pygame.init()
        pygame.display.set_caption('3D Model Projector')
        
        # Get information about displays
        displays = pygame.display.get_desktop_sizes()
        print(f"Detected displays: {len(displays)}")
        for i, display in enumerate(displays):
            print(f"Display {i}: {display[0]}x{display[1]}")
        
        # Set up display on projector
        display_flags = DOUBLEBUF | OPENGL
        if self.fullscreen:
            display_flags |= FULLSCREEN
        
        # Check if projector display is available
        if self.projector_display < len(displays):
            # Set environment variable to target secondary display
            import os
            if self.projector_display > 0:
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{displays[0][0]},{0}"
            
            self.display_width, self.display_height = displays[self.projector_display]
            print(f"Using display {self.projector_display}: {self.display_width}x{self.display_height}")
        else:
            print(f"Warning: Display {self.projector_display} not found. Using primary display.")
            self.display_width, self.display_height = 800, 600
        
        # Create display surface
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            display_flags
        )
        
        # Set up OpenGL projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect_ratio = self.display_width / self.display_height
        gluPerspective(45, aspect_ratio, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
    
    def render_model(self):
        """Render the 3D model using OpenGL"""
        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Reset the modelview matrix
        glLoadIdentity()
        
        # Position the model
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glRotatef(self.rotation_z, 0, 0, 1)
        
        # Set material properties
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glColor3f(1.0, 1.0, 1.0)  # White model
        
        # Render the model
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            # Generate normal for the face
            v0, v1, v2 = self.vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:  # Avoid division by zero
                normal = normal / norm_length
                
            glNormal3fv(normal)
            for vertex_idx in face:
                glVertex3fv(self.vertices[vertex_idx])
        glEnd()
        
        # Draw a wireframe cube to represent the bounds
        self.draw_cube_bounds()
        
        # Update the display
        pygame.display.flip()
    
    def draw_cube_bounds(self, size=2.0, color=(0.5, 0.5, 1.0, 0.3)):
        """Draw a wireframe cube to visualize the bounds"""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glColor4f(*color)
        glLineWidth(1.0)
        
        # Draw cube wireframe
        glBegin(GL_LINES)
        half = size / 2
        
        # Bottom face
        glVertex3f(-half, -half, -half)
        glVertex3f(half, -half, -half)
        
        glVertex3f(half, -half, -half)
        glVertex3f(half, half, -half)
        
        glVertex3f(half, half, -half)
        glVertex3f(-half, half, -half)
        
        glVertex3f(-half, half, -half)
        glVertex3f(-half, -half, -half)
        
        # Top face
        glVertex3f(-half, -half, half)
        glVertex3f(half, -half, half)
        
        glVertex3f(half, -half, half)
        glVertex3f(half, half, half)
        
        glVertex3f(half, half, half)
        glVertex3f(-half, half, half)
        
        glVertex3f(-half, half, half)
        glVertex3f(-half, -half, half)
        
        # Connecting edges
        glVertex3f(-half, -half, -half)
        glVertex3f(-half, -half, half)
        
        glVertex3f(half, -half, -half)
        glVertex3f(half, -half, half)
        
        glVertex3f(half, half, -half)
        glVertex3f(half, half, half)
        
        glVertex3f(-half, half, -half)
        glVertex3f(-half, half, half)
        glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def handle_input(self):
        """Handle keyboard and mouse input for model manipulation"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.rotation_x += 5
                elif event.key == pygame.K_DOWN:
                    self.rotation_x -= 5
                elif event.key == pygame.K_LEFT:
                    self.rotation_y += 5
                elif event.key == pygame.K_RIGHT:
                    self.rotation_y -= 5
                elif event.key == pygame.K_q:
                    self.rotation_z += 5
                elif event.key == pygame.K_e:
                    self.rotation_z -= 5
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom += 0.5
                elif event.key == pygame.K_MINUS:
                    self.zoom -= 0.5
                elif event.key == pygame.K_r:
                    # Reset view
                    self.rotation_x = 0
                    self.rotation_y = 0
                    self.rotation_z = 0
                    self.zoom = -5
                elif event.key == pygame.K_f:
                    # Toggle fullscreen
                    self.fullscreen = not self.fullscreen
                    flags = DOUBLEBUF | OPENGL
                    if self.fullscreen:
                        flags |= FULLSCREEN
                    self.display = pygame.display.set_mode(
                        (self.display_width, self.display_height),
                        flags
                    )
                # Add controls for adjusting the model size
                elif event.key == pygame.K_1:
                    # Decrease size
                    self.vertices *= 0.9
                    print("Model size decreased")
                elif event.key == pygame.K_2:
                    # Increase size
                    self.vertices *= 1.1
                    print("Model size increased")
                elif event.key == pygame.K_c:
                    # Recenter model
                    self.fit_object_in_cube(cube_size=2.0, center_pos=(0, 0, 0))
                    print("Model recentered")
        
        # Get mouse state for continuous rotation
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:  # Left button
            rel_x, rel_y = pygame.mouse.get_rel()
            self.rotation_y += rel_x / 5
            self.rotation_x += rel_y / 5
        else:
            pygame.mouse.get_rel()  # Reset relative position
            
        return True
    
    def run(self):
        """Main loop for projection display"""
        print("\nStarting projection display. Controls:")
        print("  Arrow keys: Rotate model")
        print("  Q/E: Rotate around Z axis")
        print("  +/-: Zoom in/out")
        print("  1/2: Decrease/Increase model size")
        print("  C: Recenter and fit model")
        print("  R: Reset view")
        print("  F: Toggle fullscreen")
        print("  ESC: Exit")
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Handle input
            running = self.handle_input()
            
            # Render model
            self.render_model()
            
            # Cap the frame rate
            clock.tick(60)
        
        pygame.quit()

def main():
    # Get the model path
    model_path = input("Enter path to 3D model file: ")
    
    # Display configuration
    use_projector = input("Use projector display? (y/n): ").lower() == 'y'
    if use_projector:
        projector_index = int(input("Enter projector display index (usually 1): "))
    else:
        projector_index = 0
    
    fullscreen = input("Run in fullscreen mode? (y/n): ").lower() == 'y'
    
    # Create and run the projector display
    projector = ProjectorDisplay(
        model_path=model_path,
        fullscreen=fullscreen,
        projector_display=projector_index
    )
    
    projector.run()

if __name__ == "__main__":
    main()