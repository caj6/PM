import threading
import numpy as np
import cv2
import pyrealsense2 as rs
import trimesh
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Face Detector
class FaceDetector:
    def __init__(self, model_path, config_path):
        self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.rotation_data = [0, 0]
        self.running = True
        threading.Thread(target=self.process_frames, daemon=True).start()

    def process_frames(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]

            blob = cv2.dnn.blobFromImage(color_image, 1.0, (300, 300), [104, 117, 123], swapRB=False)
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    startX, startY, endX, endY = box.astype("int")
                    
                    center_x = (startX + endX) // 2
                    center_y = (startY + endY) // 2

                    dx = (center_x - w // 2) / (w // 2)
                    dy = (center_y - h // 2) / (h // 2)

                    self.rotation_data[0] = np.clip(dy * -90, -90, 90)
                    self.rotation_data[1] = np.clip(dx * -90, -90, 90)
                    
                    cv2.rectangle(color_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    break

            cv2.imshow("Face Detector", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def get_rotation(self):
        return self.rotation_data

    def stop(self):
        self.running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()


class HologramViewer:
    def __init__(self, model_path, face_detector=None):
        # Initialize PyGame
        pygame.init()
        self.width, self.height = 800, 600
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Holographic 3D Model Viewer")
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up lighting for holographic effect
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set light position and properties for holographic look
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 10, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.3, 0.5, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.4, 0.6, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 1.0, 1])
        
        # Load the 3D model
        self.model = trimesh.load(model_path)
        self.vertices = self.model.vertices
        self.faces = self.model.faces
        self.face_detector = face_detector
        
        # Fixed model position (model space)
        self.model_position = [0, 0, 0]  # Centered for holographic effect
        
        # Center and scale model to fit viewing area
        self.center_and_scale_model()
        
        # Hologram effect parameters - MOVED BEFORE display list creation
        self.hologram_alpha = 0.7  # Transparency
        self.hologram_glow = 0.3   # Glow effect intensity
        self.rotation_speed = 0.1  # Auto-rotation speed
        self.auto_rotate = True    # Auto-rotate by default
        self.current_rotation = 0  # Current auto-rotation angle
        
        # Generate display list for the model (for faster rendering)
        self.model_display_list = self.create_model_display_list()
        
        # Camera parameters
        self.camera_distance = 5.0
        self.camera_elevation = 30  # Degrees (up/down)
        self.camera_azimuth = -60   # Degrees (left/right)
        
        # Configure projection
        self.setup_projection()
        
        # Running flag
        self.running = True
    
    def center_and_scale_model(self):
        # Get model dimensions
        mins = np.min(self.vertices, axis=0)
        maxs = np.max(self.vertices, axis=0)
        dimensions = maxs - mins
        center = (mins + maxs) / 2
        
        # Scale factor to normalize model size
        max_dim = np.max(dimensions)
        scale_factor = 2.0 / max_dim if max_dim > 0 else 1.0
        
        # Center the model at origin, scale it, then translate to desired position
        for i in range(len(self.vertices)):
            # Center at origin
            self.vertices[i] = self.vertices[i] - center
            # Scale
            self.vertices[i] = self.vertices[i] * scale_factor
            # Move to desired position
            self.vertices[i] = self.vertices[i] + np.array(self.model_position)
    
    def create_model_display_list(self):
        # Create a display list for faster rendering
        display_list = glGenLists(1)
        glNewList(display_list, GL_COMPILE)
        
        # Set holographic blue color for the model with transparency
        glColor4f(0.3, 0.7, 1.0, self.hologram_alpha)
        
        # Draw the model as triangles
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            # Calculate face normal for better lighting
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            # Calculate normal vector using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)  # Normalize
            
            # Set normal and draw the face
            glNormal3fv(normal)
            glVertex3fv(v0)
            glVertex3fv(v1)
            glVertex3fv(v2)
        
        glEnd()
        glEndList()
        
        return display_list
    
    def draw_hologram_base(self):
        # Draw a circular hologram projector base
        glDisable(GL_LIGHTING)
        
        # Draw circular base
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.2, 0.3, 0.4, 1.0)  # Dark blue base
        
        # Center of the circle
        glVertex3f(0, 0, -1.0)
        
        # Create circle points
        radius = 1.2
        segments = 32
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, -1.0)
        
        glEnd()
        
        # Draw hologram projection light (a subtle glow from the base)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending for glow effect
        
        glBegin(GL_TRIANGLE_FAN)
        # Center with bright color
        glColor4f(0.4, 0.6, 0.9, 0.3)
        glVertex3f(0, 0, -0.99)
        
        # Outer edge with transparent color
        glColor4f(0.3, 0.5, 0.8, 0.0)
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = radius * 0.8 * np.cos(angle)
            y = radius * 0.8 * np.sin(angle)
            glVertex3f(x, y, -0.99)
        
        glEnd()
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Reset blending
        glEnable(GL_LIGHTING)
    
    def draw_hologram_beam(self):
        # Draw the hologram projection beam
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        
        # Set the beam color (light blue with transparency)
        beam_color = (0.3, 0.6, 0.9, 0.1)  # Very transparent
        
        # Draw a cone from the base to the model
        glBegin(GL_TRIANGLE_FAN)
        # Center of the base (point light source)
        glColor4f(*beam_color)
        glVertex3f(0, 0, -1.0)
        
        # Draw the cone edge around the model
        segments = 20
        radius = 1.0
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glColor4f(beam_color[0], beam_color[1], beam_color[2], 0.0)  # Transparent edge
            glVertex3f(x, y, 0.0)  # At model height
        
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_hologram_glow(self):
        # Add a subtle glow effect around the model
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending for glow
        
        # Save current matrix
        glPushMatrix()
        
        # Scale slightly larger than the model
        scale_factor = 1.05
        glScalef(scale_factor, scale_factor, scale_factor)
        
        # Draw a slightly larger version of the model with high transparency
        glColor4f(0.3, 0.7, 1.0, self.hologram_glow)
        glCallList(self.model_display_list)
        
        # Restore matrix
        glPopMatrix()
        
        # Reset blending
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
    
    def setup_projection(self):
        # Set up the perspective projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
    
    def update_camera(self):
        # Update camera view based on current parameters
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Convert spherical coordinates to Cartesian coordinates
        phi = np.radians(90 - self.camera_elevation)  # Convert elevation to spherical coordinates
        theta = np.radians(self.camera_azimuth)
        
        # Camera position in spherical coordinates (centered on model position)
        camera_x = self.model_position[0] + self.camera_distance * np.sin(phi) * np.cos(theta)
        camera_y = self.model_position[1] + self.camera_distance * np.sin(phi) * np.sin(theta)
        camera_z = self.model_position[2] + self.camera_distance * np.cos(phi)
        
        # Set up the camera view
        gluLookAt(
            camera_x, camera_y, camera_z,                  # Camera position
            self.model_position[0], self.model_position[1], self.model_position[2],  # Look at point
            0, 0, 1                                        # Up vector
        )
    
    def handle_events(self):
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_UP:
                    self.camera_elevation += 5
                elif event.key == pygame.K_DOWN:
                    self.camera_elevation -= 5
                elif event.key == pygame.K_LEFT:
                    self.camera_azimuth += 5
                elif event.key == pygame.K_RIGHT:
                    self.camera_azimuth -= 5
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera_distance = max(1.0, self.camera_distance - 0.5)
                elif event.key == pygame.K_MINUS:
                    self.camera_distance += 0.5
                elif event.key == pygame.K_r:  # Toggle auto-rotation
                    self.auto_rotate = not self.auto_rotate
    
    def update_from_face_detector(self):
        # Update camera position based on face detection
        if self.face_detector:
            rot_x, rot_y = self.face_detector.get_rotation()
            if rot_x != 0 or rot_y != 0:
                self.camera_elevation = np.clip(-rot_x, -85, 85)  # Avoid gimbal lock at ±90°
                self.camera_azimuth = np.clip(-rot_y * 2, -180, 180)
    
    def update_auto_rotation(self):
        # Update auto-rotation angle if enabled
        if self.auto_rotate:
            self.current_rotation += self.rotation_speed
            if self.current_rotation >= 360:
                self.current_rotation = 0
    
    def draw(self):
        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.05, 0.05, 0.1, 1.0)  # Very dark blue background
        
        # Update the camera position
        self.update_camera()
        
        # Draw the hologram base and beam
        #elf.draw_hologram_base()
        #elf.draw_hologram_beam()
        
        # Save the current matrix
        glPushMatrix()
        
        # Apply auto-rotation if enabled
        if self.auto_rotate:
            glRotatef(self.current_rotation, 0, 0, 1)
        
        # Draw the model using display list
        glCallList(self.model_display_list)
        
        # Add hologram glow effect
        self.draw_hologram_glow()
        
        # Restore the matrix
        glPopMatrix()
        
        # Swap buffers to display the frame
        pygame.display.flip()
    
    def run(self):
        # Main rendering loop
        clock = pygame.time.Clock()
        while self.running:
            self.handle_events()
            self.update_from_face_detector()
            self.update_auto_rotation()
            self.draw()
            clock.tick(60)  # Target 60 fps


def main():
    """Main function to run the application"""
    print("Holographic 3D Model Viewer")
    print("---------------------------")
    print("The model is displayed as a holographic projection")
    
    try:
        model_path = r"C:\Users\junio\Downloads\GitHub\IA\PM\3D in blender\rp_dennis_posed_004_30k.OBJ"
        
        # Initialize face detector
        detector = FaceDetector(
            r"C:\Users\junio\Downloads\GitHub\IA\PM\FD\opencv_face_detector_uint8.pb",
            r"C:\Users\junio\Downloads\GitHub\IA\PM\FD\opencv_face_detector.pbtxt"
        )
        
        # Create and run the hologram viewer
        viewer = HologramViewer(model_path, detector)
        viewer.run()
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'detector' in locals():
            detector.stop()
        pygame.quit()
        print("Application closed")


if __name__ == "__main__":
    main()