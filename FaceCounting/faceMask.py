# Author: Endri Dibra 
# Bachelor Thesis: Smart Unmanned Ground Vehicle

# importing the required libraries 
import cv2
import numpy as np
import mediapipe as mp

# initializing MediaPipe FaceMesh
mpFaceMesh = mp.solutions.face_mesh

# detecting max 30 faces
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=30)

# utility for drawing landmarks
mpDrawing = mp.solutions.drawing_utils

# opening the default camera camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# checking if camera works
if not camera.isOpened():
    
    print("Error! Camera did not open.")
    
    exit()

# looping through camera frames
while camera.isOpened():
    
    # capturing frame-by-frame from the camera
    success, frame = camera.read()
    
    # checking if the frame was captured successfully
    if not success:
        
        print("Error! Failed to capture frame.")
        
        break
    
    # converting the frame to RGB as MediaPipe FaceMesh expects RGB input
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # processing the frame to detect face mesh
    results = faceMesh.process(rgbFrame)
    
    # creating a mask for the detected face
    # mask with the same size as the frame
    mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    
    # initializing face counter
    totalFaces = 0 

    # checking if any faces were detected
    if results.multi_face_landmarks:
        
        totalFaces = len(results.multi_face_landmarks)

        # iterating through the detected faces
        for face_landmarks in results.multi_face_landmarks:
        
            # drawing the face mesh landmarks on the frame
            mpDrawing.draw_landmarks(
                
                frame, 
                face_landmarks, 
                mpFaceMesh.FACEMESH_TESSELATION,
                mpDrawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                mpDrawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
            )
            
            # getting the frame dimensions
            h, w, _ = frame.shape
        
            # converting normalized landmarks to pixel coordinates
            points = np.array([
                
                [int(landmark.x * w), int(landmark.y * h)]
                for landmark in face_landmarks.landmark
            ], dtype=np.int32)

            # creating a convex hull from the facial landmarks
            hull = cv2.convexHull(points)

            # drawing the polygon mask (white on black)
            cv2.fillConvexPoly(mask, hull, (255, 255, 255))
     
    # displaying the number of faces detected
    label = f"Faces Detected: {totalFaces}"
    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # showing the video with face mesh points
    cv2.imshow("Face Mesh Detection", frame)
    
    # showing the masked face in another window
    cv2.imshow("Masked Face", mask)

    # breaking the loop when 'q:quit' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
       
        break

# releasing the camera
camera.release()

# closing all the windows
cv2.destroyAllWindows()