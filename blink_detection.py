# import cv2
# import pyautogui
# import mediapipe as mp
# import numpy as np


# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
# face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# screen_w,screen_h = pyautogui.size()

# def upscale_image(image, scale_percent):

#     # Get the original image dimensions
#     width = int(image.shape[1])
#     height = int(image.shape[0])

#     # Calculate the new dimensions
#     new_width = int(width * scale_percent / 100)
#     new_height = int(height * scale_percent / 100)

#     # Resize the image using interpolation
#     resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

#     return resized_image

# while True:

#     # Camera Input
#     _ , frame = cam.read()
#     # Flipping the frame to get upright input
#     frame = cv2.flip(frame,1)
#     # Frame height and width to get the necessary scaling
#     frame_h,frame_w,_ = frame.shape
#     rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

#     # Processing through faceMesh 
#     output = face_mesh.process(rgb_frame)
#     # Facemesh outputs landmarks of multiple faces
#     face_landmarks = output.multi_face_landmarks

#     # Incase there is a face
#     if face_landmarks:

#         # We are considering only one face    
#         landmarks = face_landmarks[0].landmark 
                
#         # Landmarks are selected to make a rectangle from right eye as ROI
#         points = [landmarks[340],landmarks[301],landmarks[6],landmarks[9],landmarks[380],landmarks[386]]
        
#         x_points = []
#         y_points = []

#         for point in points:

#             x_points.append(int(point.x*frame_w)) 
#             y_points.append(int(point.y*frame_h))
#             x = int(point.x*frame_w)
#             y = int(point.y*frame_h)
#             cv2.circle(frame,(x,y),2,(0,255,222))    


#         min_x = min(x_points)
#         max_x = max(x_points)
#         min_y = min(y_points)
#         max_y = max(y_points)
  
#     cv2.imshow("A moving curosr using Eye ball movement ",frame)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break
#     # cv2.waitKey(1)


##############################################################3

import cv2
import pyautogui
import mediapipe as mp
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def is_blink(landmarks, frame_w, frame_h):
    # Define the right eye landmarks for upper and lower eyelids
    upper_lid = landmarks[386]  # Top of the right eye
    lower_lid = landmarks[374]  # Bottom of the right eye

    # Get the screen coordinates
    upper_y = int(upper_lid.y * frame_h)
    lower_y = int(lower_lid.y * frame_h)

    # Calculate vertical distance
    vertical_distance = lower_y - upper_y
    return vertical_distance

# Initialization of variables to track blink states
blink_state = 'open'  # Initial state of the eye
blink_start = 0
closure_threshold = 5  # Threshold for eye closure, adjust based on testing
reopen_threshold = 10  # Threshold for eye reopening, adjust based on testing

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = face_mesh.process(rgb_frame)
    face_landmarks = output.multi_face_landmarks

    if face_landmarks:
        landmarks = face_landmarks[0].landmark 
        eye_distance = is_blink(landmarks, frame_w, frame_h)

        if blink_state == 'open' and eye_distance < closure_threshold:
            blink_state = 'closing'
            blink_start = cv2.getTickCount()

        elif blink_state == 'closing' and eye_distance >= reopen_threshold:
            blink_end = cv2.getTickCount()
            blink_duration = (blink_end - blink_start) / cv2.getTickFrequency()

            if 0.25 <= blink_duration <= 1.2:
                print("Intentional Blink Detected!")
            blink_state = 'open'

        elif blink_state == 'closing' and eye_distance < closure_threshold:
            blink_state = 'closed'

        elif blink_state == 'closed' and eye_distance >= reopen_threshold:
            blink_end = cv2.getTickCount()
            blink_duration = (blink_end - blink_start) / cv2.getTickFrequency()

            if 0.25 <= blink_duration <= 1.2:
                print("Intentional Blink Detected!")
            blink_state = 'open'

    cv2.imshow("A moving cursor using Eye ball movement", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



################################################################3

# import cv2
# import pyautogui
# import mediapipe as mp
# import numpy as np

# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# screen_w, screen_h = pyautogui.size()

# def upscale_image(image, scale_percent):
#     width = int(image.shape[1])
#     height = int(image.shape[0])
#     new_width = int(width * scale_percent / 100)
#     new_height = int(height * scale_percent / 100)
#     resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#     return resized_image

# # Blink detection function
# def is_blink(landmarks, frame_w, frame_h):
#     # Define the right eye landmarks for upper and lower eyelids
#     upper_lid = landmarks[386]  # Top of the right eye
#     lower_lid = landmarks[374]  # Bottom of the right eye

#     # Get the screen coordinates
#     upper_y = int(upper_lid.y * frame_h)
#     lower_y = int(lower_lid.y * frame_h)

#     # Calculate vertical distance
#     vertical_distance = lower_y - upper_y

#     # Threshold for eye closure
#     closure_threshold = 5  # You may need to adjust this based on actual testing

#     # Check if the eye is closed enough
#     if vertical_distance < closure_threshold:
#         return True
#     return False

# # Blink variables
# blink_start = 0
# blink_counting = False

# while True:
#     _, frame = cam.read()
#     frame = cv2.flip(frame, 1)
#     frame_h, frame_w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     output = face_mesh.process(rgb_frame)
#     face_landmarks = output.multi_face_landmarks

#     if face_landmarks:
#         landmarks = face_landmarks[0].landmark 
#         is_closed = is_blink(landmarks, frame_w, frame_h)

#         # Manage blink states
#         if is_closed and not blink_counting:
#             blink_start = cv2.getTickCount()
#             blink_counting = True
#         elif not is_closed and blink_counting:
#             blink_end = cv2.getTickCount()
#             blink_duration = (blink_end - blink_start) / cv2.getTickFrequency()
#             blink_counting = False

#             # Check if the blink was intentional
#             if 0.2 <= blink_duration <= 0.6:
#                 print("Intentional Blink Detected!")

#     cv2.imshow("A moving cursor using Eye ball movement", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cam.release()
# cv2.destroyAllWindows()
