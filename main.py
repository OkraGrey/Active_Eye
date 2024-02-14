import cv2
import pyautogui
import mediapipe as mp
import numpy as np


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

screen_w,screen_h = pyautogui.size()

def upscale_image(image, scale_percent):

    # Get the original image dimensions
    width = int(image.shape[1])
    height = int(image.shape[0])

    # Calculate the new dimensions
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)

    # Resize the image using interpolation
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

kernel = np.array([[-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]])



while True:

    # Camera Input
    _ , frame = cam.read()
    # Flipping the frame to get upright input
    frame = cv2.flip(frame,1)
    # Frame height and width to get the necessary scaling
    frame_h,frame_w,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Processing through faceMesh 
    output = face_mesh.process(rgb_frame)
    # Facemesh outputs landmarks of multiple faces
    face_landmarks = output.multi_face_landmarks

    # Incase there is a face
    if face_landmarks:

        # We are considering only one face    
        landmarks = face_landmarks[0].landmark 
        # Cursor Movement
    #     x = 0
    #     y = 0
    #     for i in landmarks[474:478]:
            
    #         x += i.x
    #         y += i.y
        
    #     x = int(x * frame_w)
    #     y = int(y * frame_h)

    #     x= x//4
    #     y= y//4
    #     cv2.circle(frame, (x, y), 2, (0,255,22))
        
    #     y = (screen_h / frame_h) * y
    #     x = (screen_w / frame_w) * x

    #     pyautogui.moveTo(x, y)
        
##############################################################################
        
        # Landmarks are selected to make a rectangle from right eye as ROI
        points = [landmarks[340],landmarks[301],landmarks[6],landmarks[9]]
        
        x_points=[]
        y_points=[]

        for point in points:

            x_points.append(int(point.x*frame_w)) 
            y_points.append(int(point.y*frame_h))
        #     x = int(point.x*frame_w)
        #     y = int(point.y*frame_h)
        #     cv2.circle(frame,(x,y),2,(0,255,222))    

        min_x = min(x_points)
        max_x = max(x_points)
        min_y = min(y_points)
        max_y = max(y_points)
        # print(min_x,min_y,max_x,max_y)
        # break
        # cv2.rectangle(frame,(min_x,min_y),(max_x,max_y),(0,255,22),2)

        ################## IMAGE CROPING TO GET BOUNDING BOX ##########################

        # Eye box is extracted out to get only one eye out of the face
        eyebox = frame[min_y:max_y, min_x:max_x]
        # cv2.imshow('Ap ki aankh nikal di', eyebox)

##################################################################################################################################
        
        # face_classifier  = cv2.CascadeClassifier('haarcascade_eye.xml')
        # grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # faces = face_classifier.detectMultiScale(grey_frame,1.3,5)
        # # if faces:
        # for (x,y,w,h) in faces:
        #     cv2.rectangle(frame,(x,y),(x+w,y+h),2)

####################################################################################################################################

        #Tracking points for blinking
        # for point in [landmarks[386],landmarks[374]]:
        
        #     x = int(point.x * frame_w)
        #     y = int(point.y * frame_h)
        
        #     cv2.circle(frame, (x, y), 2, (colo_val_1,colo_val_2,colo_val_3))

    # Eye box scaling
    scaled_image = upscale_image(eyebox, 400)  # Upscale by 200%
    # Sharpening of scaled image
    sharpened_image = cv2.filter2D(scaled_image, -1, kernel)
    # Saving output images for further analysis

    cv2.imwrite('eye.jpg', eyebox)  # Save the upscaled image    
    cv2.imwrite('scaled.jpg', scaled_image)  # Save the upscaled image    
    cv2.imwrite('sharped.jpg', sharpened_image)  # Save the upscaled image    
    cv2.imshow("A moving curosr using Eye ball movement ",eyebox)
    
    cv2.waitKey(1)

