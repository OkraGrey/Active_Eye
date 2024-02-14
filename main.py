import cv2
import pyautogui
import mediapipe as mp
import numpy as np
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
colo_val_1=10
colo_val_2=30
colo_val_3=2
screen_w,screen_h = pyautogui.size()

while True:

    colo_val_1=(colo_val_1+5 ) % 255
    colo_val_2=(colo_val_2+9 ) % 255
    colo_val_3=(colo_val_3+27) % 255
    _ , frame = cam.read()
    frame = cv2.flip(frame,1)
    frame_h,frame_w,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    face_landmarks = output.multi_face_landmarks
    if face_landmarks:
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
    #     cv2.circle(frame, (x, y), 2, (colo_val_1,colo_val_2,colo_val_3))
        
    #     y = (screen_h / frame_h) * y
    #     x = (screen_w / frame_w) * x

    #     pyautogui.moveTo(x, y)
        
##################################################################################################################################
        
        # eye_brow_highest_point = [landmarks[276],landmarks[ 283],landmarks[285]]
        # eye_brow_highest_point = landmarks[330:350]
        # for points in eye_brow_highest_point:
        points = [landmarks[340],landmarks[301],landmarks[6],landmarks[9]]
        # print(points[0])
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

        eyebox = frame[min_y:max_y, min_x:max_x]
        # cv2.imshow('Ap ki aankh nikal di', eyebox)

        # import cv2
        # import numpy as np
        import matplotlib.pyplot as plt

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


    def upscale_image(image, scale_percent):
        # Read the image
        # image = cv2.imread(image_path)

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



    scaled_image = upscale_image(eyebox, 400)  # Upscale by 200%
    sharpened_image = cv2.filter2D(scaled_image, -1, kernel)
    # sharpened_image = cv2.Laplacian(scaled_image, ksize=1, ddepth=cv2.CV_8U)
    cv2.imwrite('eye.jpg', eyebox)  # Save the upscaled image    
    cv2.imwrite('scaled.jpg', scaled_image)  # Save the upscaled image    
    cv2.imwrite('sharped.jpg', sharpened_image)  # Save the upscaled image    
    cv2.imshow("A moving curosr using Eye ball movement ",eyebox)
    cv2.waitKey(1)
