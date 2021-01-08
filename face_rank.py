import face_recognition
import cv2
import os
import torch
from PIL import Image
from torchvision import transforms, datasets
from my_net import Net

print('Press \'q\' to exit...')

# define transforms
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

classes = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9')

# load model
model = Net()
model_name = model.__class__.__name__
model_path = os.path.join('checkpoints',model_name,'best_model.pth.tar')
best_model = torch.load(model_path)
model.load_state_dict(best_model['state_dict'])        
model.eval()

# camera prepare
video_capture = cv2.VideoCapture(0)
process_this_frame = True
while True:
    # get a frame from camera
    ret, frame = video_capture.read()
    try:
        # resize the frame to speed up the process
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # transform to RGB image
        rgb_frame = small_frame[:,:,::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_container = torch.empty(len(face_locations),3,128,128)
            for index,face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_container[index] = transform(Image.fromarray(rgb_frame[top:bottom, left:right]))
            # predict face rank
            outputs = model(face_container)
            _, face_ranks = torch.max(outputs, 1)
        process_this_frame = not process_this_frame
        # show the face ranks on the screen
        for (top, right, bottom, left), rank in zip(face_locations, face_ranks):
            top = top * 4
            right = right * 4
            bottom = bottom * 4
            left = left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Rank:'+ classes[rank], (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)
    except:
        # in case that no face is captured
        pass
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # press 'q' to close the window
        break

video_capture.release()
cv2.destroyAllWindows()
