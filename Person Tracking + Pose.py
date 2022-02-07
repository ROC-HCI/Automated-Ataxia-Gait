#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required packages/modules first

import os
import shutil
from PIL import Image
import numpy as np
import torch
import torchvision
import cv2
from torchvision import transforms as T
from MOT.sort.sort import *


# In[3]:


# Download the pretrained Faster R-CNN model from torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.cuda()
model.eval()


# In[5]:


def crop_coordinate(img, x1,y1,x2,y2):
    h = int(-y1+y2+0.5)
    w = int(-x1+x2+0.5)
    y = int(y1+0.5)
    x = int(x1+0.5)
    b = int(0.05*h)
    cropped_img = img[max(0,y-b):min(y+h+b, len(img)), max(0,x-b):min(x+w+b, len(img[0]))]
    #   img = img*0
    #   img[max(0,y-b):min(y+h+b, len(img)), max(0,x-b):min(x+w+b, len(img[0]))] = cropped_img
    return cropped_img


# In[6]:


import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# In[7]:


transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform

def format_pred(img, min_score):
  image = Image.fromarray(img) # Load the image
  image = transform(image).cuda() # Apply the transform to the image
  pred = model([image])
  mask = (pred[0]['labels'] == 1) & (pred[0]['scores'] > min_score)
  boxes = pred[0]['boxes'][mask].tolist()
  labels = pred[0]['labels'][mask].tolist()
  scores = pred[0]['scores'][mask].tolist()
  
  det_list = []
  for i in range(len(boxes)):
    box = boxes[i]
    label = labels[i]
    score = scores[i]
    det_list.append({'bbox':box, 'labels':1, 'scores':score})

  return det_list


# In[8]:


video_name_list = os.listdir('6s_cropped_video')
folder_name = "6s_cropped_video/"
out_folder = '6s_output/'


# In[9]:


# import cv2
import numpy as np
# import Sort

mot_tracker = Sort()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(folder_name+video_name_list[47])

id_dict = {}

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


outdir_content = os.listdir('MOT/save/')
if len(outdir_content) != 0:
    for dir in outdir_content:
        shutil.rmtree('MOT/save/'+dir)

i = 0
# Read until video is completed
target_h = 800
target_w = int(target_h/2)
# out = cv2.VideoWriter('cropped_patient.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (target_w,target_h))

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, det_img = cap.read()
  if ret == True:
    
    # results = pose.process(det_img)
    arrlist = []
    det_result = format_pred(det_img, 0.9) #data[key] 
    # find the format from det_result and put the info from prediction

    for info in det_result:
        bbox = info['bbox']
        labels = info['labels']
        scores = info['scores']
        templist = bbox+[scores]
        
        if labels == 1: # label 1 is a person in MS COCO Dataset
            arrlist.append(templist)
            
    track_bbs_ids = mot_tracker.update(np.array(arrlist))
    key = '00'+str(i)
    key = key[-3:]
    print(key)
    
    for j in range(track_bbs_ids.shape[0]):  
        ele = track_bbs_ids[j, :]
        x = int(ele[0])
        y = int(ele[1])
        x2 = int(ele[2])
        y2 = int(ele[3])
        track_label = str(int(ele[4]))
        
        cropped_img = crop_coordinate(det_img, x,y,x2,y2)

        if not os.path.exists('MOT/save/'+track_label):
            id_dict[track_label] = []
            os.makedirs('MOT/save/'+track_label)
        
        
        id_dict[track_label].append(y2-y)
        
#         results = pose.process(cropped_img.copy())
        # results = pose.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        # annotated_image = cropped_img.copy()
        # mp_drawing.draw_landmarks(
        #     annotated_image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS
        # )

        # Resize image to have the same height
        h, w, channel = cropped_img.shape

        new_w = int(target_h*w/h+0.5)
        rescaled = cv2.resize(cropped_img, (new_w, target_h))

        template_img = np.zeros([target_h, target_w, channel])#+255
        if new_w > target_w:
            crop_x = int((new_w-target_w)/2) + new_w%2
            template_img[:,:] = rescaled[:, crop_x:crop_x+target_w]
        else:
            pad_x = int((target_w-new_w)/2+0.5)
            template_img[:, pad_x:pad_x + new_w] = rescaled

        cv2.imwrite('MOT/save/'+track_label+'/per_'+track_label+'_'+key+'.png', template_img)
        # if track_label==1:
        # out.write(template_img)

    i+=1
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture and video write objects
cap.release()
# out.release()

# Closes all the frames
cv2.destroyAllWindows()
    
del mot_tracker


# In[10]:


max_sum = 0
max_key = ''
id_sum = {}
for key in id_dict:
    frame_list = id_dict[key]
    last_frame = frame_list[0]
    sum_reduction = 0
    for h in frame_list:
        reduction = last_frame - h
        last_frame = h
        sum_reduction += reduction
    id_sum[key] = sum_reduction
    if sum_reduction > max_sum:
        max_sum = sum_reduction
        max_key = key

print(max_key, max_sum)




# In[11]:


print(id_sum)


# In[ ]:

### Show height map
# from matplotlib import pyplot as plt
# plt.xlabel("Frames")
# plt.ylabel("Height")
# plt.title("Frame vs Height of a person")
# plt.plot(id_dict[max_key])
# plt.legend()
# plt.show()



