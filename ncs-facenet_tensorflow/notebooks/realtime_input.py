import cv2
import os
import facenet_ncs
from time import sleep

# 人脸录入路径
targets_list_dir = '/home/xilinx/jupyter_notebooks/ncs-facenet_tensorflow/targets'

Delay = 2.0
rectangle_color = (255, 0, 0)
text_color = (255, 0, 0)

rect_width = 10
offset = int(rect_width/2)

def realtime_input(frame, hdmi_out, graph, videoIn):
    
    ret,real_frame = videoIn.read()
    real_frame = real_frame[80:400, 200:440]
    
    cv2.rectangle(frame, (0+offset, 0+offset),(frame.shape[1]-offset-1, frame.shape[0]-offset-1),rectangle_color, 10)
    cv2.putText(frame, 'Input...', (66, 88), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)
    frame_out = hdmi_out.newframe()
    frame_out[:,:,:] = frame[:,:,:]
    hdmi_out.writeframe(frame_out)
    print('\r\n')
    print('!!!Please Input Name!!!')
    name = input()
    print('\r\n')
    cv2.imwrite('/home/xilinx/jupyter_notebooks/ncs-facenet_tensorflow/targets/' + name + '.jpg', real_frame)
    # 获取人脸目标路径
    temp_list = {}
    targets_list = os.listdir(targets_list_dir)
    targets_list = [i for i in targets_list if i.endswith('.jpg')]
    # ！！！人脸录入 ！！！（获取目标人脸特征集）
    targets_feature = facenet_ncs.feature(targets_list, temp_list, graph)
    cv2.putText(frame, 'Input...OK OK OK', (66, 88), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)
    frame_out = hdmi_out.newframe()
    frame_out[:,:,:] = frame[:,:,:]
    hdmi_out.writeframe(frame_out)
    sleep(Delay)
    return targets_feature, targets_list