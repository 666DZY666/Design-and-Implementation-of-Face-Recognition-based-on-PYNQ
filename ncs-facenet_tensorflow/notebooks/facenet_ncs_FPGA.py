import numpy
import cv2

# 差异度（欧氏距离）阈值——越小，识别越准|越难
FACE_MATCH_THRESHOLD = 0.55
matching = False
v = 0
text_color = (0, 255, 0)
rectangle_color = (0, 255, 0)

# ！！！图像特征提取（128维特征值—列表）——在NCS中调用模型-facenet.graph ！！！
def run_inference(image_to_classify, facenet_graph):
    
    # 图像预处理———大小、通道顺序
    resized_image = preprocess_image(image_to_classify)
    # 图像传至 NCS
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)
    # ！！！特征提取 ！！！
    output, userobj = facenet_graph.GetResult()
    # 返回特征值
    return output

# # 识别结果显示
# def overlay_on_image(display_image, matching):
#     rect_width = 10
#     offset = int(rect_width/2)
#     #if (image_info != None):
#     #    cv2.putText(display_image, image_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#     if (matching):
#         # match, green rectangle
#         cv2.rectangle(display_image, (0+offset, 0+offset),
#                       (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
#                       (0, 255, 0), 10)
#     else:
#         # not a match, red rectangle
#         cv2.rectangle(display_image, (0+offset, 0+offset),
#                       (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
#                       (0, 0, 255), 10)
        
# create a preprocessed image from the source image that matches the
# network expectations and return it

# 图像格式标准化调整
def whiten_image(source_image):
    
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# # 图像预处理(ARM)———大小、通道顺序、格式调整
# def preprocess_image(src):
    
#     # 大小调整
#     NETWORK_WIDTH = 160
#     NETWORK_HEIGHT = 160
#     # resize
#     preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))
#     # BGR——>RGB
#     preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
#     # 格式调整
#     preprocessed_image = whiten_image(preprocessed_image)
#     # 返回预处理后的图像
#     return preprocessed_image

# 图像预处理(FPGA)———大小、通道顺序、格式调整
def preprocess_image(src):
    
    # 大小调整
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    # resize
    xv2.resize(src, preprocessed_image)
    # convert：BGR——>RGB
    xv2.cvtColor(preprocessed_image)
    # 格式调整
    preprocessed_image = whiten_image(preprocessed_image)
    # 返回预处理后的图像
    return preprocessed_image

# 人脸识别比对（计算欧氏距离，越小越匹配）
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    # 欧氏距离
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    print('Total Difference is: ' + str(total_diff))
    # ！！！识别———判别是否是同一人脸 ！！！
    if (total_diff < FACE_MATCH_THRESHOLD):
        return True
    # differences between faces was over the threshold above so
    # they didn't match.
    else:
        return False

# ！（批量）特征提取 ！
def feature(targets_list, temp_list, graph):
    
    i = 0
    for target in targets_list :
        # 读取一个目标
        target = cv2.imread('/home/xilinx/jupyter_notebooks/ncs-facenet_tensorflow/targets/' + target)
        # 目标特征集合（字典）
        temp_list[i] = run_inference(target, graph)
        i = i + 1
    # 返回特征集合
    return temp_list
    
# ！！！人脸识别（顶层模块）！！！
def run_images(targets_feature, targets_list, graph, input_frame1, input_frame2):
    
    global matching,v,text_color,rectangle_color

    rect_width = 10
    offset = int(rect_width/2)
  
    l = len(targets_feature)
    f = 0
    i = 0
        
    # ！待检测图像特征提取 ！
    input_feature = run_inference(input_frame2, graph)
    
    # ！！！人脸识别 —— 遍历目标特征集合（字典），循环比对 ！！！
#    for target_feature in targets_feature :
    while(i < l):
        # 匹配
        if (face_match(targets_feature[i], input_feature)):
            v = v + 1
            if(v == 3):
                v = 0
                matching = True
                text_color = (0, 255, 0)
                rectangle_color = (0, 255, 0)
            break
        # 不匹配
        else:
            f =f + 1
        if f == l:
            v = 0
            matching = False
            rectangle_color = (0, 0, 255)
        i = i + 1
    # 匹配 —— 绿框+名字    
    if (matching):
        n = len(targets_list[i])
        # match, green rectangle
        cv2.rectangle(input_frame1, (0+offset, 0+offset),(input_frame1.shape[1]-offset-1, input_frame1.shape[0]-offset-1),rectangle_color, 10)
        cv2.putText(input_frame1, targets_list[i][:n - 4], (66, 88), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)
    # 不匹配 —— 红框
    else:
        # not a match, red rectangle
        cv2.rectangle(input_frame1, (0+offset, 0+offset),(input_frame1.shape[1]-offset-1, input_frame1.shape[0]-offset-1),rectangle_color, 10)
        
    return input_frame1