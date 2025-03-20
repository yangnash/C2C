import gradio as gr
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import app

from dataset import gtransform

def preprocess_frame(frame):
    # 将numpy数组转换为PIL图像
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
        
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
            gtransform.GroupResize(256),
            gtransform.GroupCenterCrop(224),
            gtransform.ToTensor(),
            gtransform.GroupNormalize(img_mean, img_std)
        ])
    return transform([frame])  # 将单帧包装成列表传入

# 读取视频并提取帧
def process_video(video_path, in_duration=8, seg_length=1):
    cap = cv2.VideoCapture(video_path)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    
    # 先读取所有帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    total_frames = len(all_frames)
    cap.release()
    
    # 计算采样索引
    if in_duration == 1:
        indices = [total_frames // 2]
    elif total_frames <= in_duration:
        indices = [i * total_frames // in_duration for i in range(in_duration)]
    else:
        offset = (total_frames / in_duration - seg_length) / 2.0
        indices = [int(i * total_frames / in_duration + offset + j)
                    for i in range(in_duration)
                    for j in range(seg_length)]
    print(total_frames,len(all_frames),indices)
    # 根据索引选择帧并进行预处理
    selected_frames = []
    for idx in indices:
        if 0 <= idx < len(all_frames):  # 确保索引有效
            frame = all_frames[idx]
            processed_frame = preprocess_frame(frame)
            selected_frames.append(processed_frame[0])  # preprocess_frame返回的是列表
            
    if not selected_frames:
        raise ValueError("没有成功读取到任何有效帧")
        
    return torch.stack(selected_frames)

# 定义Gradio接口
def recognize_action(video_path):
    frames = process_video(video_path)
    frames = torch.Tensor(frames).unsqueeze(0).cuda()  # 添加批次维度
    # app.predict(frames)
    return app.predict(frames)

# 创建Gradio界面
iface = gr.Interface(
    fn=recognize_action,
    inputs=gr.Video(label="Upload a video"),
    outputs="text",
    title="TSM Action Recognition",
    examples=[['/opt/data/private/bishe/C2C/codes/examples/10077.webm']]
)

# 启动Gradio应用
iface.launch()