import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict

def calculate_iou(box1: Dict, box2: Dict) -> float:
    """计算两个框的IOU(交并比)"""
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
    y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(boxes: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """非极大值抑制(NMS)处理"""
    if len(boxes) == 0:
        return []
    
    # 按置信度从高到低排序
    boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        
        boxes = [
            box for box in boxes 
            if calculate_iou(current, box) < iou_threshold
        ]
    
    return keep

class MushroomDetector:
    def __init__(self, model_path: str, confidence: float = 0.25):
        """初始化蘑菇检测器"""
        # 设置ONNX运行时选项
        options = ort.SessionOptions()
        options.log_severity_level = 3  # 减少日志输出
        
        self.session = ort.InferenceSession(model_path, sess_options=options)
        print("\n蘑菇检测模型输入信息:")
        for input in self.session.get_inputs():
            print(f"  名称: {input.name}, 形状: {input.shape}, 类型: {input.type}")
        print("\n蘑菇检测模型输出信息:")
        for output in self.session.get_outputs():
            print(f"  名称: {output.name}, 形状: {output.shape}, 类型: {output.type}")
        self.input_name = self.session.get_inputs()[0].name
        self.confidence = confidence
        
    def predict(self, img_path: str) -> List[Dict]:
        """预测蘑菇位置"""
        # 读取并预处理图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img_height, img_width = img.shape[:2]
        
        # YOLO格式预处理
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, ...]
        
        # 运行推理
        outputs = self.session.run(None, {self.input_name: img_input})
        
        # 处理模型输出
        detections = []
        if len(outputs) > 0:
            # 使用第一个输出(25200x6)
            output = outputs[0][0]  # 去掉batch维度
            for detection in output:
                x, y, w, h, conf, class_id = detection[:6]
                if conf > self.confidence:
                    # 从640x640归一化坐标转换回原始图像尺寸
                    x_center = x / 640 * img_width
                    y_center = y / 640 * img_height
                    width = w / 640 * img_width
                    height = h / 640 * img_height
                    
                    # 转换为左上角坐标
                    x = int(x_center - width/2)
                    y = int(y_center - height/2)
                    w = int(width)
                    h = int(height)
                    
                    # 确保坐标在合理范围内
                    x = max(0, min(x, img_width-1))
                    y = max(0, min(y, img_height-1))
                    w = max(0, min(w, img_width-1 - x))
                    h = max(0, min(h, img_height-1 - y))
                    
                    if w > 0 and h > 0:  # 确保宽高有效
                        detections.append({
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'confidence': float(conf)
                        })
        
        # 应用非极大值抑制
        detections = non_max_suppression(detections, iou_threshold=0.5)
        
        # 按面积从大到小排序
        detections.sort(key=lambda x: x['w'] * x['h'], reverse=True)
        return detections
    
    def visualize(self, img_path: str, boxes: List[Dict], output_path: str):
        """可视化检测结果"""
        img = cv2.imread(img_path)
        for box in boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"Mushroom {box['confidence']:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        cv2.imwrite(output_path, img)

def init_detector(model_path: str, confidence: float = 0.25) -> MushroomDetector:
    """初始化蘑菇检测器"""
    return MushroomDetector(model_path, confidence)

def process_img(img_path: str) -> List[Dict]:
    """处理单张蘑菇图片(兼容原接口)"""
    # 使用默认模型路径
    model_path = 'best.onnx'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    detector = init_detector(model_path)
    return detector.predict(img_path)

if __name__ == '__main__':
    # 保留原测试代码
    imgs_folder = './imgs/'
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
        print(f"测试图片文件夹 {imgs_folder} 不存在，已创建")
    
    img_paths = [f for f in os.listdir(imgs_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def now():
        return int(time.time()*1000)
    
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = float('inf')
    
    for img_path in img_paths:
        print(img_path, ':')
        last_time = now()
        result = process_img(os.path.join(imgs_folder, img_path))
        run_time = now() - last_time
        print('result:\n', result)
        print('run time: ', run_time, 'ms')
        print()
        
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    
    if img_paths:
        print('\n')
        print('avg time: ', int(count_time/len(img_paths)), 'ms')
        print('max time: ', max_time, 'ms')
        print('min time: ', min_time, 'ms')
    else:
        print(f"没有找到测试图片，请将图片放入 {imgs_folder} 文件夹")
