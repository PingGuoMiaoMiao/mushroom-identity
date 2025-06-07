import argparse
import json
import os
import time
import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.quantization import quantize_dynamic, QuantType
from process import init_detector, process_img

def optimize_onnx_model(input_model_path, output_model_path):
    """优化ONNX模型"""
    print(f"正在优化模型: {input_model_path}...")
    opt_options = FusionOptions('bert')
    opt_options.enable_attention = False
    
    optimized_model = optimizer.optimize_model(
        input_model_path,
        'bert',
        num_heads=0,
        hidden_size=0,
        optimization_options=opt_options
    )
    
    optimized_model.save_model_to_file(output_model_path)
    print(f"优化后的模型已保存到: {output_model_path}")

def quantize_onnx_model(input_model_path, output_model_path):
    """量化ONNX模型"""
    print(f"正在量化模型: {input_model_path}...")
    quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QUInt8
    )
    print(f"量化后的模型已保存到: {output_model_path}")

def benchmark_model(detector, image_path, num_runs=100):
    """基准测试模型性能"""
    print("正在进行基准测试...")
    
    # 预热
    process_img(detector, image_path)
    
    # 正式测试
    start = time.time()
    for _ in range(num_runs):
        process_img(detector, image_path)
    end = time.time()
    
    avg_time = (end - start) / num_runs
    print(f"平均推理时间: {avg_time * 1000:.2f} ms")

def process_single_image(detector, image_path, output_path):
    """处理单张蘑菇图片"""
    start_time = time.time()
    detections = detector.predict(image_path)
    elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 保存结果
    result = {
        os.path.basename(image_path): {
            "detections": detections,
            "inference_time_ms": elapsed_time
        }
    }
    with open(output_path.replace('.jpg', '.txt'), 'w') as f:
        json.dump(result, f, indent=2)

    # 可视化结果
    detector.visualize(image_path, detections, output_path)
    
    return elapsed_time

def process_folder(detector, input_folder, output_folder, confidence):
    """处理整个蘑菇图片文件夹"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = {}
    inference_times = []
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"result_{filename}")
            
            start_time = time.time()
            detections = detector.predict(image_path)
            elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
            inference_times.append(elapsed_time)
            
            results[filename] = {
                "detections": detections,
                "inference_time_ms": elapsed_time
            }
            
            # 可视化结果
            detector.visualize(image_path, detections, output_path)
    
    # 保存所有结果到一个TXT文件
    output_txt = os.path.join(output_folder, 'detections.txt')
    with open(output_txt, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 计算并打印性能统计信息
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        max_time = max(inference_times)
        min_time = min(inference_times)
        
        print("\n性能统计:")
        print(f"处理图片数量: {len(inference_times)}")
        print(f"平均推理时间: {avg_time:.2f} ms")
        print(f"最大推理时间: {max_time:.2f} ms")
        print(f"最小推理时间: {min_time:.2f} ms")
        
        # 将统计信息也保存到文件中
        stats = {
            "total_images": len(inference_times),
            "avg_inference_time_ms": avg_time,
            "max_inference_time_ms": max_time,
            "min_inference_time_ms": min_time,
            "all_inference_times": inference_times
        }
        stats_path = os.path.join(output_folder, 'performance_stats.txt')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
    
    return inference_times

def main():
    parser = argparse.ArgumentParser(description='蘑菇检测系统')
    parser.add_argument('--image', help='输入蘑菇图片路径')
    parser.add_argument('--folder', help='输入蘑菇图片文件夹路径')
    parser.add_argument('--output', required=True, help='输出路径')
    parser.add_argument('--model', default='best.onnx', help='蘑菇检测模型路径')
    parser.add_argument('--confidence', type=float, default=0.25, help='检测置信度阈值')
    
    # 新增优化相关参数
    parser.add_argument('--optimize', action='store_true', help='优化ONNX模型')
    parser.add_argument('--quantize', action='store_true', help='量化ONNX模型')
    parser.add_argument('--benchmark', action='store_true', help='运行基准测试')
    
    args = parser.parse_args()

    # 模型优化处理
    model_path = args.model
    if args.optimize:
        optimized_path = args.model.replace('.onnx', '_optimized.onnx')
        optimize_onnx_model(args.model, optimized_path)
        model_path = optimized_path
    
    if args.quantize:
        quantized_path = model_path.replace('.onnx', '_quantized.onnx')
        quantize_onnx_model(model_path, quantized_path)
        model_path = quantized_path

    # 初始化检测器
    detector = init_detector(model_path, confidence=args.confidence)

    # 基准测试
    if args.benchmark and args.image:
        benchmark_model(detector, args.image)

    # 处理图像
    if args.image:
        elapsed_time = process_single_image(detector, args.image, args.output)
        print(f"\n单张图片处理完成，推理时间: {elapsed_time:.2f} ms")
    elif args.folder:
        inference_times = process_folder(detector, args.folder, args.output, args.confidence)
    else:
        print("请指定 --image 或 --folder 参数")

if __name__ == '__main__':
    main()
