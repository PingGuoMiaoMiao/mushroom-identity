## 复赛仓库格式

```
├── src/ #项目代码
├── doc/ #文档
├── slide/ #PPT
├── video/ #演示视频
├── run.sh #启动脚本（可选）
└── README.md #说明启动流程
```

**项目代码格式**

```
src/
├── main.py          # 主程序入口
├── process.py       # 核心处理逻辑
├── models/
│   └── best.onnx    # 模型文件
├── yolov5_env       # 虚拟依赖
├── images/          # 输入图片
├── requirement.txt  # 依赖
└── outputs/         # 输出目录
```

#### **​1. 环境准备​**

确保已安装 `Python 3.8+` 和 `pip`，并进入项目目录：

```
cd mushroom-identity/src
```
# 
#### ​**​2. 激活虚拟环境​**​

```
python -m venv yolov5_env 
source yolov5_env/bin/activate  
```
✅ ​**​验证激活成功​**​：命令行前缀应显示 `(yolov5_env)`
### 方法1：通过 `pip` 逐个安装（推荐）

`# 激活虚拟环境后执行（确保在yolov5_env环境下) 
```
pip install onnx onnxruntime opencv-python numpy
```
### 方法2：通过 `requirements.txt` 批量安装

1. 在项目目录（如 `src/`）下创建 `requirements.txt` 文件，内容如下：
```
onnxruntime>=1.15.0
opencv-python>=4.7.0
numpy>=1.23.0 
```
   
2. 执行安装命令：
```
pip install -r requirements.txt
```
验证安装是否成功
```
 pip show onnxruntime opencv-python numpy # 检查已安装的版本
```

#### ​**​3. 检查依赖​**​

```
pip list | grep -E "onnxruntime|opencv-python|numpy"
```

🔧 ​**​若缺少依赖​**​，执行：

```
pip install -r requirements.txt
```

#### ​**​4. 运行程序​**

| 运行相关参数       | 作用          |
| ------------ | ----------- |
| --output     | 输出路径        |
| --model      | 蘑菇检测模型路径    |
| --folder     | 输入蘑菇图片文件夹路径 |
| --confidence | 检测置信度阈值     |
| --image      | 输入蘑菇图片路径    |

| 优化相关的参数     | 作用       |
| ----------- | -------- |
| --optimize  | 优化ONNX模型 |
| --quantize  | 量化ONNX模型 |
| --benchmark | 运行基准测试   |


```
python main.py --folder images --output output --model model/best.onnx
```

✅ ​**​预期输出​**​：

- 终端打印推理日志（如 `Detection completed!`）
- 结果图片保存至 `./output/`

#### ​**​5. 退出虚拟环境​**​

```
deactivate
```

---

### ​**​一键启动​**(暂未实现)​

若已配置 `run.sh`，可直接执行：

`chmod +x run.sh  # 添加执行权限（仅首次需要） ./run.sh         # 自动完成环境激活和运行`

​**​`run.sh` 示例内容​**​：

```
#!/bin/bash
cd src 
source yolov5_env/bin/activate
python main.py --folder images --output output --model model/best.onnx 
deactivate
```
