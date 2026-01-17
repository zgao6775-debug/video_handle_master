import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    # 获取 PyTorch 编译时使用的 CUDA 版本
    cuda_version = torch.version.cuda
    print(f"PyTorch 编译时使用的 CUDA 版本: {cuda_version}")

    # 获取当前设备的名称
    device_name = torch.cuda.get_device_name(0)
    print(f"当前使用的 GPU: {device_name}")
else:
    print("未检测到可用的 CUDA 设备。PyTorch 正在使用 CPU。")
