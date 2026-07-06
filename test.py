import torch
print(torch.cuda.is_available())  # 返回 True 表示 GPU 可用，False 则不可用

if torch.cuda.is_available():
    print(f"✅ CUDA 可用！检测到 {torch.cuda.device_count()} 块 GPU")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"   当前 CUDA 版本: {torch.version.cuda}")
else:
    print("❌ CUDA 不可用，请检查驱动和 PyTorch 版本")