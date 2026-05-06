import subprocess, sys

print("=" * 50)
print("KIỂM TRA MÔI TRƯỜNG")
print("=" * 50)

# Python version
print(f"\nPython: {sys.version}")

# PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("⚠ Không có GPU CUDA → chạy bằng CPU sẽ rất chậm")
except ImportError:
    print("❌ Chưa cài PyTorch!")

# mamba_ssm
try:
    import mamba_ssm
    print(f"\nmamba_ssm: ✓ đã cài ({mamba_ssm.__version__})")
except ImportError:
    print(f"\nmamba_ssm: ❌ CHƯA CÀI")
    print("  → Chạy: pip install mamba-ssm causal-conv1d")
except Exception as e:
    print(f"\nmamba_ssm: ⚠ Lỗi khi import: {e}")
    print("  → Có thể cần CUDA để compile")

# einops
try:
    import einops
    print(f"einops:    ✓ ({einops.__version__})")
except ImportError:
    print("einops:    ❌ pip install einops")

# tqdm
try:
    import tqdm
    print(f"tqdm:      ✓")
except ImportError:
    print("tqdm:      ❌ pip install tqdm")

print("\n" + "=" * 50)
print("KẾT LUẬN:")
try:
    import torch
    has_cuda = torch.cuda.is_available()
    try:
        import mamba_ssm
        has_mamba = True
    except:
        has_mamba = False
    
    if has_cuda and has_mamba:
        print("✓ Đủ điều kiện chạy EVSSM đầy đủ!")
    elif not has_cuda:
        print("⚠ Không có GPU → EVSSM sẽ rất chậm (vài phút/ảnh)")
        print("  Gợi ý: dùng script deblur nhẹ hơn (OpenCV / NAFNet-CPU)")
    elif not has_mamba:
        print("⚠ Thiếu mamba_ssm → cần cài hoặc dùng script thay thế")
except:
    pass
print("=" * 50)