"""
deblur_images_final.py
======================
Viết lại từ đầu với tư duy chuyên gia xử lý ảnh:

NGUYÊN NHÂN LỖI TRƯỚC:
- EVSSM tile ghép → seam artifact (màu loạn ảnh 1)
- Lucy-Richardson → amplify noise thay vì deblur (hạt ảnh 2)
- Dùng 1 pipeline cho mọi loại ảnh → sai vì VizWiz có nhiều loại blur khác nhau

GIẢI PHÁP MỚI:
1. Tắt hoàn toàn EVSSM tile (gây artifact)
2. Detect loại blur của từng ảnh → chọn pipeline phù hợp
3. Denoise TRƯỚC khi sharpen (thứ tự quan trọng!)
4. Dùng blind deconvolution đúng cách cho defocus blur
5. Chỉ dùng EVSSM nguyên ảnh nếu ảnh đủ nhỏ (< 512px)
"""

import sys, torch, cv2
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm

# =====================================================
#  CẤU HÌNH
# =====================================================
# Dùng EVSSM chỉ khi ảnh nhỏ hơn ngưỡng này (tránh OOM)
EVSSM_MAX_SIDE     = 500    # px — ảnh lớn hơn → bỏ qua EVSSM
DENOISE_STRENGTH   = 5      # 3-10: mức khử noise (thấp=nhẹ, cao=mạnh)
SHARPEN_FINAL      = 1.7    # sharpen cuối (1.0-2.5)
CONTRAST_FINAL     = 1.15   # contrast cuối
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CHECKPOINT = PROJECT_ROOT / 'experiments' / 'pretrained_models' / 'net_g_GoPro.pth'
INPUT_DIR  = Path(__file__).resolve().parent / 'data' / 'images_original'
OUTPUT_DIR = Path(__file__).resolve().parent / 'data' / 'images_deblurred'


# ─────────────────────────────────────────
#  BƯỚC 1: PHÂN TÍCH LOẠI BLUR
# ─────────────────────────────────────────
def analyze_image(img_np: np.ndarray) -> dict:
    """
    Đo các chỉ số của ảnh để chọn pipeline phù hợp.
    Returns: {'blur_score', 'noise_score', 'is_dark', 'blur_type'}
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Blur score: Laplacian variance — thấp = mờ nhiều
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Noise score: so sánh ảnh gốc vs đã smooth
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = np.std(gray.astype(float) - smoothed.astype(float))

    # Độ sáng trung bình
    brightness = np.mean(gray)
    is_dark = brightness < 80

    # Phân loại: nếu blur_score thấp VÀ noise cao → noisy + defocus
    # nếu blur_score thấp VÀ noise thấp → motion/defocus thuần
    blur_type = 'noisy' if noise_score > 8 else 'clean_blur'

    return {
        'blur_score': blur_score,
        'noise_score': noise_score,
        'is_dark': is_dark,
        'blur_type': blur_type,
        'is_very_blurry': blur_score < 50,
    }


# ─────────────────────────────────────────
#  BƯỚC 2: DENOISE (chạy TRƯỚC sharpen)
# ─────────────────────────────────────────
def smart_denoise(img_np: np.ndarray, strength: int, is_dark: bool) -> np.ndarray:
    """
    Khử noise trước khi sharpen.
    Nếu sharpen trước denoise → noise bị khuếch đại (lỗi ảnh 2).
    """
    h = strength
    # fastNlMeans: h=filter strength, templateWindowSize, searchWindowSize
    # Tăng strength nếu ảnh tối (thường nhiều noise hơn)
    if is_dark:
        h = min(h + 3, 15)

    denoised = cv2.fastNlMeansDenoisingColored(
        img_np,
        None,
        h=h,           # luminance noise filter
        hColor=h,      # color noise filter
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised


# ─────────────────────────────────────────
#  BƯỚC 3A: DEFOCUS DEBLUR (ảnh sạch)
# ─────────────────────────────────────────
def defocus_deblur(img_np: np.ndarray) -> np.ndarray:
    """
    Wiener deconvolution với estimated PSF cho defocus blur.
    Hiệu quả hơn Lucy-Richardson và không amplify noise
    vì đã denoise ở bước trước.
    """
    result = np.zeros_like(img_np, dtype=np.float32)

    # Ước lượng PSF dạng disk (defocus = circular blur)
    psf_size = 15
    psf = np.zeros((psf_size, psf_size), dtype=np.float32)
    cv2.circle(psf, (psf_size//2, psf_size//2), psf_size//4, 1, -1)
    psf /= psf.sum()

    noise_power = 0.005  # regularization — tránh ringing artifact

    for c in range(3):
        channel = img_np[:, :, c].astype(np.float32) / 255.0
        H, W = channel.shape

        # FFT-based Wiener filter
        img_fft = np.fft.fft2(channel)
        psf_fft = np.fft.fft2(psf, s=(H, W))
        psf_conj = np.conj(psf_fft)
        psf_power = np.abs(psf_fft) ** 2

        # Wiener filter: H* / (|H|² + SNR⁻¹)
        wiener = psf_conj / (psf_power + noise_power)
        restored = np.real(np.fft.ifft2(img_fft * wiener))
        result[:, :, c] = np.clip(restored, 0, 1) * 255

    return result.astype(np.uint8)


# ─────────────────────────────────────────
#  BƯỚC 3B: EDGE SHARPENING (ảnh có noise)
# ─────────────────────────────────────────
def gentle_sharpen(img_np: np.ndarray) -> np.ndarray:
    """
    Với ảnh nhiều noise: KHÔNG dùng deconvolution.
    Chỉ sharpen cạnh (edge) bằng bilateral + unsharp nhẹ.
    """
    # Bilateral filter giữ cạnh, làm mịn vùng phẳng
    bilateral = cv2.bilateralFilter(img_np, d=7,
                                    sigmaColor=50, sigmaSpace=7)

    # Unsharp mask rất nhẹ (sigma=0.5, strength=0.8)
    blurred = cv2.GaussianBlur(bilateral, (0, 0), 0.5)
    diff = bilateral.astype(np.float32) - blurred.astype(np.float32)

    # Chỉ áp dụng ở cạnh rõ ràng (threshold=8)
    edge_mask = np.abs(diff).max(axis=2, keepdims=True) > 8
    result = np.clip(
        bilateral.astype(np.float32) + 0.8 * diff * edge_mask,
        0, 255
    ).astype(np.uint8)

    return result


# ─────────────────────────────────────────
#  BƯỚC 4: FINAL ENHANCEMENT
# ─────────────────────────────────────────
def final_enhance(img_pil: Image.Image, sharpen: float,
                  contrast: float) -> Image.Image:
    """Sharpen + contrast + CLAHE nhẹ để hoàn thiện."""
    # CLAHE nhẹ
    img_lab = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_pil = Image.fromarray(
        cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    )

    # Sharpen
    img_pil = ImageEnhance.Sharpness(img_pil).enhance(sharpen)

    # Contrast
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)

    return img_pil


# ─────────────────────────────────────────
#  LOAD EVSSM (chỉ dùng cho ảnh nhỏ)
# ─────────────────────────────────────────
model = None
device = torch.device('cpu')
to_tensor = T.ToTensor()
to_pil    = T.ToPILImage()

if (PROJECT_ROOT / 'models' / 'EVSSM.py').exists() and CHECKPOINT.exists():
    try:
        print("Đang load EVSSM...")
        from models.EVSSM import EVSSM
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EVSSM(inp_channels=3, out_channels=3, dim=48,
                      num_blocks=[6, 6, 12], ffn_expansion_factor=3, bias=False)
        ckpt = torch.load(CHECKPOINT, map_location=device)
        state = ckpt.get('params', ckpt.get('params_ema',
                ckpt.get('state_dict', ckpt)))
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        print(f"✓ EVSSM sẵn sàng (chỉ dùng cho ảnh ≤{EVSSM_MAX_SIDE}px)\n")
    except Exception as e:
        print(f"⚠ Không load được EVSSM: {e}\n")
        model = None
else:
    print("⚠ Không tìm thấy EVSSM → dùng CV pipeline\n")


# ─────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────
def process_image(img_pil: Image.Image) -> Image.Image:
    """
    Pipeline thông minh: phân tích → chọn kỹ thuật phù hợp
    """
    img_np = np.array(img_pil)
    W, H = img_pil.size
    info = analyze_image(img_np)

    # ── Bước 1: Denoise TRƯỚC (luôn luôn) ──
    img_np = smart_denoise(img_np,
                           strength=DENOISE_STRENGTH,
                           is_dark=info['is_dark'])

    # ── Bước 2A: EVSSM nếu ảnh nhỏ đủ ──
    evssm_used = False
    if model is not None and max(W, H) <= EVSSM_MAX_SIDE:
        try:
            t = to_tensor(Image.fromarray(img_np)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t).clamp(0, 1).squeeze(0).cpu()
            img_np = np.array(to_pil(out))
            evssm_used = True
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass  # fallback to CV methods

    # ── Bước 2B: CV deblur nếu không dùng EVSSM ──
    if not evssm_used:
        if info['blur_type'] == 'clean_blur':
            # Defocus blur thuần → Wiener deconvolution
            img_np = defocus_deblur(img_np)
        else:
            # Nhiều noise → gentle edge sharpen
            img_np = gentle_sharpen(img_np)

    # ── Bước 3: Final enhance ──
    img_pil = final_enhance(
        Image.fromarray(img_np),
        sharpen=SHARPEN_FINAL,
        contrast=CONTRAST_FINAL
    )

    return img_pil


# ─────────────────────────────────────────
#  CHẠY
# ─────────────────────────────────────────
if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
    print(f"❌ Không có ảnh: {INPUT_DIR}")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

imgs = (list(INPUT_DIR.glob('*.jpg')) +
        list(INPUT_DIR.glob('*.jpeg')) +
        list(INPUT_DIR.glob('*.png')))

done  = sum(1 for p in imgs if (OUTPUT_DIR / p.name).exists())
print(f"Tổng: {len(imgs)} | Đã xử lý: {done} | Còn: {len(imgs)-done}")

errors = 0
for img_path in tqdm(imgs, desc='Đang xử lý'):
    dest = OUTPUT_DIR / img_path.name
    if dest.exists():
        continue
    try:
        img_pil  = Image.open(img_path).convert('RGB')
        result   = process_image(img_pil)
        result.save(dest, quality=95)
    except Exception as e:
        errors += 1
        tqdm.write(f'  ❌ {img_path.name}: {e}')

print(f'\n✓ Xong! {len(imgs)-errors}/{len(imgs)} ảnh')
print(f'Kết quả: {OUTPUT_DIR}')