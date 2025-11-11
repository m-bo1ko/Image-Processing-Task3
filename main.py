import sys
import cv2
import math
import random
import numpy as np

def ensure_3d(img):
    if img.ndim == 2:
        return img[:, :, np.newaxis]
    return img

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return ensure_3d(img)

def save_image(path, img):
    img = ensure_3d(img)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(path, img)

def brightness(img, value):
    img = ensure_3d(img)
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                new_val = int(img[y, x, ch]) + int(value)
                if new_val < 0:
                    new_val = 0
                elif new_val > 255:
                    new_val = 255
                result[y, x, ch] = new_val
    return result

def contrast(img, value):
    img = ensure_3d(img)
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    mean = np.mean(img)

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                new_val = (img[y, x, ch] - mean) * value + mean
                if new_val < 0:
                    new_val = 0
                elif new_val > 255:
                    new_val = 255
                result[y, x, ch] = int(new_val)
    return result

def negative(img):
    img = ensure_3d(img)
    return 255 - img

def hflip(img):
    img = ensure_3d(img)
    return img[:, ::-1, :]

def vflip(img):
    img = ensure_3d(img)
    return img[::-1, :, :]

def dflip(img):
    img = ensure_3d(img)
    h, w, c = img.shape
    result = np.zeros((w, h, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                result[x, y, ch] = img[y, x, ch]
    return result

def shrink(img, factor):
    img = ensure_3d(img)
    h, w, c = img.shape
    new_h = int(h / factor)
    new_w = int(w / factor)
    result = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for y in range(new_h):
        for x in range(new_w):
            src_y = int(y * factor)
            src_x = int(x * factor)
            result[y, x] = img[src_y, src_x]
    return result

def enlarge(img, factor):
    img = ensure_3d(img)
    h, w, c = img.shape
    new_h = int(h * factor)
    new_w = int(w * factor)
    result = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for dy in range(factor):
                for dx in range(factor):
                    new_y = y * factor + dy
                    new_x = x * factor + dx
                    if new_y < new_h and new_x < new_w:
                        result[new_y, new_x] = img[y, x]
    return result

def median_filter(img, kernel_size):
    img = ensure_3d(img)
    h, w, c = img.shape
    pad = kernel_size // 2
    padded = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.uint8)

    for y in range(h + 2 * pad):
        for x in range(w + 2 * pad):
            src_y = y - pad
            src_x = x - pad
            if src_y < 0:
                src_y = -src_y
            elif src_y >= h:
                src_y = 2 * h - src_y - 2
            if src_x < 0:
                src_x = -src_x
            elif src_x >= w:
                src_x = 2 * w - src_x - 2
            padded[y, x] = img[src_y, src_x]

    result = np.zeros((h, w, c), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                window = []
                for ky in range(-pad, pad + 1):
                    for kx in range(-pad, pad + 1):
                        window.append(padded[y + pad + ky, x + pad + kx, ch])
                window.sort()
                result[y, x, ch] = window[(kernel_size * kernel_size) // 2]

    return result

def gmean_filter(img, size):
    img = ensure_3d(img)
    h, w, c = img.shape
    pad = size // 2
    padded = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.uint8)

    for y in range(h + 2 * pad):
        for x in range(w + 2 * pad):
            src_y = y - pad
            src_x = x - pad
            if src_y < 0:
                src_y = -src_y
            elif src_y >= h:
                src_y = 2 * h - src_y - 2
            if src_x < 0:
                src_x = -src_x
            elif src_x >= w:
                src_x = 2 * w - src_x - 2
            padded[y, x] = img[src_y, src_x]

    result = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                product = 1.0
                count = 0
                for ky in range(-pad, pad + 1):
                    for kx in range(-pad, pad + 1):
                        val = int(padded[y + pad + ky, x + pad + kx, ch])
                        if val == 0:
                            val = 1
                        product *= val
                        count += 1
                geom = product ** (1.0 / count)
                if geom > 255:
                    geom = 255
                result[y, x, ch] = int(geom)
    return result

def add_gaussian_noise(img, sigma):
    img = ensure_3d(img)
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                noise = random.gauss(0, sigma)
                val = int(img[y, x, ch]) + noise
                if val < 0:
                    val = 0
                elif val > 255:
                    val = 255
                result[y, x, ch] = int(val)
    return result

def add_salt_pepper_noise(img, prob=0.05):
    img = ensure_3d(img)
    h, w, c = img.shape
    result = img.copy()

    total_pixels = h * w
    num_salt = int(total_pixels * prob)
    num_pepper = int(total_pixels * prob)

    for _ in range(num_salt):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        result[y, x] = 255

    for _ in range(num_pepper):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        result[y, x] = 0

    return result

def mse(img1, img2):
    img1 = ensure_3d(img1)
    img2 = ensure_3d(img2)
    if img1.shape != img2.shape:
        raise ValueError("Images must have same size")
    diff = (img1.astype(int) - img2.astype(int)) ** 2
    return diff.sum() / diff.size

def pmse(img1, img2):
    return mse(img1, img2) / (255.0 ** 2)

def snr(img1, img2):
    img1 = ensure_3d(img1)
    img2 = ensure_3d(img2)
    if img1.shape != img2.shape:
        raise ValueError("Images must have same size")
    signal = (img1.astype(int) ** 2).sum()
    noise = ((img1.astype(int) - img2.astype(int)) ** 2).sum()
    if noise == 0:
        return float('inf')
    return 10 * math.log10(signal / noise)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return float('inf')
    return 10 * math.log10((255.0 ** 2) / m)

def md(img1, img2):
    img1 = ensure_3d(img1)
    img2 = ensure_3d(img2)
    if img1.shape != img2.shape:
        raise ValueError("Images must have same size")
    return np.abs(img1.astype(int) - img2.astype(int)).max()

def compute_histogram(img, channel=0):
    # Compute histogram for specified channel (or grayscale)
    if img.ndim == 3:
        data = img[:, :, channel].flatten()
    else:
        data = img.flatten()
    return np.bincount(data, minlength=256).astype(np.float64)

def histogram_rayleigh(img, channel=0, gmin=0, gmax=255, alpha=50.0):
    # Rayleigh histogram equalization (H3)
    if img.ndim == 3:
        img_gray = img[:, :, channel]
    else:
        img_gray = img

    hist = compute_histogram(img_gray)
    N = img_gray.size
    cdf = np.cumsum(hist) / N  # Cumulative distribution function

    # Avoid log(0) by clipping CDF
    cdf = np.clip(cdf, 1e-10, 1.0)

    # Rayleigh transformation: g = sqrt(-2*alpha^2 * ln(1 - CDF))
    transform = np.sqrt(-2 * (alpha ** 2) * np.log(1 - cdf))

    # Build lookup table
    lut = np.zeros(256, dtype=np.float32)
    for f in range(256):
        lut[f] = gmin + (gmax - gmin) * (transform[f] - transform[0]) / (transform[-1] - transform[0] + 1e-8)

    # Apply LUT
    enhanced = lut[img_gray]

    # Clip and convert
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Replace channel if needed
    if img.ndim == 3:
        result = img.copy()
        result[:, :, channel] = enhanced
    else:
        result = enhanced

    return ensure_3d(result)

# --- Image characteristics based on histogram ---
def image_characteristics(img, channel=0):
    # Compute all stats from histogram
    hist = compute_histogram(img, channel)
    N = img.shape[0] * img.shape[1]

    # Mean
    mean = sum(m * hist[m] for m in range(256)) / N

    # Variance
    variance = sum((m - mean) ** 2 * hist[m] for m in range(256)) / N

    # Standard deviation
    stdev = math.sqrt(variance)

    # Variation coefficient I
    varcoi = stdev / mean if mean > 0 else 0.0

    # Asymmetry coefficient
    if stdev > 0:
        skew = sum((m - mean) ** 3 * hist[m] for m in range(256)) / (N * (stdev ** 3))
    else:
        skew = 0.0

    return mean, variance, stdev, varcoi, skew

def main():
    if len(sys.argv) == 1 or "--help" in sys.argv:
        print("""
Command-line image processing

Usage:
    python main.py --command -input=in.png -output=out.png [params]

Commands:
    --brightness        -value=40
    --contrast          -value=1.5
    --negative
    --hflip
    --vflip
    --dflip
    --shrink            -factor=2
    --enlarge           -factor=2
    --median            -kernel=3
    --gmean             -size=3
    --noise-gaussian    -sigma=25
    --noise-saltpepper  -p=0.05
    --mse   -ref=ref.png
    --pmse  -ref=ref.png
    --snr   -ref=ref.png
    --psnr  -ref=ref.png
    --md    -ref=ref.png
""")
        sys.exit(0)

    cmd = None
    args = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            cmd = arg
        elif "=" in arg and arg.startswith("-"):
            key, val = arg[1:].split("=", 1)
            args[key] = val

    if "input" not in args:
        print("Error: -input=... is required")
        sys.exit(1)

    img = load_image(args["input"])

    if cmd == "--brightness":
        val = float(args.get("value", 0))
        result = brightness(img, val)

    elif cmd == "--contrast":
        val = float(args.get("value", 1))
        result = contrast(img, val)

    elif cmd == "--negative":
        result = negative(img)

    elif cmd == "--hflip":
        result = hflip(img)

    elif cmd == "--vflip":
        result = vflip(img)

    elif cmd == "--dflip":
        result = dflip(img)

    elif cmd == "--shrink":
        f = float(args.get("factor", 2))
        result = shrink(img, f)

    elif cmd == "--enlarge":
        f = int(args.get("factor", 2))
        result = enlarge(img, f)

    elif cmd == "--median":
        k = int(args.get("kernel", 3))
        result = median_filter(img, k)

    elif cmd == "--gmean":
        s = int(args.get("size", 3))
        result = gmean_filter(img, s)

    elif cmd == "--noise-gaussian":
        sigma = float(args.get("sigma", 25))
        result = add_gaussian_noise(img, sigma)

    elif cmd == "--noise-saltpepper":
        p = float(args.get("p", 0.05))
        result = add_salt_pepper_noise(img, p)

    elif cmd == "--mse":
        ref = load_image(args["ref"])
        print("MSE:", mse(img, ref))
        return

    elif cmd == "--pmse":
        ref = load_image(args["ref"])
        print("PMSE:", pmse(img, ref))
        return

    elif cmd == "--snr":
        ref = load_image(args["ref"])
        print("SNR:", snr(img, ref))
        return

    elif cmd == "--psnr":
        ref = load_image(args["ref"])
        print("PSNR:", psnr(img, ref))
        return

    elif cmd == "--md":
        ref = load_image(args["ref"])
        print("MD:", md(img, ref))
        return

    elif cmd == "--hrayleigh":
        channel = int(args.get("channel", 0))
        gmin = int(args.get("gmin", 0))
        gmax = int(args.get("gmax", 255))
        alpha = float(args.get("alpha", 50.0))
        result = histogram_rayleigh(img, channel, gmin, gmax, alpha)

    elif cmd in ["--cmean", "--cvariance", "--cstdev", "--cvarcoi", "--casyco"]:
        channel = int(args.get("channel", 0))
        mean, variance, stdev, varcoi, skew = image_characteristics(img, channel)
        if cmd == "--cmean":
            print(f"Mean: {mean:.4f}")
        elif cmd == "--cvariance":
            print(f"Variance: {variance:.4f}")
        elif cmd == "--cstdev":
            print(f"Standard deviation: {stdev:.4f}")
        elif cmd == "--cvarcoi":
            print(f"Variation coefficient I: {varcoi:.4f}")
        elif cmd == "--casyco":
            print(f"Asymmetry coefficient: {skew:.4f}")
        return  # No image output for characteristics

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    if "output" in args:
        save_image(args["output"], result)
        print(f"Saved result to {args['output']}")
    else:
        print("No output file specified (-output=...)")



if __name__ == "__main__":
    main()