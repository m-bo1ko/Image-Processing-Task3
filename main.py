import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def save_image(path, img):
    cv2.imwrite(path, img)


def to_grayscale(img):
    h, w, c = img.shape
    gray = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            s = 0
            for ch in range(c):
                s += int(img[y, x, ch])
            gray[y, x] = s // c
    return gray


def binarize(img_gray, threshold):
    h, w = img_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            mask[y, x] = 1 if int(img_gray[y, x]) >= threshold else 0
    return mask


def mask_to_color(mask):
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = 255 if mask[y, x] != 0 else 0
            for ch in range(3):
                result[y, x, ch] = v
    return result


def brightness(img, value):
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
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    total = 0
    count = h * w * c
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                total += int(img[y, x, ch])
    mean = total / count

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
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                result[y, x, ch] = 255 - img[y, x, ch]
    return result


def hflip(img):
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                result[y, x, ch] = img[y, w - 1 - x, ch]
    return result


def vflip(img):
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                result[y, x, ch] = img[h - 1 - y, x, ch]
    return result


def dflip(img):
    h, w, c = img.shape
    result = np.zeros((w, h, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                result[x, y, ch] = img[y, x, ch]
    return result


def shrink(img, factor):
    h, w, c = img.shape
    new_h = int(h / factor)
    new_w = int(w / factor)
    result = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for y in range(new_h):
        for x in range(new_w):
            src_y = int(y * factor)
            src_x = int(x * factor)
            for ch in range(c):
                result[y, x, ch] = img[src_y, src_x, ch]
    return result


def enlarge(img, factor):
    h, w, c = img.shape
    new_h = int(h * factor)
    new_w = int(w * factor)
    result = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                for dy in range(factor):
                    for dx in range(factor):
                        new_y = y * factor + dy
                        new_x = x * factor + dx
                        if new_y < new_h and new_x < new_w:
                            result[new_y, new_x, ch] = img[y, x, ch]
    return result


def median_filter(img, kernel_size):
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
            for ch in range(c):
                padded[y, x, ch] = img[src_y, src_x, ch]

    result = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                window = []
                for ky in range(-pad, pad + 1):
                    for kx in range(-pad, pad + 1):
                        window.append(int(padded[y + pad + ky, x + pad + kx, ch]))
                window.sort()
                median_val = window[len(window) // 2]
                result[y, x, ch] = median_val
    return result


def gmean_filter(img, size):
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
            for ch in range(c):
                padded[y, x, ch] = img[src_y, src_x, ch]

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
    import random
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
    h, w, c = img.shape
    result = img.copy().astype(np.uint8)

    total_pixels = h * w
    num_salt = int(total_pixels * prob)
    num_pepper = int(total_pixels * prob)

    for _ in range(num_salt):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        for ch in range(c):
            result[y, x, ch] = 255

    for _ in range(num_pepper):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        for ch in range(c):
            result[y, x, ch] = 0

    return result


def mse(img1, img2):
    h, w, c = img1.shape
    total = 0
    count = h * w * c
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                diff = int(img1[y, x, ch]) - int(img2[y, x, ch])
                total += diff * diff
    return total / count


def pmse(img1, img2):
    h, w, c = img1.shape
    total = 0
    count = h * w * c
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                diff = int(img1[y, x, ch]) - int(img2[y, x, ch])
                total += diff * diff
    mse_val = total / count
    pmse_val = mse_val / (255.0 ** 2)
    return pmse_val


def snr(img1, img2):
    h, w, c = img1.shape
    signal_sum = 0
    noise_sum = 0
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                s = int(img1[y, x, ch])
                n = int(img1[y, x, ch]) - int(img2[y, x, ch])
                signal_sum += s * s
                noise_sum += n * n
    if noise_sum == 0:
        return float('inf')
    return 10 * np.log10(signal_sum / noise_sum)


def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    return 10 * math.log10((255.0 ** 2) / mse_val)


def md(img1, img2):
    h, w, c = img1.shape
    max_diff = 0
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                diff = abs(int(img1[y, x, ch]) - int(img2[y, x, ch]))
                if diff > max_diff:
                    max_diff = diff
    return max_diff


def compute_histogram(img, ch):
    img = img.astype(np.float64)
    h, w, c = img.shape
    hist = [0] * 256
    for y in range(h):
        for x in range(w):
            v = int(img[y, x, ch])
            if 0 <= v < 256:
                hist[v] += 1
    return hist


def save_histogram(img, ch, filename="histogram.png"):
    hist = compute_histogram(img, ch)
    plt.figure(figsize=(8, 4))
    plt.bar(range(256), hist, color='gray')
    plt.title(f"Histogram for channel {ch}")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Histogram saved as {filename}")


def cmean(img):
    img = img.astype(np.float64)
    h, w, c = img.shape
    means = []
    for ch in range(c):
        total = 0.0
        for y in range(h):
            for x in range(w):
                total += img[y, x, ch]
        means.append(total / (h * w))
    return means


def cvariance(img):
    img = img.astype(np.float64)
    h, w, c = img.shape
    vars_ = []
    means = cmean(img)
    for ch in range(c):
        mean = means[ch]
        total = 0.0
        for y in range(h):
            for x in range(w):
                diff = img[y, x, ch] - mean
                total += diff * diff
        vars_.append(total / (h * w))
    return vars_


def cstdev(img):
    img = img.astype(np.float64)
    vars_ = cvariance(img)
    return [math.sqrt(v) for v in vars_]


def cvarcoi(img):
    img = img.astype(np.float64)
    means = cmean(img)
    stdevs = cstdev(img)
    result = []
    for m, s in zip(means, stdevs):
        result.append(s / m if m != 0 else 0)
    return result


def casyco(img):
    img = img.astype(np.float64)
    h, w, c = img.shape
    asym = []
    means = cmean(img)
    stdevs = cstdev(img)
    for ch in range(c):
        mean = means[ch]
        std = stdevs[ch]
        if std == 0:
            asym.append(0)
            continue
        total = 0.0
        for y in range(h):
            for x in range(w):
                total += ((img[y, x, ch] - mean) / std) ** 3
        asym.append(total / (h * w))
    return asym


def cflatco(img):
    img = img.astype(np.float64)
    h, w, c = img.shape
    flat = []
    means = cmean(img)
    stdevs = cstdev(img)
    for ch in range(c):
        mean = means[ch]
        std = stdevs[ch]
        if std == 0:
            flat.append(0)
            continue
        total = 0.0
        for y in range(h):
            for x in range(w):
                total += ((img[y, x, ch] - mean) / std) ** 4
        flat.append(total / (h * w) - 3)
    return flat


def cvarcoii(img):
    img = img.astype(np.float64)
    h, w, c = img.shape
    result = []
    N = h * w
    for ch in range(c):
        hist = compute_histogram(img, ch)
        s = sum(h * h for h in hist)
        result.append((s) / (N * N))
    return result


def centropy(img):
    img = img.astype(np.float64)
    h, w, c = img.shape
    entropies = []
    for ch in range(c):
        hist = compute_histogram(img, ch)
        total = h * w
        e = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                e -= p * math.log(p, 2)
        entropies.append(e)
    return entropies


def hrayleigh(img, alpha):
    img = img.astype(np.float64)
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.uint8)
    total = h * w

    for ch in range(c):
        hist = compute_histogram(img, ch)
        CDF = []
        s = 0
        for v in hist:
            s += v
            CDF.append(s / total)

        mapping = []
        for p in CDF:
            if p >= 1: p = 0.999999
            val = math.sqrt(-2 * alpha * alpha * math.log(1 - p))
            mapping.append(min(255, max(0, int(val))))

        for y in range(h):
            for x in range(w):
                result[y, x, ch] = mapping[int(img[y, x, ch])]

    return result


def universal_convolution(img, kernel):
    img = img.astype(np.float64)
    h, w, c = img.shape
    kh, kw = len(kernel), len(kernel[0])
    pad_h, pad_w = kh // 2, kw // 2

    result = np.zeros((h, w, c), dtype=np.float64)
    ksum = np.sum(kernel)

    for ch in range(c):
        for y in range(h):
            for x in range(w):
                s = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        ny = min(max(y + ky - pad_h, 0), h - 1)
                        nx = min(max(x + kx - pad_w, 0), w - 1)
                        s += img[ny, nx, ch] * kernel[ky][kx]
                if ksum != 0:
                    result[y, x, ch] = s / ksum
                else:
                    result[y, x, ch] = s
    return result


def optimized_slowpass(img):
    img = img.astype(np.float64)
    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.float64)
    ksum = 16.0

    for ch in range(c):
        for y in range(h):
            for x in range(w):
                y0, y1, y2 = max(y - 1, 0), y, min(y + 1, h - 1)
                x0, x1, x2 = max(x - 1, 0), x, min(x + 1, w - 1)

                s = (img[y0, x0, ch] + img[y0, x2, ch] +
                     img[y2, x0, ch] + img[y2, x2, ch])
                s += (img[y0, x1, ch] + img[y1, x0, ch] +
                      img[y1, x2, ch] + img[y2, x1, ch] + img[y0, x1, ch] + img[y1, x0, ch] +
                      img[y1, x2, ch] + img[y2, x1, ch])
                s += img[y1, x1, ch] * 4

                result[y, x, ch] = s / ksum

    return result


def oll(img, eps=1e-12):
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    h, w, c = img.shape
    result = np.zeros((h, w, c), dtype=np.float64)

    neighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for ch in range(c):
        for y in range(h):
            for x in range(w):
                center = img[y, x, ch]
                prod = 1.0
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        prod *= img[ny, nx, ch]
                    else:
                        prod *= center

                val = center ** 4 / (prod + eps)

                result[y, x, ch] = 0.25 * abs(np.log(val + eps))

    max_val = result.max()
    if max_val > 0:
        result = result / max_val
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result


def manual_pad(img, pad_h, pad_w):
    h, w = img.shape
    out = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.uint8)
    out[pad_h:pad_h + h, pad_w:pad_w + w] = img
    return out


def dilation(img, se):
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    img_p = manual_pad(img, ph, pw)
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            region = img_p[i:i + sh, j:j + sw]
            if np.any(np.logical_and(region, se)):
                out[i, j] = 1
    return out


def erosion(img, se):
    sh, sw = se.shape
    ph, pw = sh // 2, sw // 2
    img_p = manual_pad(img, ph, pw)
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            region = img_p[i:i + sh, j:j + sw]
            if np.all(np.logical_or(np.logical_not(se), region)):
                out[i, j] = 1
    return out


def opening(img, se):
    return dilation(erosion(img, se), se)


def closing(img, se):
    return erosion(dilation(img, se), se)


def hit_or_miss(image, se_present, se_absent):
    eroded_img = erosion(image, se_present)
    eroded_inv = erosion(1 - image, se_absent)
    return np.logical_and(eroded_img, eroded_inv).astype(np.uint8)


# def M4(A):
#     X_prev = A.copy().astype(np.uint8)
#     B1 = np.ones((3, 3), dtype=np.uint8)
#
#     while True:
#         internal_pixels = erosion(X_prev, B1)
#
#         X_new = np.logical_and(X_prev, np.logical_not(internal_pixels)).astype(np.uint8)
#
#         if np.array_equal(X_new, X_prev):
#             return X_new
#         X_prev = X_new


def M4(bin_img, template=None, thresh=0.9, allow_missing=0):
    A = (bin_img != 0).astype(np.uint8)
    if template is None:
        template = np.array([[1,0,0],
                             [1,0,0],
                             [1,1,1]], dtype=np.uint8)
    T = (template != 0).astype(np.uint8)
    t_sum = int(T.sum())
    if t_sum == 0:
        return np.zeros_like(A, dtype=np.uint8)

    # Try OpenCV
    try:
        import cv2
        # matchTemplate expects 8-bit single-channel image; result is float map of correlation
        res = cv2.matchTemplate(A.astype(np.uint8), T.astype(np.uint8), cv2.TM_CCORR_NORMED)
        # res shape is (H-tH+1, W-tW+1). threshold on res
        hits = (res >= thresh).astype(np.uint8)
        mask = np.zeros_like(A, dtype=np.uint8)
        th, tw = T.shape
        # place hits at top-left of each match; if you prefer center, adjust indices
        ys, xs = np.nonzero(hits)
        for y, x in zip(ys, xs):
            mask[y:y+th, x:x+tw] |= 1  # mark area of match (or set center only)
        return mask

    except Exception:
        pass

    # Try scipy.signal.convolve2d
    try:
        from scipy.signal import convolve2d
        conv = convolve2d(A, T[::-1, ::-1], mode='valid')  # sum of overlapping ones
        # allow_missing: conv >= t_sum - allow_missing
        if allow_missing <= 0:
            # use relative threshold if thresh in (0,1)
            if 0 < thresh <= 1:
                min_required = int(np.ceil(t_sum * thresh))
            else:
                min_required = t_sum
        else:
            min_required = max(0, t_sum - int(allow_missing))

        hits = (conv >= min_required)
        mask = np.zeros_like(A, dtype=np.uint8)
        th, tw = T.shape
        ys, xs = np.nonzero(hits)
        for y, x in zip(ys, xs):
            mask[y:y+th, x:x+tw] |= 1
        return mask

    except Exception:
        pass

    # Fallback: numpy sliding window (works but медленнее)
    # Create view of all patches using as_strided
    H, W = A.shape
    th, tw = T.shape
    if H < th or W < tw:
        return np.zeros_like(A, dtype=np.uint8)

    shape = (H - th + 1, W - tw + 1, th, tw)
    strides = (A.strides[0], A.strides[1], A.strides[0], A.strides[1])
    from numpy.lib.stride_tricks import as_strided
    patches = as_strided(A, shape=shape, strides=strides)
    # compute sum of patch * template for each patch
    # patches is view shape (ny, nx, th, tw)
    prod = (patches * T).sum(axis=(2,3))
    if allow_missing <= 0:
        if 0 < thresh <= 1:
            min_required = int(np.ceil(t_sum * thresh))
        else:
            min_required = t_sum
    else:
        min_required = max(0, t_sum - int(allow_missing))

    hits = (prod >= min_required)
    mask = np.zeros_like(A, dtype=np.uint8)
    ys, xs = np.nonzero(hits)
    for y, x in zip(ys, xs):
        mask[y:y+th, x:x+tw] |= 1
    return mask


SE_cross = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)

SE_square = np.ones((3, 3), dtype=np.uint8)

SE_line = np.array([
    [1, 1, 1]
], dtype=np.uint8)


def region_growing(image, seeds, threshold):
    img = image.astype(np.float32)
    h, w = img.shape
    threshold = float(threshold)

    labels = np.zeros_like(image, dtype=np.int32)
    current_label = 1

    region_mean = {}
    region_size = {}

    if seeds == "automatic":
        seeds = [(x, y) for x in range(h) for y in range(w)]
    elif seeds == "manual":
        seeds = [(0, 0), (5, 5)]
    else:
        seeds = [(x, y) for x, y in seeds if 0 <= x < h and 0 <= y < w]

    for seed_x, seed_y in seeds:

        if labels[seed_x, seed_y] != 0:
            continue

        queue = [(seed_x, seed_y)]
        labels[seed_x, seed_y] = current_label

        collected = [img[seed_x, seed_y]]

        while queue:
            x, y = queue.pop(0)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):

                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < h and 0 <= ny < w):
                        continue

                    if labels[nx, ny] != 0:
                        continue

                    if abs(img[nx, ny] - img[x, y]) <= threshold:
                        labels[nx, ny] = current_label
                        queue.append((nx, ny))
                        collected.append(img[nx, ny])

        region_mean[current_label] = float(np.mean(collected))
        region_size[current_label] = len(collected)
        current_label += 1

    borders = np.zeros_like(labels, dtype=bool)

    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        shifted = np.roll(labels, (dx, dy), axis=(0, 1))
        borders |= (labels != shifted)

    border_coords = np.argwhere(borders)

    adjacent = set()

    for x, y in border_coords:
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= nx < h and 0 <= ny < w:
                a = labels[x, y]
                b = labels[nx, ny]
                if a != 0 and b != 0 and a != b:
                    adjacent.add(tuple(sorted((a, b))))

    parent = {lab: lab for lab in np.unique(labels) if lab > 0}

    for a, b in adjacent:

        pa = parent[a]
        pb = parent[b]

        if pa == pb:
            continue

        similar = abs(region_mean[pa] - region_mean[pb]) <= threshold

        if similar:
            low = min(pa, pb)
            high = max(pa, pb)
            for k in parent:
                if parent[k] == high:
                    parent[k] = low

    for old, new in parent.items():
        while parent[new] != new:
            new = parent[new]
        labels[labels == old] = new

    return labels


def main():
    if len(sys.argv) == 1 or "--help" in sys.argv:
        print("""
Command-line image processing (manual pixel-by-pixel version)

Usage:
    python imgproc_manual.py --command -input=in.png -output=out.png [params]

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
    --cmean, --cvariance, --cstdev, --cvarcoi, --casyco, --cflatco, --cvarcoii, --centropy
    --hrayleigh     -alpha=50
    --slowpass
    --oll           -eps=1e-6

    Morphology (on internally binarized grayscale image, output 0/255 mask):
    --morph-dilate  -se=cross|square|line -thresh=128
    --morph-erode   -se=cross|square|line -thresh=128
    --morph-open    -se=cross|square|line -thresh=128
    --morph-close   -se=cross|square|line -thresh=128
    --morph-m4    -se=cross|square|line -thresh=128

    Region growing (manual segmentation on grayscale):
    --regiongrow    -sy=0 -sx=0 -threshold=10
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

    elif cmd == "--histogram":
        ch = int(args.get("channel", 0))
        out_file = args.get("output", "histogram.png")
        save_histogram(img, ch, out_file)
        return

    elif cmd == "--cmean":
        print("Mean:", cmean(img))
        return

    elif cmd == "--cvariance":
        print("Variance:", cvariance(img))
        return

    elif cmd == "--cstdev":
        print("StdDev:", cstdev(img))
        return

    elif cmd == "--cvarcoi":
        print("VarCoeff I:", cvarcoi(img))
        return

    elif cmd == "--casyco":
        print("Asymmetry:", casyco(img))
        return

    elif cmd == "--cflatco":
        print("Flattening:", cflatco(img))
        return

    elif cmd == "--cvarcoii":
        print("VarCoeff II:", cvarcoii(img))
        return

    elif cmd == "--centropy":
        print("Entropy:", centropy(img))
        return

    elif cmd == "--hrayleigh":

        alpha = float(args.get("alpha", 10))
        result = hrayleigh(img, alpha)

    elif cmd == "--slowpass":
        result = optimized_slowpass(img)

    elif cmd == "--oll":
        eps = float(args.get("eps", 1e-6))
        result = oll(img, eps)

    elif cmd in ("--morph-dilate", "--morph-erode", "--morph-open", "--morph-close", "--morph-m4"):
        se_name = args.get("se", "cross")
        thresh = int(args.get("thresh", 128))

        gray = to_grayscale(img)
        bin_img = binarize(gray, thresh)

        if se_name == "square":
            B = SE_square
        elif se_name == "line":
            B = SE_line
        else:
            B = SE_cross

        if cmd == "--morph-dilate":
            mask = dilation(bin_img, B)
        elif cmd == "--morph-erode":
            mask = erosion(bin_img, B)
        elif cmd == "--morph-open":
            mask = opening(bin_img, B)
        elif cmd == "--morph-close":
            mask = closing(bin_img, B)
        else:
            template_L = np.array([[1, 1, 0],
                                   [0, 1, 0],
                                   [0, 1, 0]], dtype=np.uint8)

            mask = M4(bin_img, template=template_L, thresh=0.9, allow_missing=1)

        result = mask_to_color(mask)

    elif cmd == "--regiongrow":
        sy = int(args.get("sy", 0))
        sx = int(args.get("sx", 0))
        threshold = int(args.get("threshold", 10))

        gray = to_grayscale(img)
        seg_mask = region_growing(gray, [(sy, sx)], threshold)
        result = mask_to_color(seg_mask)

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