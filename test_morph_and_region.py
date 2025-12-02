# test_morph_and_region.py
import sys
import pathlib
import importlib.util
import numpy as np
import collections

# --- Динамически загружаем main.py как модуль ip ---
here = pathlib.Path(__file__).parent.resolve()
main_path = here / "main.py"
spec = importlib.util.spec_from_file_location("ip_module", str(main_path))
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)

# ---- вспомогательные функции ----
def to_bin(a):
    return (np.array(a, dtype=np.uint8) != 0).astype(np.uint8)

def assert_mask_equal(a, b):
    a = to_bin(a)
    b = to_bin(b)
    assert a.shape == b.shape, f"Shapes differ: {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), f"Masks differ\nA:\n{a}\nB:\n{b}"

def neighbors4(y, x, h, w):
    for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx

def is_connected(mask, p1, p2):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    q = collections.deque()
    sy, sx = p1
    ty, tx = p2
    if mask[sy, sx] == 0 or mask[ty, tx] == 0:
        return False
    q.append((sy, sx))
    visited[sy, sx] = True
    while q:
        y, x = q.popleft()
        if (y, x) == (ty, tx):
            return True
        for ny, nx in neighbors4(y, x, h, w):
            if not visited[ny, nx] and mask[ny, nx] == 1:
                visited[ny, nx] = True
                q.append((ny, nx))
    return False

# ---------------------------
# 1) Dilation / Erosion
# ---------------------------
def test_dilation_with_cross_se():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 1
    out = ip.dilation(img, ip.SE_cross)
    expected = to_bin([
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
    ])
    assert_mask_equal(out, expected)

def test_erosion_with_square_se():
    img = to_bin([
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ])
    out = ip.erosion(img, ip.SE_square)
    expected = to_bin([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    assert_mask_equal(out, expected)

# ---------------------------
# 2) Opening / Closing
# ---------------------------
def test_opening_removes_small_noise():
    img = to_bin([
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
    ])
    out = ip.opening(img, ip.SE_square)
    assert out[4,3] == 0
    assert out[2,2] == 1

def test_closing_fills_small_hole():
    img = to_bin([
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ])
    out = ip.closing(img, ip.SE_square)
    assert out[2,2] == 1

# ---------------------------
# 3) Hit-or-Miss (HMT)
# ---------------------------
def test_hit_or_miss_detects_corner():
    img = to_bin([
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    se_fore = to_bin([
        [0,0,0],
        [0,1,1],
        [0,1,0],
    ])
    se_back = to_bin([
        [1,1,1],
        [1,0,0],
        [1,0,1],
    ])
    out = ip.hit_or_miss(img, se_fore, se_back)
    assert out.shape == img.shape
    assert out[1,1] == 1
    assert np.sum(out) == 1

def test_hit_or_miss_no_false_positive_on_similar():
    img = to_bin([
        [0,0,0,0],
        [0,1,1,0],
        [0,0,1,0],
        [0,0,0,0],
    ])
    se_fore = to_bin([
        [0,0,0],
        [0,1,1],
        [0,1,0],
    ])
    se_back = to_bin([
        [1,1,1],
        [1,0,0],
        [1,0,1],
    ])
    out = ip.hit_or_miss(img, se_fore, se_back)
    assert np.sum(out) == 0

# ---------------------------
# 4) Region growing + merging
# ---------------------------
def test_region_growing_single_seed_grows_whole_object():
    gray = np.zeros((7, 7), dtype=np.uint8)
    gray[1:6, 1:6] = 100
    # Сделали центр одинаковым с окружением (100), чтобы при threshold=10 рост происходил
    gray[3, 3] = 100
    mask = ip.region_growing(gray, [(3, 3)], threshold=10)
    assert np.all(mask[1:6, 1:6] == 1)
    assert mask[0, 0] == 0

def test_region_growing_two_seeds_no_merge_low_threshold():
    h, w = 15, 41
    gray = np.zeros((h, w), dtype=np.uint8)
    gray[5:10, 5:10] = 50
    gray[5:10, 31:36] = 50
    gray[7:8, 19:30] = 80
    seed_left = (6, 6)
    seed_right = (6, 33)
    mask_multi_low = ip.region_growing(gray, [seed_left, seed_right], threshold=10)
    mask_left = ip.region_growing(gray, [seed_left], threshold=10)
    mask_right = ip.region_growing(gray, [seed_right], threshold=10)
    union_lr = ((mask_left != 0) | (mask_right != 0)).astype(np.uint8)
    assert_mask_equal(mask_multi_low, union_lr)
    assert not is_connected(mask_multi_low, seed_left, seed_right)

def test_region_growing_two_seeds_merge_high_threshold():
    h, w = 15, 41
    gray = np.zeros((h, w), dtype=np.uint8)
    gray[5:10, 5:10] = 50
    gray[5:10, 31:36] = 50
    gray[7:8, 19:30] = 80
    seed_left = (6, 6)
    seed_right = (6, 33)
    # Увеличили порог до 60, чтобы гарантировать переход 50 -> 80 -> 50 в этой конфигурации
    mask_multi_high = ip.region_growing(gray, [seed_left, seed_right], threshold=60)
    assert is_connected(mask_multi_high, seed_left, seed_right)
    assert mask_multi_high[seed_left] == 1
    assert mask_multi_high[seed_right] == 1
    m_left = ip.region_growing(gray, [seed_left], threshold=60)
    m_right = ip.region_growing(gray, [seed_right], threshold=60)
    union_lr = ((m_left != 0) | (m_right != 0)).astype(np.uint8)
    assert_mask_equal(mask_multi_high, union_lr)

# ---------------------------
# 5) Edge cases
# ---------------------------
def test_region_growing_seed_out_of_bounds_ignored():
    gray = np.full((5,5), 100, dtype=np.uint8)
    mask = ip.region_growing(gray, [(-1,-1), (2,2), (10,10)], threshold=5)
    mask_single = ip.region_growing(gray, [(2,2)], threshold=5)
    assert_mask_equal(mask, mask_single)

def test_morphology_respects_binary_input_0_1():
    img_0255 = np.zeros((5,5), dtype=np.uint8)
    img_0255[2,2] = 255
    bin01 = (img_0255 // 255).astype(np.uint8)
    out = ip.dilation(bin01, ip.SE_cross)
    assert out[2,2] == 1
    assert out[1,2] == 1
    assert out[3,2] == 1
