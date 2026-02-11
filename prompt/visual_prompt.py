import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


COLOR_MAP = {
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
}


def extract_contour(mask_slice, thickness=3):
    if mask_slice.sum() == 0:
        return np.zeros_like(mask_slice, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = ndimage.binary_erosion(mask_slice, kernel)
    contour = mask_slice.astype(bool) & ~eroded
    if thickness > 1:
        contour = ndimage.binary_dilation(contour, kernel, iterations=thickness - 1)
    return contour


def keep_largest_component(mask_slice):
    labeled, num_features = ndimage.label(mask_slice)
    if num_features <= 1:
        return mask_slice
    component_sizes = ndimage.sum(mask_slice, labeled, range(1, num_features + 1))
    largest = np.argmax(component_sizes) + 1
    return (labeled == largest).astype(mask_slice.dtype)


def overlay_contour_on_slice(ct_slice, contour, color=(0, 0, 255), alpha=1.0):
    if ct_slice.ndim == 2:
        ct_min, ct_max = ct_slice.min(), ct_slice.max()
        if ct_max > ct_min:
            normalized = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(ct_slice, dtype=np.uint8)
        rgb = np.stack([normalized] * 3, axis=-1)
    else:
        rgb = ct_slice.copy()
    if contour.any():
        for c in range(3):
            rgb[contour, c] = int(alpha * color[c] + (1 - alpha) * rgb[contour, c])
    return rgb


def generate_visual_prompt_volume(ct_volume, mask_volume, color="blue", thickness=3):
    color_rgb = COLOR_MAP.get(color, (0, 0, 255))
    prompted_volume = np.zeros((*ct_volume.shape, 3), dtype=np.uint8)
    num_slices = ct_volume.shape[0]
    for i in range(num_slices):
        ct_slice = ct_volume[i]
        mask_slice = mask_volume[i] if mask_volume is not None else None
        ct_min, ct_max = ct_slice.min(), ct_slice.max()
        if ct_max > ct_min:
            normalized = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(ct_slice, dtype=np.uint8)
        rgb = np.stack([normalized] * 3, axis=-1)
        if mask_slice is not None and mask_slice.sum() > 0:
            cleaned = keep_largest_component(mask_slice)
            contour = extract_contour(cleaned, thickness)
            for c in range(3):
                rgb[contour, c] = color_rgb[c]
        prompted_volume[i] = rgb
    return prompted_volume


def generate_prompted_nifti(ct_volume, mask_volume, color="blue", thickness=3):
    color_rgb = COLOR_MAP.get(color, (0, 0, 255))
    output = ct_volume.copy().astype(np.float32)
    num_slices = ct_volume.shape[0]
    has_prompt = np.zeros(num_slices, dtype=bool)
    for i in range(num_slices):
        mask_slice = mask_volume[i]
        if mask_slice.sum() > 0:
            cleaned = keep_largest_component(mask_slice)
            contour = extract_contour(cleaned, thickness)
            output[i][contour] = output[i].max() + 100
            has_prompt[i] = True
    return output, has_prompt


def compute_contour_boundary(mask_2d):
    H, W = mask_2d.shape
    contour = np.zeros_like(mask_2d, dtype=bool)
    for y in range(H):
        for x in range(W):
            if mask_2d[y, x] == 0:
                continue
            is_boundary = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W:
                        is_boundary = True
                        break
                    if abs(int(mask_2d[ny, nx]) - int(mask_2d[y, x])) > 0:
                        is_boundary = True
                        break
                if is_boundary:
                    break
            contour[y, x] = is_boundary
    return contour
