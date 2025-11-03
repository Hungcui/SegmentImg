import numpy as np
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.measure import label, regionprops

try:
    import pydensecrf.densecrf as densecrf
    CRF_AVAILABLE = True
except Exception:
    CRF_AVAILABLE = False

class PostProcessor:
    def __init__(self, use_morphology=True, morphology_size=3, min_blob_size=100, use_crf=False, crf_iters=5):
        self.use_morphology = use_morphology
        self.morphology_size = morphology_size
        self.min_blob_size = min_blob_size
        self.use_crf = use_crf and CRF_AVAILABLE
        self.crf_iters = crf_iters

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        if not self.use_morphology: return mask
        selem = disk(self.morphology_size)
        mask = binary_opening(mask > 0, selem).astype(mask.dtype)
        mask = binary_closing(mask > 0, selem).astype(mask.dtype)
        return mask

    def filter_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        if self.min_blob_size <= 0: return mask
        labeled = label(mask > 0)
        out = np.zeros_like(mask)
        for prop in regionprops(labeled):
            if prop.area >= self.min_blob_size:
                out[labeled == prop.label] = mask[labeled == prop.label]
        return out

    def apply_crf(self, image: np.ndarray, mask_probs: np.ndarray) -> np.ndarray:
        if not self.use_crf: return np.argmax(mask_probs, axis=-1)
        H, W = image.shape[:2]; C = mask_probs.shape[-1]
        unary = -np.log(mask_probs + 1e-8).reshape((C, -1))
        d = densecrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
        Q = d.inference(self.crf_iters)
        return np.argmax(Q, axis=0).reshape((H, W))

    def process(self, image: np.ndarray, mask: np.ndarray, mask_probs: np.ndarray = None) -> np.ndarray:
        result = self.apply_morphology(mask.copy())
        result = self.filter_small_blobs(result)
        if self.use_crf and mask_probs is not None:
            result = self.apply_crf(image, mask_probs)
        return result
