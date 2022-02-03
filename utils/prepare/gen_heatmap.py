import math
from typing import Any, Callable, List, Tuple, Union
import imageio
import skimage
import skimage.color
import cv2 as cv
import numpy as np

class GenImgHeatmap:
    def __init__(self, radius, kernel_func=None):
        # type: (int, Callable[[float], float]) -> None
        self.radius = radius

        if kernel_func is None:
            kernel_func = lambda _: 1

        mask = np.zeros((2 * radius + 1, 2 * radius + 1))
        for u in range(0, mask.shape[0]):
            for v in range(0, mask.shape[1]):
                r = math.sqrt((u-radius)**2+(v-radius)**2)
                if r < radius:
                    mask[u, v] += kernel_func(r)

        self.mask = mask

        pass

    def __call__(self, height, width, keypoints):
        # type: (int, int, List[Tuple[int, int]]) -> np.ndarray
        '''
        generate a heatmap of a single image
        '''
        heatmap = np.zeros((height + 2 * self.radius, width + 2 * self.radius))
        for kp_u, kp_v in keypoints:
            heatmap[kp_u:kp_u+2*self.radius+1, kp_v:kp_v+2*self.radius+1] += self.mask

        heatmap = heatmap[self.radius:self.radius+height, self.radius:self.radius+width]
        heatmap_max = heatmap.max()
        heatmap_min = heatmap.min()
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        
        return np.ascontiguousarray(heatmap)