from skimage import measure
import skimage

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.filters import threshold_otsu, threshold_minimum, threshold_mean, threshold_yen, threshold_local
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, area_closing
from skimage.color import label2rgb

def generate_attention_coordinates(attention_map, thresh_method='local', block_size=21, method='gaussian',
                                   min_area=32*32, max_area=160*160, num_bboxes=1, random_crop_sz=112):
    "Generate coordinates from a single attention heatmap (sz,sz)"
    if   thresh_method == 'local':  thresh = threshold_local(attention_map, block_size=block_size, method=method)
    elif thresh_method == 'mean':   thresh = threshold_mean(attention_map)
    else: raise Exception("thresh_method unknown")

    # calculate attention cooords
    coordinates = []
    for region in regionprops(label(closing(attention_map > thresh))):
        if (region.area >= min_area) and (region.area <= max_area):
            minr, minc, maxr, maxc = region.bbox
            vals = attention_map[minr:maxr, minc:maxc]
            conf = vals.max()
            coordinates.append ((conf, [minr, minc, maxr, maxc]))

    # sort by global maxpool
    top_coordinates = sorted(coordinates, key=lambda o: o[0], reverse=True)[:num_bboxes]
    coordinates     = [o[1] for o in top_coordinates]

    # fill random bbox coordinates for missing
    sz = attention_map.shape[0]
    bbox_diff = int(num_bboxes - len(coordinates))
    if bbox_diff > 0:
        for _ in range(bbox_diff):
            low = random_crop_sz//2
            high = sz-low
            cr = np.random.choice(range(low,high))
            cc = np.random.choice(range(low,high))
            coordinates.append([cr-low, cc-low, cr+low,cc+low])
    return coordinates     