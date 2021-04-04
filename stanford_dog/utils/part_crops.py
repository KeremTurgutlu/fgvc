from fastai.vision.all import *

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def fix_bbox_sz(coords, targ_bbox_sz, targ_img_sz):
    """
    coords: coordinates in (minr, minc, maxr, maxc)
    targ_bbox_sz: 1 size in px of a square bbox
    targ_img_sz: size in px of the cropped image
    """
    minr, minc, maxr, maxc = coords
    rdiff, cdiff = targ_bbox_sz - (maxr - minr), targ_bbox_sz - (maxc - minc)
    if rdiff != 0:
        if maxr + rdiff <= targ_img_sz:
            maxr += rdiff
        elif minr - rdiff >= 0:
            minr -= rdiff
        else:
            raise Exception(f"Can't fix bbox row size to match {targ_img_sz}")

    if cdiff != 0:
        if maxc + cdiff <= targ_img_sz:
            maxc += cdiff    
        elif minc - cdiff >= 0:
            minc -= cdiff
        else:
            raise Exception(f"Can't fix bbox column size to match {targ_img_sz}")    
    
    return minr, minc, maxr, maxc


def _generate_batch_crops(sorted_scores, sorted_idxs, crop_scores, grid_img_pct, kernel_size, targ_sz, targ_bbox_sz,num_bboxes, nms_thresh):
    batch_final_resized_coords = []

    for scores, idxs in zip(sorted_scores, sorted_idxs):

        top_coords = []
        for score,idx in zip(scores, idxs):
            div, mod = divmod(idx, crop_scores.size(-1))
            minr, minc, maxr, maxc = div, mod, div+kernel_size, mod+kernel_size
            top_coords += [[minr, minc, maxr, maxc, score]]

        top_coords = np.vstack(top_coords)
        top_nms_idxs = nms(top_coords, thresh=nms_thresh)

        final_resized_coords = (top_coords[top_nms_idxs][:, :-1]*grid_img_pct*targ_sz)[:num_bboxes].astype(int)
        final_resized_coords = array([fix_bbox_sz(o, targ_bbox_sz, targ_sz) for o in final_resized_coords])
        batch_final_resized_coords.append(final_resized_coords)

    stacked_bboxes = np.vstack(batch_final_resized_coords)
    assert np.all((stacked_bboxes[:, 2] - stacked_bboxes[:, 0]) == targ_bbox_sz)
    assert np.all((stacked_bboxes[:, 3] - stacked_bboxes[:, 1]) == targ_bbox_sz)
    
    return batch_final_resized_coords


def generate_batch_crops(attention_maps, source_sz=384, targ_sz=448, targ_bbox_sz=112, num_bboxes=2, nms_thresh=0.1):
    """ 
    attention_maps: a batch of attention maps with bs x grid_sz x grid_sz (torch.tensor - detached)
    source_sz : source image size in px that attention maps were generated
    targ_sz : target image size in px that crops will be taken
    targ_bbox_pct: desired percentage of area a crop will cover in target image
    num_bboxes: number of bboxes to return per sample
    """
    
    # number of grids on side of a square attention map
    grid_size = attention_maps.size(-1) 
    grid_img_pct = 1/grid_size
    targ_bbox_pct = targ_bbox_sz/targ_sz
    
    # calculate scores for all crop candidates
    kernel_size = int(targ_bbox_pct*grid_size)
    AvgPoolLayer = nn.AvgPool2d(kernel_size=kernel_size, stride=1)
    crop_scores = AvgPoolLayer(attention_maps) 
    crop_scores_flat = crop_scores.view(crop_scores.size(0),-1)

    # sort by score
    crop_scores_idxs = torch.sort(crop_scores_flat, descending=True)
    sorted_scores, sorted_idxs = crop_scores_idxs.values.numpy(), crop_scores_idxs.indices.numpy()
    
    return _generate_batch_crops(sorted_scores, sorted_idxs, crop_scores, grid_img_pct, 
                                 kernel_size, targ_sz, targ_bbox_sz, num_bboxes, nms_thresh)