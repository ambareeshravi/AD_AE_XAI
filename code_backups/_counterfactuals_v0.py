import numpy as np

from skimage.metrics import mean_squared_error
from PIL import Image
from skimage.segmentation import mark_boundaries

def get_segments(img, height, width, imgheight, imgwidth):
    segments = []
    for i in range(0,imgheight, height):
        for j in range(0,imgwidth, width):
            box = (j, i, j+width, i+height)
            segments.append(box)
    return segments

def calculate_error(_img1, _img2):
    return mean_squared_error(_img1, _img2)

def get_reconstruction(_img, model):
    _reshaped = _img.reshape((1, 128, 128, 3))
    return model.predict(_reshaped)[0]

def minimum_edits(_img1, _img2, _threshold, _filter_size, model):
    img_height = np.array(_img1).shape[0]
    img_width = np.array(_img1).shape[1]
    segs1 = get_segments(_img1, _filter_size, _filter_size, img_height, img_width)
    segs2 = get_segments(_img2, _filter_size, _filter_size, img_height, img_width)
    edits = []

    current_img = np.array(_img1, copy=True)
    cur_iter = 0
    while True:
        if (cur_iter > len(segs1)):
            break
        max_error = 0
        seg_idx = 0
        for idx, seg in enumerate(segs2):
            edited_img = np.array(current_img, copy=True)
            edited_img[seg[0]: seg[2], seg[1]: seg[3], :] = _img2[seg[0]: seg[2], seg[1]: seg[3], :]
            edited_reconstruction = get_reconstruction(edited_img, model)
            edited_error = calculate_error(edited_img, edited_reconstruction)
            if edited_error > max_error:
                max_error = edited_error
                seg_idx = idx
        best_seg = segs2.pop(seg_idx)
        current_img[best_seg[0]: best_seg[2], best_seg[1]: best_seg[3], :] = _img2[best_seg[0]: best_seg[2], best_seg[1]: best_seg[3], :]
        edits.append(best_seg)
        if (max_error > _threshold):
            return edits
        cur_iter += 1
    return edits

def draw_boundaries(_edits, _img, _path):
    masks = np.zeros((128, 128)).astype(int)
    for _edit in _edits:
        masks[_edit[0]: _edit[2], _edit[1]: _edit[3]] = 1
    _final_img = mark_boundaries(_img / 2 + 0.5, masks)
    _final_img = (_final_img * 255).astype(np.uint8)
    _im = Image.fromarray(_final_img)
    _im.save(_path)


class Counterfactual:
    def __init__(self, model):
        self.model = model

    def explain(self, output_path, norms, anomalies, threshold_pct, block_size, anomaly_type):
        norm_idx = 0
        for norm in norms:
            anomaly_index = 0
            for anomaly in anomalies:
                normal_reconstructed = get_reconstruction(norm, self.model)
                normal_err = calculate_error(norm, normal_reconstructed)

                anomaly_reconstructed = get_reconstruction(anomaly, self.model)
                anomaly_err = calculate_error(anomaly, anomaly_reconstructed)

                if normal_err < anomaly_err:
                    threshold = anomaly_err * threshold_pct
                    edits_made = minimum_edits(norm, anomaly, threshold, block_size, self.model)
                    img_path = '%snormal%s-%s%s.png' % (output_path, str(norm_idx), anomaly_type, str(anomaly_index))
                    draw_boundaries(edits_made, anomaly, img_path)
                anomaly_index += 1
            norm_idx += 1