"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Anchors
"""

# inspired by 
# https://github.com/matterport/Mask_RCNN

import numpy as np
import utils


def anchors_from_fmap(scales, ratios, fmap_size, fmap_stride):
    """
    Vectorized implementation for generating anchors from a feature map.

    Parameters
    ----------
    scales : list
        The scales of anchors s.t. scale**2 = height * width where 
        (height, width) = fmap_size, for all scale in scales.
    ratios : list
        The height/width ratio.
    fmap_size : tuple
        The size of the feature map.
    fmap_stride : integer
        The stride on the image to obtain the feature map.

    Returns
    -------
    anchors : numpy array,  [num_anchors, 4]
        Anchors, where 4 is the corner coordinates (y1, x1, y2, x2).

    """
    # all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    
    # all combinations of anchors' height and width
    heights = scales * np.sqrt(ratios)
    widths = scales / np.sqrt(ratios)
    
    # all combinations of anchors' center
    y_shifts = np.arange(fmap_size[0]) * fmap_stride
    x_shifts = np.arange(fmap_size[1]) * fmap_stride
    x_shifts, y_shifts = np.meshgrid(x_shifts, y_shifts)
    
    # all combinations of x_shifts and widths, and y_shifts and heights
    anchor_widths, anchor_x_centers = np.meshgrid(widths, x_shifts)
    anchor_heights, anchor_y_centers = np.meshgrid(heights, y_shifts)
    
    # anchors' center and size, shape of 
    # [product of fmap_size * number of anchors per pixel, 2]
    anchor_centers = np.stack([anchor_y_centers, anchor_x_centers], axis=2)
    anchor_sizes = np.stack([anchor_heights, anchor_widths], axis=2)
    anchor_centers = anchor_centers.reshape((-1,2))
    anchor_sizes = anchor_sizes.reshape((-1,2))
    
    # convert center coordinates to corner coordinates (y1, x1, y2, x2)
    anchors = np.concatenate([
        anchor_centers - 0.5 * anchor_sizes, 
        anchor_centers + 0.5 * anchor_sizes
        ], axis=1)
    
    return anchors


def anchors_from_fpn(scales, ratios, fmap_sizes, fmap_strides, denser=False):
    """
    Generates anchors from FPN.

    Parameters
    ----------
    scales : list
        The scales of anchors s.t. scale**2 = height * width where 
        (height, width) = fmap_size for all scale in scales. Note that each 
        scale is corresponding to a feature map of FPN, i.e., 
        len(scales) = len(fmap_sizes).
    ratios : list
        The height/width ratio.
    fmap_sizes : tuple
        The sizes of the feature maps.
    fmap_strides : integer
        The strides of the feature maps.
    denser : boolean, optional
        Whether to use denser scales. The default is False. If used, scale for
        each feature map is extended to 
        [scale*2**0, scale*2**(1/3), scale*2**(2/3)].

    Returns
    -------
    anchors : list
        Set of anchors, each of shape [num_anchors_i, 4] where 4 is the corner 
        coordinates (y1, x1, y2, x2) and num_anchors_i is number of anchors at 
        each fmap.

    """
    assert len(fmap_sizes) == len(scales), \
        'Number of FPN outputs must be equal to len(scales).'
        
    anchors = []
    for i in range(len(scales)):
        scales_fmap = scales[i]
        if denser:
            scales_fmap *= np.array([2**0, 2**(1/3), 2**(2/3)])
        # print(i, scales_fmap)
        anchors_i = anchors_from_fmap(
            scales_fmap, ratios, fmap_sizes[i], fmap_strides[i])
        anchors.append(anchors_i)
    
    # anchors = np.concatenate(anchors, axis=0)
    return anchors
    

def anchors_targets(anchors, boxes, box_class_ids, num_object_classes):
    """
    Assigns anchors to classification and regression targets.

    Parameters
    ----------
    anchors : numpy array, [num_anchors, 4]
        Anchors, where 4 is the corner coordinates (y1, x1, y2, x2).
    boxes : numpy array, [num_boxes, 4]
        Ground-truth object boxes, where 4 is the corner coordinates.
    box_class_ids : numpy array, [num_boxes, ]
        Classes of boxes which are > 0, i.e., doesn't include background class 
        denoted by 0.
    num_object_classes : integer
        The number of ground-truth object classes, i.e., each class is positive 
        integer.

    Returns
    -------
    anchor_indicators : numpy array, [num_anchors,]
        An array indicates negative -1, neutral 0 and positive 1 anchors.
    anchor_class_ids : numpy array, [num_anchors, num_object_classes]
        Classification targets. Note that rows with all zeros indicates 
        negative anchors, i.e., background.
    anchor_offsets : numpy array, [num_anchors, 4]
        Regression targets, where 4 is the center coordinates (y, x, h, w). 
        Note that the indices of anchor offsets containing object match the 
        indices of anchor class ids.

    """
    num_anchors = anchors.shape[0]
    anchor_indicators = np.zeros((num_anchors,), np.int32)
    anchor_class_ids = np.zeros((num_anchors, num_object_classes), np.int32)
    anchor_offsets = np.zeros((num_anchors, 4), np.float32)
    
    # compute iou between anchors and boxes, [num_anchors, num_boxes]
    ious = utils.compute_ious(anchors, boxes)
    
    # find corresponding box index and iou score for every anchor
    idxes = np.argmax(ious, axis=1)
    iou_scores = ious[np.arange(num_anchors), idxes]
    
    # set anchor_indicators to -1 if iou_scores < 0.4
    anchor_indicators[iou_scores < 0.4] = -1
    
    # dealing with the case when all anchors are assigned to negative
    pos_idxes = np.where(ious == np.max(ious, axis=0))[0]
    anchor_indicators[pos_idxes] = 1
    
    # set anchor_indicators to 1 if iou_scores >= 0.5
    anchor_indicators[iou_scores >= 0.5] = 1
    
    # positive anchors's indices
    ids = np.where(anchor_indicators == 1)[0]
    
    # anchor_class_ids
    pos_anchor_class_ids = box_class_ids[idxes[ids]] # [num_positive_anchors,]
    anchor_class_ids[ids, pos_anchor_class_ids - 1] = 1
    
    # anchor_offsets
    anchor_offsets[ids] = utils.compute_offsets(
        anchors[ids], boxes[idxes[ids]])
    
    return anchor_indicators, anchor_class_ids, anchor_offsets