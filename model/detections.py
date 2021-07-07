"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Detections
"""

import tensorflow as tf

import utils


def select_top_scoring(
        anchors,
        probs, 
        offsets,
        confidence_threshold=0.05, 
        num_top_scoring=1000,
        window=[0,0,512,512],
        batch_size=2,
        offsets_mean=None,
        offsets_std=None
        ):
    """
    Selects top-scoring refined anchors.

    Parameters
    ----------
    anchors : tf tensor, [batch_size, num_anchors, 4]
        Anchors.  
    probs : tf tensor, [batch_size, num_anchors, num_object_classes]
        Anchors' probabilities to contain object.        
    offsets : tf tensor, [batch_size, num_anchors, 4]
        Anchors' offsets from gt boxes.
    confidence_threshold : float, optional    
        The minimum selection's probabilites. The default is 0.05.
    num_top_scoring : integer, optional
        The number of top-scoring selections. The default is 1000.
    window : list, optional
        The corner coordinates (y1, x1, y2, x2) used when clipping refined 
        anchors (after applied offsets). The default is [0,0,512,512].
    batch_size : integer, optional
        The batch size of anchors, probs or offsets. The default is 2.
    offsets_mean, offsets_std : float
        The mean and std of anchor offsets for a given dataset. If offsets are 
        normalized, they will be used to de-normalize offsets.

    Returns
    -------
    anchor_idxes : tf tensor, [(batch_size * num_anchors)_filtered, 2]
        Selected top-scoring indices where 
        (batch_size * num_anchors)_filtered <= batch_size * num_anchors and
        2 is (batch_idx, anchor_idx).
    anchors : tf tensor, [(batch_size * num_anchors)_filtered, 4]
        Flitered anchors.
    class_ids : tf tensor, [(batch_size * num_anchors)_filtered, ]
        Filtered anchors' class ids.
    scores : tf tensor, [(batch_size * num_anchors)_filtered, ]
        Filtered anchors' scores.

    """
    num_anchors = tf.shape(probs)[1]
    num_classes = tf.shape(probs)[2]
    
    # class_ids, [batch_size, num_anchors]
    class_ids = tf.argmax(probs, axis=-1, output_type=tf.int32)
    # reshape class_ids, [batch_size * num_anchors, ]
    class_ids_reshaped = tf.reshape(class_ids, (batch_size * num_anchors,))
    
    # reshape probs, [batch_size * num_anchors, num_classes]
    probs_reshaped = tf.reshape(probs, (-1, num_classes))
    # scores, [batch_size * num_anchors,]
    scores = tf.gather_nd(
        probs_reshaped,
        tf.stack([tf.range(batch_size*num_anchors), class_ids_reshaped], axis=1)
        )
    # reshape scores, [batch_size, num_anchors]
    scores = tf.reshape(scores, (batch_size, num_anchors))
    
    # filter low confidence, [(batch_size*num_anchors)_filtered1, 2] where 2 is 
    # (batch_idx, anchor_idx)
    threshold_idxes = tf.where(scores > confidence_threshold)
    
    # select top-scoring indices, [(batch_size*num_anchors)_filtered2, 2] where 
    # 2 is (batch_idx, anchor_idx)
    anchor_idxes = []
    for b in range(batch_size):
        batch_idxes = tf.where(tf.gather(threshold_idxes, 0, axis=1) == b)[:,0]
        num_anchors_per_img = tf.shape(batch_idxes)[0]
        k = tf.minimum(num_anchors_per_img, num_top_scoring)
        top_idxes = tf.math.top_k(tf.gather(scores, b), k)[1]
        anchor_idxes_per_img = tf.stack([tf.repeat(b, k), top_idxes], axis=1)
        anchor_idxes.append(anchor_idxes_per_img)
    anchor_idxes = tf.concat(anchor_idxes, axis=0)
    
    # filter class_ids & scores, [(batch_size*num_anchors)_filtered2, ]
    class_ids = tf.gather_nd(class_ids, anchor_idxes)
    scores = tf.gather_nd(scores, anchor_idxes)
    
    # select top-scoring anchors and then,
    # refine by applying offsets and clipping,
    # resulting in shape [(batch_size*num_anchors)_filtered2, 4]
    anchors = tf.gather_nd(anchors, anchor_idxes)
    offsets = tf.gather_nd(offsets, anchor_idxes)
    # de-normalize offsets if needed
    if offsets_mean is not None and offsets_std is not None:
        offsets = tf.add(tf.multiply(offsets, offsets_std), offsets_mean)
    anchors = utils.apply_offsets(anchors, offsets)
    anchors = utils.clip_boxes(anchors, window)
    
    return anchor_idxes, anchors, class_ids, scores


class SelectTopScoring(tf.keras.layers.Layer):
    """
    Defines a selecting top-scoring layer as a subclass of TF layer.
    
    Parameters
    ----------
    inputs : list
        Includes anchors, probs, offsets and window, where the first three are
        the same as in select_top_scoring(), but 
        * window : tf tensor, [1, 4]
            used when clipping refined anchors (after applied offsets), where
            1 is the batch_idx assuming that all images in the batch share the 
            same 4 corner coordinates (y1, x1, y2, x2).
    
    """
    def __init__(
            self, 
            confidence_threshold=0.05, 
            num_top_scoring=1000,
            batch_size=2,
            offsets_mean=None,
            offsets_std=None,
            **kwarg
            ):
        super(SelectTopScoring, self).__init__(**kwarg)
        self.confidence_threshold = confidence_threshold
        self.num_top_scoring = num_top_scoring
        self.batch_size = batch_size
        self.offsets_mean = offsets_mean
        self.offsets_std = offsets_std
        
    def call(self, inputs):
        anchors, probs, offsets = inputs[0], inputs[1], inputs[2]
        # window, [1, 4]
        window = inputs[3]
        return select_top_scoring(
            anchors,
            probs, 
            offsets,
            self.confidence_threshold, 
            self.num_top_scoring,
            window,
            self.batch_size,
            self.offsets_mean,
            self.offsets_std)


def nms_fpn(
        list_anchor_idxes, 
        list_anchors, 
        list_class_ids, 
        list_scores,
        max_objects_per_class_per_img=100,
        iou_threshold=0.5,
        batch_size=2
        ):
    """
    Applies non-maximum suppression (NMS) to all FPN levels.

    Parameters
    ----------
    list_anchor_idxes : list
        Set of anchors' indices at each FPN level, each is 
        [batch_size * num_anchors_fmap, 2] where 2 is (batch_idx, anchor_idx).
    list_anchors : list
        Set of anchors at each FPN level, each is 
        [batch_size * num_anchors_fmap, 4].
    list_class_ids : list
        Set of anchors' class ids at each FPN level, each is 
        [batch_size * num_anchors_fmap, ].
    list_scores : list
        Set of anchors' scores at each FPN level, each is 
        [batch_size * num_anchors_fmap, ].
    max_objects_per_class_per_img : integer, optional
        The maximum number of objects over all images for a particular class. 
        The default is 100.
    iou_threshold : float, optional
        An iou threshold for NMS. The default is 0.5.
    batch_size : integer, optional
        The batch size of each FPN level's anchor indices, anchors, class ids
        or scores. The default is 2.

    Returns
    -------
    anchors_batch : list
        Set of anchors after NMS for each image, each has shape 
        [num_anchors_per_img_filtered * num_fmaps, 4].
    class_ids_batch : list
        Set of corresponding class ids after NMS for each image, each has shape
        [num_anchors_per_img_filtered * num_fmaps, ]
    scores_batch : list
        Set of corresponding scores after NMS for each image, each has shape
        [num_anchors_per_img_filtered * num_fmaps, ].

    """
    # merge all FPN levels
    # [batch_size * num_anchors_fmap * num_fmaps, 2] where 2 is 
    # (batch_idx, anchor_idx)
    anchor_idxes = tf.concat(list_anchor_idxes, axis=0)
    # [batch_size * num_anchors_fmap * num_fmaps, 4]
    anchors = tf.concat(list_anchors, axis=0)
    # [batch_size * num_anchors_fmap * num_fmaps, ]
    class_ids = tf.concat(list_class_ids, axis=0)
    # [batch_size * num_anchors_fmap * num_fmaps, ]
    scores = tf.concat(list_scores, axis=0)
    
    # unique classes, [num_classes, ]
    ids = tf.unique(class_ids)[0]
    
    # batch indicators, [batch_size * num_anchors_fmap * num_fmaps, ], each
    # indicates which batch where the image belongs to 
    batch_indicators = tf.gather(anchor_idxes, 0, axis=1)
    
    # max number of objects in a class for the batch of images
    max_objects_per_class = batch_size * max_objects_per_class_per_img
    
    def nms_per_class(class_id):
        """
        Applies NMS to a given class.

        Parameters
        ----------
        class_id : integer
            The object class id.

        Returns
        -------
        select_idxes : tf tensor, [max_objects_per_class, ]
            Selected indices after NMS, padded with -1 if needed.

        """
        idxes = tf.where(class_ids == class_id)[:,0]
        idxes = tf.cast(idxes, ids.dtype)
        nms_idxes = tf.image.non_max_suppression(
            boxes=tf.gather(anchors, idxes), 
            scores=tf.gather(scores, idxes), 
            max_output_size=max_objects_per_class,
            iou_threshold=iou_threshold)
        # [(batch_size * num_anchors_fmap * num_fmaps)_per_class_filtered, ]
        select_idxes = tf.gather(idxes, nms_idxes)
        # pad with -1 to have same shape for all classes, 
        # [max_objects_per_class, ]
        gap = max_objects_per_class - tf.shape(select_idxes)[0]
        select_idxes = tf.pad(select_idxes, [[0,gap]], constant_values=-1)
        return select_idxes
    
    # parallel computing applied to all classes, 
    # [num_classes, max_objects_per_class]
    select_idxes = tf.map_fn(nms_per_class, ids)
    # remove -1 paddings, 
    # [(batch_size * num_anchors_fmap * num_fmaps)_filtered, ]
    select_idxes = tf.reshape(select_idxes, [-1])
    select_idxes = tf.gather(select_idxes, tf.where(select_idxes > -1)[:,0])
    
    # [(batch_size * num_anchors_fmap * num_fmaps)_filtered, ]
    batch_indicators = tf.gather(batch_indicators, select_idxes)
    # [(batch_size * num_anchors_fmap * num_fmaps)_filtered, 4]
    anchors = tf.gather(anchors, select_idxes)
    # [(batch_size * num_anchors_fmap * num_fmaps)_filtered, ]
    class_ids = tf.gather(class_ids, select_idxes)
    scores = tf.gather(scores, select_idxes)
    
    # get detections for each image 
    anchors_batch, class_ids_batch, scores_batch = [], [], []
    for b in range(batch_size):
        idxes = tf.where(batch_indicators == b)[:,0]
        # [(num_anchors_fmap * num_fmaps)_per_img_filtered, 4]
        anchors_per_img = tf.gather(anchors, idxes)
        # [(num_anchors_fmap * num_fmaps)_per_img_filtered, ]
        class_ids_per_img = tf.gather(class_ids, idxes)
        scores_per_img = tf.gather(scores, idxes)
        
        anchors_batch.append(anchors_per_img)
        class_ids_batch.append(class_ids_per_img)
        scores_batch.append(scores_per_img)
    
    return anchors_batch, class_ids_batch, scores_batch


class NMS_FPN(tf.keras.layers.Layer):
    """
    Defines a class NMS as a subclass of TF layer.
    
    """
    def __init__(
            self, 
            max_objects_per_class_per_img=100,
            iou_threshold=0.5,
            batch_size=2,
            **kwarg
            ):
        super(NMS_FPN, self).__init__(**kwarg)
        self.max_objects_per_class_per_img = max_objects_per_class_per_img
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        
    def call(self, inputs):
        list_anchor_idxes = inputs[0]
        list_anchors = inputs[1]
        list_class_ids = inputs[2]
        list_scores = inputs[3]
        return nms_fpn(
            list_anchor_idxes, 
            list_anchors, 
            list_class_ids, 
            list_scores,
            self.max_objects_per_class_per_img,
            self.iou_threshold,
            self.batch_size)