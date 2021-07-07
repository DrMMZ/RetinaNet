"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Utilities
"""

# some functions such as resize_image(), resize_masks(), masks_boxes(), 
# Dataset(), compute_ap() and compute_mAP() are very strongly influenced by
# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py

import time
import cv2
import scipy.ndimage
import skimage.io, skimage.color, skimage.transform
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


############################################################
#  Visualization
############################################################

def draw_boxes(
        image, 
        boxes=None, 
        color=(255, 0, 0), 
        thickness=2, 
        captions=None
        ):
    """
    Displays an image with bounding boxes.

    Parameters
    ----------
    image : numpy array, [height, width, channels]
        The shape of the image in pixels.
    boxes : numpy array, [num_boxes, 4], optional
        The bounding boxes, where 4 is the corner coordinates (y1, x1, y2, x2). 
        The default is None, meaning displaying the image only.
    color : tuple, optional
        The color of boxes (red, green, blue) in pixels. The default is 
        (255, 0, 0), i.e., red.
    thickness : integer, optional
        The thickness of boundary in boxes. The default is 2.
    captions : list, optional
        The captions of boxes with length of num_boxes. The default is None.

    Returns
    -------
    image1 : numpy array, [height, width, channels]
        The image with boxes.

    """
    boxes = boxes.astype(np.int32)
    num_boxes = boxes.shape[0]
    image1 = image.copy()
    
    for i in range(num_boxes):
        y1, x1, y2, x2 = boxes[i]
        image1 = cv2.rectangle(image1, (x1,y1), (x2,y2), color, thickness)
        
        if captions:
            text = captions[i]
            image1 = cv2.putText(
                image1, 
                text, 
                (x1+1, y1+5), 
                fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=1, 
                color=(255, 255, 255), 
                thickness=1)
            
    return image1


############################################################
#  Boxes
############################################################

def compute_ious(boxes1, boxes2):
    """
    Vectorized implementation for computing IoU overlaps between two sets of 
    boxes.

    Parameters
    ----------
    boxes1, boxes2 : numpy array, [num_boxes, 4]
        The set of bounding boxes, where 4 is the corner coordinates.

    Returns
    -------
    ious : numpy array, [num_boxes1, num_boxes2]
        The IoU overlaps.

    """
    # repeat every row of boxes1 num_boxes2 times, and boxes2 
    # num_boxes1 times
    b1 = np.repeat(boxes1, repeats=boxes2.shape[0], axis=0)
    b2 = np.tile(boxes2, reps=(boxes1.shape[0], 1))
    
    # compute intersection areas
    b1_y1, b1_x1, b1_y2, b1_x2 = np.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = np.split(b2, 4, axis=1)
    y1 = np.maximum(b1_y1, b2_y1)
    x1 = np.maximum(b1_x1, b2_x1)
    y2 = np.minimum(b1_y2, b2_y2)
    x2 = np.minimum(b1_x2, b2_x2)
    inter_areas = np.maximum(x2-x1, 0) * np.maximum(y2-y1, 0)
    
    # compute union areas
    b1_areas = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_areas = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_areas = b1_areas + b2_areas - inter_areas
    
    # compute IoU overlaps and reshape to [num_boxes1, num_boxes2]
    ious = inter_areas / union_areas
    return ious.reshape([boxes1.shape[0], boxes2.shape[0]])


def compute_offsets(boxes1, boxes2):
    """
    Computes the differences between two sets of boxes.

    Parameters
    ----------
    boxes1, boxes2 : numpy array, [num_boxes, 4]
        The set of bounding boxes, where 4 is the corner coordinates.

    Returns
    -------
    offsets : numpy array, [num_boxes, 4]
        Offsets from boxes1 to boxes2 in the form of the center coordinates, 
        i.e., (y, x, h, w).

    """
    boxes1 = boxes1.astype(np.float32)
    boxes2 = boxes2.astype(np.float32)
    
    height1 = boxes1[:,2] - boxes1[:,0]
    width1 = boxes1[:,3] - boxes1[:,1]
    y_center1 = boxes1[:,0] + 0.5 * height1
    x_center1 = boxes1[:,1] + 0.5 * width1
    
    height2 = boxes2[:,2] - boxes2[:,0]
    width2 = boxes2[:,3] - boxes2[:,1]
    y_center2 = boxes2[:,0] + 0.5 * height2
    x_center2 = boxes2[:,1] + 0.5 * width2
    
    dy = (y_center2 - y_center1) / height1
    dx = (x_center2 - x_center1) / width1
    dh = np.log(height2 / height1)
    dw = np.log(width2 / width1)
    
    offsets = np.stack([dy, dx, dh, dw], axis=1)
    return offsets


def apply_offsets(boxes, offsets):
    """
    Modify boxes in an image by the given offsets.

    Parameters
    ----------
    boxes : tf tensor, [num_boxes, 4]
        The set of bounding boxes, where 4 is the corner coordinates.
    offsets : tf tensor, [num_boxes, 4] 
        The given offsets, where 4 is the center coordinates.

    Returns
    -------
    boxes1 : tf tensor, [num_boxes, 4]
        The resulting boxes.
    
    """
    boxes = tf.cast(boxes, dtype=offsets.dtype)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
    
    # convert to center_y, center_x, h, w
    h = y2 - y1
    w = x2 - x1
    center_y = y1 + 0.5 * h
    center_x = x1 + 0.5 * w
    
    # apply offsets to boxes
    off_y, off_x, off_h, off_w = tf.split(offsets, 4, axis=-1)
    center_y += off_y * h
    center_x += off_x * w
    h *= tf.math.exp(off_h)
    w *= tf.math.exp(off_w)
    
    # convert back to corner coordinates
    y1 = center_y - 0.5 * h
    x1 = center_x - 0.5 * w
    y2 = y1 + h
    x2 = x1 + w
    boxes1 = tf.concat([y1,x1,y2,x2], axis=1)
    return boxes1


def clip_boxes(boxes, window):
    """
    Clips boxes to a given window.

    Parameters
    ----------
    boxes : tf tensor, [num_boxes, 4]
        The bounding boxes of an image where 4 is the corner coordinates.
    window : tuple/list
         The corner coordinates (y1, x1, y2, x2) represents the location to 
         clip on.

    Returns
    -------
    boxes1 : tf tensor, [num_boxes, 4]
        The resulting boxes.
    
    """
    window = tf.cast(window, dtype=boxes.dtype)
    wy1, wx1, wy2, wx2 = tf.split(window, 4, axis=-1) 
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
    
    # clip boxes s.t. wy1<=y1,y2<=wy2, wx1<=x1,x2<=wx2
    y1 = tf.maximum(tf.minimum(y1,wy2), wy1)
    y2 = tf.maximum(tf.minimum(y2,wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1,wx2), wx1)
    x2 = tf.maximum(tf.minimum(x2,wx2), wx1)
    boxes1 = tf.concat([y1,x1,y2,x2], axis=1)
    return boxes1


############################################################
#  Image Preprocessing
############################################################

def load_image(image_path):
    """
    Loads an image into RGB.

    Parameters
    ----------
    image_path : string
        The image file path.

    Returns
    -------
    image : numpy array, [height, width, 3]
        Loaded image where 3 is RGB channels.

    """
    image = skimage.io.imread(image_path)
    
    # convert to RGB if it is grayscale
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image) 
        
    # convert to RGB if it is RGBA
    if image.shape[-1] == 4:
        image = image[..., :3]  
        
    return image


def resize_image(
        image, 
        shortest_side=512, 
        longest_side=1024, 
        upscale_factor=1.0,
        mode='crop'
        ):
    """
    Resizes an image without changing the aspect ratio.

    Parameters
    ----------
    image : numpy array, [height, width, channels]
        The image needed to be resized.
    shortest_side : integer, optional
        The shortest side of the resized image. Note that it only applies to
        the 'pad_fpn' and 'crop' mode. The default is 512.
    longest_side : integer, optional
        The longest side of the resized image. Note that it only applies to the
        'pad_square' mode. The default is 1024.
    upscale_factor : float, optional
        The scale factor >= 1.0 to upscale the image. The default is 1.0.
    mode : string, optional
        The resizing method in {'none'', 'pad_square', 'pad_fpn', 'crop}. The 
        default is 'crop'.
        * 'none' : [height, width, channels], no resizing nor padding applied.
        * 'pad_square' : [longest_side, longest_side, channels], padded with 0 
        to keep the same aspect ratio.
        * 'pad_fpn' : [l1, l2, channels] where l1 and l2 >= shortest_side s.t.
        shortest_side % 128 == 0, padded with 0 to keep the same aspect ratio.
        * 'crop' : [shorest_side, shortest_side, channels].

    Returns
    -------
    image : numpy array, [resized_h, resized_w, channels]
        The resized image depending on the mode with the same data type as the
        input image.
    window : tuple
        The corner coordinates (y1, x1, y2, x2) indicates the location of the
        resized image before padding.
    scale : float
        The scale factor from original to resized image, and used in the later
        resize_mask().
    padding : list
        The padding applied to the original image in the form of pixels
        [(top, bottom), (left, right), (0, 0)], and used in the later
        resize_mask().
    crop : tuple
        The coordinates (y1, x1, h, w) indicates the location of the
        cropped image over the resized image, and used in the later 
        resize_mask().

    """
    assert mode in ['none', 'pad_square', 'pad_fpn', 'crop']
    
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1.0
    padding = [(0,0), (0,0), (0,0)]
    crop = None
    
    if mode == "none":
        return image, window, scale, padding, crop
    
    # upscale for all modes, scale >= 1.0
    scale = max(1.0, shortest_side / min(h, w))
    if scale < upscale_factor:
        scale = upscale_factor
        
    # downscale only if mode == 'pad_square', scale < 1.0
    if mode == 'pad_square':
        max_side = max(h, w)
        if round(max_side * scale) > longest_side:
            scale = longest_side / max_side
            
    # resize image using bilinear
    if scale != 1.0:
        image = skimage.transform.resize(
            image, 
            output_shape=(round(h * scale), round(w * scale)),
            preserve_range=True, 
            anti_aliasing=False)
        
    if mode == 'pad_square':
        h, w = image.shape[:2]
        top_pad = (longest_side - h) // 2
        bottom_pad = longest_side - h - top_pad
        left_pad = (longest_side - w) // 2
        right_pad = longest_side - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        
    elif mode == 'pad_fpn':
        # hard code minimum image size
        min_fpn_stride = 2**7
        assert shortest_side % min_fpn_stride == 0, \
            'Shortest side must be a multiple of 128.'
        h, w = image.shape[:2]
        if h % min_fpn_stride > 0:
            max_h = h - (h % min_fpn_stride) + min_fpn_stride
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        if w % min_fpn_stride > 0:
            max_w = w - (w % min_fpn_stride) + min_fpn_stride
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        
    else:
        h, w = image.shape[:2]
        # note that h (or w) - shortest_side maybe 0
        y = random.randint(0, (h - shortest_side))
        x = random.randint(0, (w - shortest_side))
        crop = (y, x, shortest_side, shortest_side)
        image = image[y:y + shortest_side, x:x + shortest_side]
        window = (0, 0, shortest_side, shortest_side)
        
    return image.astype(image_dtype), window, scale, padding, crop


############################################################
#  Masks
############################################################

def resize_masks(masks, scale, padding, crop=None):
    """
    Resizes a set of masks of an image.

    Parameters
    ----------
    masks : numpy array, [height, width, num_masks]
        The set of masks of an image, each mask has shape [height, width] a 
        binary array representing an object.
    scale : float
        The scale factor outputed by resize_image().
    padding : list
        The padding outputed by resize_image().
    crop : tuple, optional
        The cropping outputed by resize_image(). The default is None.

    Returns
    -------
    masks : numpy array, [resized_h, resized_w, num_masks]
        The resized masks with the same data type as the input masks.

    """
    masks_dtype = masks.dtype
    
    # resize, note that scipy zoom is faster than skimage resize
    h, w = masks.shape[:2]
    masks = scipy.ndimage.zoom(masks, zoom=[scale, scale, 1], order=0)
    
    if crop is not None:
        y1, x1, h, w = crop
        y2 = y1 + h
        x2 = x1 + w
        masks = masks[y1:y2, x1:x2]
    else:
        masks = np.pad(masks, padding)
        
    return masks.astype(masks_dtype)


def masks_boxes(masks):
    """
    Loads the corresponding bounding boxes from a set of masks.

    Parameters
    ----------
    masks : numpy array, [height, width, num_masks]
        The set of masks of an image, each mask has shape [height, width] a 
        binary array representing an object.

    Returns
    -------
    boxes : numpy array, [num_masks, 4] 
        The corresponding bounding boxes extracted from masks, where 4 is the 
        corner coordinates (y1, x1, y2, x2).

    """
    num_objects = masks.shape[-1]
    boxes = np.zeros((num_objects, 4), np.int32)
    
    for i in range(num_objects):
        mask = masks[:,:,i]
        horizontal_idxes = np.where(np.any(mask, axis=0))[0]
        vertical_idxes = np.where(np.any(mask, axis=1))[0]
        
        if horizontal_idxes.shape[0]:
            x1, x2 = horizontal_idxes[0], horizontal_idxes[-1]
            y1, y2 = vertical_idxes[0], vertical_idxes[-1]
            # note that corner coordinates are close-open
            x2 += 1
            y2 += 1
            
        else:
            # no object, maybe caused by cropping
            y1, x1, y2, x2 = 0, 0, 0, 0
            
        boxes[i] = np.array([y1, x1, y2, x2])
        
    return boxes.astype(np.int32)


############################################################
#  Augmentation
############################################################

class FlipLR(object):
    """flip left or right augmentation"""
    def augment(self, seed, image):
        # horizontal flip
        with tf.device('/cpu:0'):
            tf.random.set_seed(seed)
            image = tf.image.random_flip_left_right(image)
            return image
        

class FlipUD(object):
    """flip up or down augmentation"""
    def augment(self, seed, image):
        # vertical flip
        with tf.device('/cpu:0'):
            tf.random.set_seed(seed)
            image = tf.image.random_flip_up_down(image)
            return image
        
        
class Rotate(object):
    """randomly rotate 90, 180 or 270 degrees counter-clockwise"""
    def augment(self, seed, image):
        with tf.device('/cpu:0'):
            k = seed % 4
            image = tf.image.rot90(image, k)
            return image
        
        
class Rotate1(object):
    """randomly rotate within 90 degree counter-clockwise"""
    def augment(self, seed, image):
        # note that seed in [0,100]
        if seed % 2 == 0:
            rad = (seed/100) * (0.5 * np.math.pi)
        else:
            rad = 0
        # print('angle', rad*180/np.math.pi)
        with tf.device('/cpu:0'):
            image = tfa.image.rotate(
                images=image,
                angles=rad,
                fill_mode='constant',
                fill_value=0)
            return image
        

class Contrast(object):
    """random contrast"""
    def __init__(self, lower=0.5, upper=2.0):
        self.lower = lower
        self.upper = upper
    def augment(self, seed, image):
        with tf.device('/cpu:0'):
            tf.random.set_seed(seed)
            image = tf.image.random_contrast(image, self.lower, self.upper)
            return image
        
        
class Augment(object):
    """
    Defines an augmentation class.
    
    """
    def __init__(self, augmenters):
        """
        A constructor.

        Parameters
        ----------
        augmenters : list
            Each element is an augmentation defined in [FlipLR, FlipUD, Rotate,
            Rotate1, Contrast].

        Returns
        -------
        None.

        """
        self.augmenters = augmenters
    def augment(self, seed, image):
        """
        Transforms the given image.

        Parameters
        ----------
        seed : integer
            A positive integer for a seed.
        image : numpy/tf tensor
            The given image needed to be transformed.

        Returns
        -------
        image : numpy/tf tensor
            The transformed image.

        """
        for a in self.augmenters:
            image = a.augment(seed, image)
        return image


############################################################
#  Dataset
############################################################

class Dataset(object):
    """
    Defines a dataset class.
    
    """
    
    def __init__(self):
        # image_info, list of dictionaries 
        # {'id':image_id, 'path':path, 'height':height, 'width':width, ...}
        self.image_info = []
        
        # class_info, list of dictionaries
        # {'id':class_id, 'name':class_name}
        self.class_info = [{'id':0, 'name':'background'}]
    
        
    def add_image(self, image_id, path, **kwargs):
       """
       Adds image information to the dataset.

       Parameters
       ----------
       image_id : string/integer
           The image ID.
       path : string
           The image path.
       **kwargs : string/integer
           Additional information, e.g., image's height, width and so on.

       Returns
       -------
       None.

       """
       image_info = {'id':image_id, 'path':path,}
       image_info.update(kwargs)
       self.image_info.append(image_info)
       
       
    def add_class(self, class_id, class_name):
         """
         Adds class information to the dataset.
 
         Parameters
         ----------
         class_id : integer
             The class ID.
         class_name : string
             The corresponding class name.
 
         Returns
         -------
         None.
 
         """       
         for info in self.class_info:
             if info['id'] == class_id:
                 # skip if class_id already existed
                 return
         self.class_info.append({'id':class_id, 'name':class_name})
         
         
    def prepare(self):
        """
        Prepares the dataset and needs to be called before using it.

        Returns
        -------
        None.

        """
        # info from class_info and image_info
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [
            a_class_info['name'] for a_class_info in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
        
        
    def get_info(self):
        """
        Gets the dataset information.

        Returns
        -------
        class_ids : numpy array, [num_classes, ]
            The class ids.
        class_names : list
            Each element is a name corresponding to the class id.

        """
        print('There are %d images and %d classes.' \
              % (self.num_images, self.num_classes))
        return self.class_ids, self.class_names
        
        
    def get_image_info(self, image_id):
        """
        Gets an image information.

        Parameters
        ----------
        image_id : integer
            An image id from image_ids.

        Returns
        -------
        image_info : dictionary
            The image information with keys 'id', 'path' and so on; see 
            add_image().

        """
        return self.image_info[image_id]
    
    
    def reference(self, image_id_str):
        """
        Gets the image id from the name of image.

        Parameters
        ----------
        image_id_str : string
            The name of image, image_info['id'].

        Returns
        -------
        image_id : integer
            The corresponding id.

        """
        for image_id in self.image_ids:
            if self.get_image_info(image_id)['id'] == image_id_str:
                return image_id
    
    
    def load_image(self, image_id):
        """
        Loads an image into RGB.

        Parameters
        ----------
        image_id : integer
            An image id from image_ids.

        Returns
        -------
        image : numpy array, [height, width, 3]
            The RGB image in pixel.

        """
        image_path = self.get_image_info(image_id)['path']
        image = load_image(image_path)
        return image
    
    
    def load_masks_ids(self, image_id):
        """
        Loads a set of masks and corresponding class ids from an image. It 
        needs to be re-defined in a subclass.

        Parameters
        ----------
        image_id : integer
            The image id from image_ids.

        Returns
        -------
        masks : numpy array, [h, w, num_objects]
            A binary array indicates the locations of objects in the image.
        class_ids : numpy array, [num_objects, ]
            The set of corresponding class ids for masks.

        """
        print('Need to re-define in a subclass.')
        masks = np.empty([0,0,0])
        class_ids = np.empty([0], np.int32)
        return masks, class_ids
    
    
    def load_data(
            self, 
            image_id, 
            shortest_side=512, 
            longest_side=1024, 
            upscale_factor=1.0,
            mode='crop',
            augmenters=None, # e.g., [utils.Rotate(), utils.FlipLR()],
            max_num_crops=10,
            verbose=0
            ):  
        """
        Loads preprocessed image and corresponding ground-truth bounding boxes,
        class ids and other information.

        Parameters
        ----------
        image_id : integer
            The image id from image_ids.
        shortest_side, longest_side, upscale_factor, mode : 
            See resize_image().
        augmenters : 
            See Augment().
        max_num_crops : integer, optional
            A maximum number of crops when mode='crop' and there is no object
            after a crop. The default is 10.
        verbose : binary, optional
            Whether to print out the preprocessing time. The default is 0.

        Returns
        -------
        image1 : numpy array, [resized_h, resized_w, 3]
            The preprocessed image.
        boxes : numpy array, [num_objects, 4]
            The ground-truth bounding boxes, where 4 is the corner coordinates. 
        class_ids1 : numpy array, [num_objects, ]
            The set of corresponding class ids for boxes.
        cache : list
            Includes masks, window, scale, padding, crop, original image shape
            and preprocessed image shape.

        """
        # load image, masks and class ids
        t1 = time.time()
        image = self.load_image(image_id)
        t2 = time.time()
        if verbose: print('loading image: %fs' %(t2-t1))
        t1 = time.time()
        masks, class_ids = self.load_masks_ids(image_id)
        t2 = time.time()
        if verbose: print('loading masks: %fs' %(t2-t1))
        image_shape = image.shape
        
        # loop if there is no object after cropping
        num_objects, count = 0, 1
        while num_objects == 0 and count <= max_num_crops:
            if count > 1:
                print('\nNo object %s' % (self.get_image_info(image_id)['id']))
        
            # resize image and masks accordingly
            t1 = time.time()
            image1, window, scale, padding, crop = resize_image(
                image, 
                shortest_side, 
                longest_side, 
                upscale_factor, 
                mode)
            if verbose: print('resize mode:', mode)
            t2 = time.time()
            if verbose: print('resizing image: %fs' %(t2-t1))
            t1 = time.time()
            masks1 = resize_masks(masks, scale, padding, crop)
            t2 = time.time()
            if verbose: print('resizing masks: %fs' %(t2-t1))
            
            # augmentation
            if augmenters is not None:
                # note that seed in [0,100] for Rotate1
                seed = np.random.randint(100)
                a = Augment(augmenters)
                t1 = time.time()
                image1 = a.augment(seed, image1).numpy()
                t2 = time.time()
                if verbose: print('augmenting image: %fs' %(t2-t1))
                mask_augmenters = []
                for x in augmenters:
                    names = ['Rotate', 'Rotate1', 'FlipLR', 'FlipUD']
                    if type(x).__name__ in names:
                        mask_augmenters.append(x)
                a = Augment(mask_augmenters)
                t1 = time.time()
                masks1 = a.augment(seed, masks1).numpy()
                t2 = time.time()
                if verbose: print('augmenting masks: %fs' %(t2-t1))
                
            # filter no object masks caused by cropping
            obj_idxes = np.where(np.sum(masks1, axis=(0,1)) > 0)[0]
            masks1 = masks1[:, :, obj_idxes]
            class_ids1 = class_ids[obj_idxes]
            
            # get corresponding boxes from masks
            boxes = masks_boxes(masks1)
            
            num_objects = obj_idxes.shape[0]
            if count > 1 and num_objects > 0:
                print('\nGot objects after %d croppings.' % (count))
            count += 1
    
        cache = (
            masks1, 
            window, 
            scale, 
            padding, 
            crop, 
            image_shape, 
            image1.shape)
        return image1, boxes, class_ids1, cache
            

############################################################
#  mAP Metric
############################################################

def compute_ap(
        gt_boxes, 
        gt_class_ids, 
        pred_boxes,
        pred_class_ids, 
        pred_scores, 
        iou_threshold=0.5, 
        iou_score_threshold=0.0
        ): 
    """
    Computes Average Precision (AP) of an image for a IoU threshold.

    Parameters
    ----------
    gt_boxes : numpy array, [num_gt_boxes, 4]
        The ground-truth bounding boxes of the image.
    gt_class_ids : numpy array, [num_gt_boxes, ]
        The ground-truth class ids corresponding to gt_boxes.
    pred_boxes : numpy array, [num_pred_boxes, 4]
        The predicted bounding boxes of the image.
    pred_class_ids : numpy array, [num_pred_boxes, ]
        The predicted class ids corresponding to pred_boxes.
    pred_scores : numpy array, [num_pred_boxes, ]
        The predicted scores corresponding to pred_class_ids.
    iou_threshold : float, optional
        A IoU threshold. The default is 0.5.
    iou_score_threshold : float, optional
        A threshold to remove boxes. The default is 0.0.

    Returns
    -------
    ap : float
        The AP result of the image.

    """
    # sort predictions by scores from high to low
    idxes = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[idxes]
    pred_class_ids = pred_class_ids[idxes]
    pred_scores = pred_scores[idxes]
    
    # compute IoU overlaps [pred_boxes, gt_boxes]
    ious = compute_ious(pred_boxes, gt_boxes)
    
    # find matches between predictions and ground-truth
    pred_matches = -1 * np.ones([pred_boxes.shape[0]])
    gt_matches = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # sort gt boxes by ious from high to low
        gt_idxes = np.argsort(ious[i])[::-1]
        # remove low IoU scores
        low_score_idxes = np.where(
            ious[i, gt_idxes] < iou_score_threshold)[0]
        if low_score_idxes.size > 0:
            gt_idxes = gt_idxes[:low_score_idxes[0]]
        # find matches
        for j in gt_idxes:
            if gt_matches[j] > -1:
                continue
            iou_i_j = ious[i, j]
            if iou_i_j < iou_threshold:
                break
            if pred_class_ids[i] == gt_class_ids[j]:
                gt_matches[j] = i
                pred_matches[i] = j
                break
            
    # compute precision and recall at each predicted box
    cum_tps = np.cumsum(pred_matches > -1)
    precisions =  cum_tps / (np.arange(len(pred_matches)) + 1)
    #print(precisions, precisions.dtype)
    recalls = cum_tps.astype(np.float32) / len(gt_matches)
    #print(recalls, recalls.dtype)
    
    # pad with start and end values
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    
    # no increasing in precisions
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i+1])
        
    # compute mean average precision over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum(
        (recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap


def compute_mAP(
        gt_boxes, 
        gt_class_ids, 
        pred_boxes, 
        pred_class_ids, 
        pred_scores, 
        iou_thresholds=None, 
        verbose=True
        ):
    """
    Computes Mean Average Precision (mAP) of an image for a range of IoU 
    thresholds.

    Parameters
    ----------
    gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores : 
        Same as compute_ap().
    iou_thresholds : numpy array, optional
        A range of IoU thresholds, each is used to compute AP of the image. The 
        default is None.
    verbose : boolean, optional
        Whether to display AP and mAP results. The default is True.

    Returns
    -------
    mAP : float
        The mAP result of the image.

    """
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    mAP = []
    for iou_threshold in iou_thresholds:
        ap = compute_ap(
            gt_boxes, 
            gt_class_ids, 
            pred_boxes, 
            pred_class_ids, 
            pred_scores, 
            iou_threshold=iou_threshold)
        if verbose:
            print('AP @%.2f: %.3f' % (iou_threshold, ap))
        mAP.append(ap)
    mAP = np.array(mAP).mean()  
    
    if verbose:
        print('mAP @%.2f-%.2f: %.3f' % (
            iou_thresholds[0], iou_thresholds[-1], mAP))
        
    return mAP
