'''
A helper function to generate a numpy array defining anchor boxes.

Copyright (C) 2020 Alf O'Kenney,

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, InputSpec, Layer

import numpy as np

from bounding_box_utils.bounding_box_utils import convert_coordinates

def build_priors(loc_tensor, img_dim, this_scale, next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0], two_boxes_for_ar1=True,
                 clip_boxes=False, variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids', normalize_coords=False):
    '''
    A nuympy array containing anchor box coordinates and variances based on the
    input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each
    spatial unit of the input tensor. The number of anchor boxes created per
    unit depends on the arguments `aspect_ratios` and `two_boxes_for_ar1`, in
    the default case it is 4. The boxes are parameterized by the coordinate
    tuple `(xmin, xmax, ymin, ymax)`.

    The logic implemented by this layer is identical to the logic in the module
    `ssd_box_encode_decode_utils.py`.

    The purpose of having this layer in the network is to make the model
    self-sufficient at inference time. Since the model is predicting offsets to
    the anchor boxes (rather than predicting absolute box coordinates directly),
    one needs to know the anchor box coordinates in order to construct the final
    prediction boxes from the predicted offsets. If the model's output tensor
    did not contain the anchor box coordinates, the necessary information to
    convert the predicted offsets back to absolute coordinates would be missing
    in the model output. The reason why it is necessary to predict offsets to
    the anchor boxes rather than to predict absolute box coordinates directly is
    explained in `README.md`.

    Input shape:
        4D tensor of shape `(batch, height, width, channels)`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis
        contains the four anchor box coordinates and the four variance values
        for each box.

    All arguments need to be set to the same values as in the box encoding
    process, otherwise the behavior is undefined. Some of these arguments
    are explained in more detail in the documentation of the `SSDBoxEncoder`
    class.

    Required Arguments:

        loc_tensor (tensor):
            4D tensor of shape `(batch, height, width, channels)`.
            The output of the localization predictor layer.

        img_dim (int):
            The side length of the input images.
            NOTE: it is assumed that the input images have the same dimension
                  for width and height.

        this_scale (float):
            A float in [0, 1], the scaling factor for the size of the
            generated anchor boxes as a fraction of the shorter side of the
            input image.

        next_scale (float):
            A float in [0, 1], the next larger scaling factor. Only relevant
            if `two_boxes_for_ar1 == True`.

    Optional Arguments:

        aspect_ratios (list):
            The list of aspect ratios for which default boxes are to be
            generated for this layer.

        two_boxes_for_ar1 (bool):
            Only relevant if `aspect_ratios` contains 1. If `True`, two
            default boxes will be generated for aspect ratio 1. The first
            will be generated using the scaling factor for the respective
            layer, the second one will be generated using geometric mean of
            said scaling factor and next bigger scaling factor.

        clip_boxes (bool):
            If `True`, clips the anchor box coordinates to stay within image
            boundaries.

        variances (list):
            A list of 4 floats >0. The anchor box offset for each coordinate
            will be divided by its respective variance value.

        coords (str):
            The box coordinate format to be used internally in the model
            (i.e. this is not the input format of the ground truth labels).
            Can be either 'centroids' for the format `(cx, cy, w, h)`
            (box center coordinates, width, and height), 'corners' for the
            format `(xmin, ymin, xmax,  ymax)`, or 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`.

        normalize_coords (bool):
            Set to `True` if the model uses relative instead of absolute
            coordinates, i.e. if the model predicts box coordinates within
            [0,1] instead of absolute coordinates.
    '''

    # =========================== Perform Checks ===========================

    if this_scale < 0 or next_scale < 0 or this_scale > 1:
        raise ValueError(
            ('`this_scale` must be in [0, 1] and `next_scale` must be >0, '
                'but `this_scale` == {}, `next_scale` == {}').format(
                    this_scale, next_scale)
        )

    if len(variances) != 4:
        raise ValueError(
            ('4 variance values must be pased, but {} values were '
                'received.').format(len(variances))
        )
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError(
            ('All variances must be >0, but the variances given are '
                '{}'.format(variances))
        )

    # Compute the number of boxes per cell
    n_boxes = len(aspect_ratios)
    if 1 in aspect_ratios and two_boxes_for_ar1:
        n_boxes += 1

    # Compute the box [width, height] for all aspect ratios
    wh_list = []
    scaled_dim =  this_scale * img_dim
    for ar in aspect_ratios:
        if ar == 1:
            # Compute the regular anchor box for aspect ratio 1.
            box_dim = scaled_dim
            wh_list.append([box_dim]*2)
            if two_boxes_for_ar1:
                # Compute one slightly larger version using the geometric
                # mean of this scale value and the next.
                box_dim = np.sqrt(this_scale * next_scale) * img_dim
                wh_list.append([box_dim]*2)
        else:
            wh_list.append([scaled_dim * np.sqrt(ar), scaled_dim / np.sqrt(ar)])
    wh_list = np.array(wh_list)

    # We need the shape of the input tensor
    feature_map_dim = loc_tensor.shape[1]

    # Compute the grid of box center points. They are identical for all
    # aspect ratios.

    # Compute the step sizes, i.e. how far apart the anchor box center
    # points will be vertically and horizontally.
    step_size = img_dim / feature_map_dim

    # Compute the offsets, i.e. at what pixel values the first anchor box
    # center point will be from the top and from the left of the image.
    offset_size = 0.5

    # Now that we have the offsets and step sizes, compute the grid of
    # anchor box center points.
    centers = np.linspace(
        offset_size * step_size,
        (offset_size + feature_map_dim - 1) * step_size,
        feature_map_dim
    )
    cx_grid, cy_grid = np.meshgrid(centers, centers)

    # Create a 4D tensor template of shape
    #   `(feature_map_dim, feature_map_dim, n_boxes, 4)`
    # where the last dimension will contain `(cx, cy, w, h)`
    boxes_tensor = np.zeros(
        (feature_map_dim, feature_map_dim, n_boxes, 4))

    boxes_tensor[:, :, :, 0] = np.expand_dims(cx_grid, -1)  # Set cx
    boxes_tensor[:, :, :, 1] = np.expand_dims(cy_grid, -1)  # Set cy
    boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w (broadcast)
    boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h (broadcast)

    # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
    boxes_tensor = convert_coordinates(
        boxes_tensor, start_index=0, conversion='centroids2corners')

    # If `clip_boxes` is enabled, clip the coordinates to lie within the
    # image boundaries
    if clip_boxes:
        x_coords = boxes_tensor[:, :, :, [0, 2]]
        x_coords[x_coords >= img_dim] = img_dim - 1
        x_coords[x_coords < 0] = 0
        boxes_tensor[:, :, :, [0, 2]] = x_coords
        y_coords = boxes_tensor[:, :, :, [1, 3]]
        y_coords[y_coords >= img_dim] = img_dim - 1
        y_coords[y_coords < 0] = 0
        boxes_tensor[:, :, :, [1, 3]] = y_coords

    # If `normalize_coords` is enabled, normalize the coordinates to be
    # within [0,1]
    if normalize_coords:
        boxes_tensor[:, :, :, [0, 2]] /= img_dim
        boxes_tensor[:, :, :, [1, 3]] /= img_dim

    # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we
    #       don't have to unnecessarily convert back and forth.
    if coords in ['centroids', 'minmax']:
        # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)` or to
        # `(xmin, xmax, ymin, ymax)`.
        boxes_tensor = convert_coordinates(
            boxes_tensor,
            start_index=0,
            conversion='corners2' + coords,
            border_pixels='half'
        )

    # Create a tensor to contain the variances and append it to
    # `boxes_tensor`. This tensor has the same shape as `boxes_tensor` and
    # simply contains the same 4 variance values for every position in the
    # last axis. Has shape
    #   `(feature_map_dim, feature_map_dim, n_boxes, 4)`
    variances_tensor = np.zeros_like(boxes_tensor)
    variances_tensor += variances  # Long live broadcasting

    # Now `boxes_tensor` becomes a tensor of shape
    #   `(feature_map_dim, feature_map_dim, n_boxes, 8)`
    return np.concatenate((boxes_tensor, variances_tensor), axis=-1)
