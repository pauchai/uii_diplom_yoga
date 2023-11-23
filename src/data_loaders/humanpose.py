import math
import os
import random
import matplotlib.pyplot as plt

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from ..utils.heatmap import gen_gt_heatmap
from ..utils.keypoints import normalize_landmark
from ..utils.pre_processing import square_crop_with_keypoints
from .augmentation import augment_img
from .augmentation_utils import random_occlusion

class DataSequence(Sequence):
  def __init__(self, images_folder, label_file,   batch_size = 8, input_size = (128, 128), heatmap_size = (64, 64), heatmap_sigma = 4, shuffle=True, augment=False, random_flip=False, clip_landmark = False, symmetry_point_ids = None ):
    self.image_folder = images_folder
    self.label_file = label_file
    self.batch_size = batch_size
    self.input_size = input_size
    self.heatmap_size = heatmap_size
    self.heatmap_sigma = heatmap_sigma
    self.symmetry_point_ids = symmetry_point_ids
    self.shuffle = shuffle
    self.augment = augment
    self.random_flip = random_flip
    self.clip_landmark = clip_landmark

    with open(self.label_file, "r") as fp:
      self.anno = json.load(fp)

    if shuffle:
      random.shuffle(self.anno)

  def __len__(self):
    return math.ceil(len(self.anno) / float(self.batch_size))
  def _load_batch_data_from_anno(self, idx):
    return self.anno[idx *  self.batch_size: (1 + idx) * self.batch_size]
  def __getitem__(self, idx):
    batch_data = self._load_batch_data_from_anno(idx)
    batch_image = []
    batch_landmark = []
    batch_heatmap = []
    for data in batch_data:
      image, landmark, heatmap = self.load_data(self.image_folder, data)
      batch_image.append(image)
      batch_landmark.append(landmark)
      batch_heatmap.append(heatmap)
    batch_image = np.array(batch_image)
    batch_heatmap = np.array(batch_heatmap)
    batch_landmark = np.array(batch_landmark)

    batch_image = self.preprocess_images(batch_image)
    batch_landmark = self.preprocess_landmarks(batch_landmark)

    if self.clip_landmark:
      batch_landmark[batch_landmark < 0] = 0
      batch_landmark[batch_landmark > 1] = 1

    return batch_image, (batch_landmark, batch_heatmap)
    #return batch_image, batch_heatmap

  def load_data(self, image_folder, data):
    path = os.path.join(image_folder, data['image'])
    image = cv2.imread(path)
    landmark = data["points"]
    bbox = data["bbox"]

    landmark = np.array(landmark)

    # Convert all (-1, -1) to (0, 0)
    for i in range(landmark.shape[0]):
        if landmark[i][0] == -1 and landmark[i][1] == -1:
            landmark[i, :] = [0, 0]

    # Generate visibility mask
    # visible = inside image + not occluded by simulated rectangle
    # (see BlazePose paper for more detail)
    visibility = np.ones((landmark.shape[0], 1), dtype=int)
    for i in range(len(visibility)):
        if 0 > landmark[i][0] or landmark[i][0] >= image.shape[1] \
                or 0 > landmark[i][1] or landmark[i][1] >= image.shape[0]:
            visibility[i] = 0

    if self.augment:
      image, landmark = square_crop_with_keypoints(
          image, bbox, landmark, pad_value="random"
      )


    old_img_size = np.array([image.shape[1], image.shape[0]])
    #resize image
    image = cv2.resize(image, self.input_size)
    #resize landmark
    landmark = (
        landmark * np.divide(np.array(self.input_size).astype(float), old_img_size)
    )

    ### START FLIPPING
    if self.random_flip and  random.choice([0,1]):
      # Horizintal flip
      # and update ther order of landmarks point
      image = cv2.flip(image, 1)

      # Mark missing keypoints
      missing_idxs = []
      for i in range(landmark.shape[0]):
        if landmark[i, 0 ] == 0 and landmark[i, 1] == 0:
          missing_idxs.append(i)

      # Flip landmark
      landmark[:, 0] = self.input_size[0] - landmark[:, 0]

      # Restore missing keypoints
      for i in missing_idxs:
        landmark[i, 0] = 0
        landmark[i, 1] = 1

      if self.symmetry_point_ids is not None:
        for p1, p2 in self.symmetry_point_ids:

          l = landmark[p1, :].copy()
          landmark[p1, :] = landmark[p2, :].copy()
          landmark[p2, :] = l

    ##### END FLIPPING


    #random occlusion by drowing random rectangles
    if self.augment and random.random() < 0.2:
      landmark = landmark.reshape(-1, 2)
      image, visibility = random_occlusion(image, landmark, visibility=visibility,
                                          rect_ratio=((0.2, 0.5), (0.2, 0.5)),
                                          rect_color="random"
                                         )

    #augment
    if (self.augment):
      print("make augment")
      image, landmark = augment_img(image, landmark)


    #concatenate visibility into landmark
    visibility = np.array(visibility)
    visibility = visibility.reshape((landmark.shape[0], 1))
    landmark = np.hstack((landmark, visibility))


    # Heatmap
    gtmap = None
    gtmap_kps = landmark.copy()
    gtmap_kps[:, :2] = (
              np.array(gtmap_kps[:, :2]).astype(float) *
              np.array(self.heatmap_size) / np.array(self.input_size)
                        ).astype(int)
    gtmap = gen_gt_heatmap(gtmap_kps, self.heatmap_sigma, self.heatmap_size)

    return image, landmark, gtmap

  def preprocess_images(self, images):
    for i in range(images.shape[0]):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    images = images / 255.0
    images -= mean
    return images

  def preprocess_landmarks(self, landmarks):

        first_dim = landmarks.shape[0]
        landmarks = landmarks.reshape((-1, 3))
        landmarks = normalize_landmark(landmarks, self.input_size)
        landmarks = landmarks.reshape((first_dim, -1))
        return landmarks




