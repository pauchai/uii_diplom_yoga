import matplotlib.pyplot as plt
import cv2
import numpy as np 

class Visualizer():
  def __init__(self, skeleton_graph):
    self.skeleton_graph = skeleton_graph
  def image_batch_show(self,image_batch):
    fig, axs = plt.subplots(1, len(image_batch), figsize = (35, 5))
    for i, image in enumerate(image_batch):
      axs[i].imshow(image)
    plt.show()


  def landmark_batch_image_joints_show(self, image_batch, landmark_batch):
    fig, axs = plt.subplots(1, len(image_batch), figsize = (35,5))


    for i, (image, landmarks) in enumerate(zip(image_batch, landmark_batch)):


      #рисуем сочленения
      for  from_to_points in self.skeleton_graph:
        landmark_from_idx = from_to_points[0]
        landmark_to_idx = from_to_points[1]

        x0, y0, _ = landmarks[landmark_from_idx]
        x1, y1, _ = landmarks[landmark_to_idx]
        # нарисовать линии между (x0,y0) (x1,x1)
        axs[i].plot([x0, x1], [y0, y1], linewidth=2, color='r')

      # рисуем суставы
      x, y, v = zip(*landmarks)
      colors = ["r" if int(val) == 1 else  "b" for val in v] # 'r' for visible 1 'b' for 0
      axs[i].scatter(x, y, c= colors, marker='o')  # , 'o' for circular marker
      im = axs[i].imshow(image, vmin  = -2, vmax = 2)
      cbar = fig.colorbar(im, ax=axs[i], shrink = 1.0)

    plt.show()

  def heatmap_batch_vstacked_show(self,heatmap_batch):
    fig, axs = plt.subplots(1, len(heatmap_batch), figsize = (25, 35))
    heatmaps = None
    for i, heatmap in enumerate(heatmap_batch):
      heatmaps = np.transpose(heatmap, (2,0,1))
      axs[i].imshow(np.vstack(heatmaps))
    plt.show()


  def heatmap_batch_onimage_show(self,image_batch, heatmap_batch,  background_weight = 0.3, heatmap_weight = 0.8):
    fig, axs = plt.subplots(1, len(heatmap_batch), figsize = (25, 35))
    heatmaps = None
    for i, (image, heatmap) in enumerate(zip(image_batch, heatmap_batch)):
        bimg = image.copy()

        axs[i].imshow(
          self.draw_heatmap_on_image(bimg, heatmap.copy(), background_weight = background_weight, heatmap_weight = heatmap_weight),
          vmin = -2,
          vmax = 2

          
        )

    plt.show()

  def draw_heatmap_on_image(self, img, heatmap, background_weight = 0.3, heatmap_weight = 0.8):
    if (img.shape[-2] != heatmap.shape[-2] ):
      #print("Image shape:", img.shape)
      #print("Heatmap shape:", heatmap.shape)
      heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    #heatmap = heatmap * 255


     # Sum across channels
    summed_heatmap = np.sum(heatmap, axis=-1, keepdims=True)
    summed_heatmap = np.tile(summed_heatmap,3)

    # Normalize to the range [0, 1]
    normalized_heatmap = (summed_heatmap - np.min(summed_heatmap)) / (np.max(summed_heatmap) - np.min(summed_heatmap))

    # Overlay the heatmap on the image
    overlaid_image = ((img + 0.5) * background_weight + normalized_heatmap * heatmap_weight )   # Adjust this blending based on your preference

    return overlaid_image
