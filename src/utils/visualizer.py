import matplotlib.pyplot as plt
import cv2
import numpy as np 

class Visualizer():
  def __init__(self, skeleton_graph, limit = 5):
    self.skeleton_graph = skeleton_graph
    self.fig_size = (35,5)
    self.limit = limit
  def image_batch_show(self,image_batch):
    limit = min(len(image_batch), self.limit)
    image_batch = np.clip(image_batch[:limit],0,1)
    fig, axs = plt.subplots(1, len(image_batch), figsize = self.fig_size)
    for i, image in enumerate(image_batch):
      axs[i].imshow(image)
    plt.show()

  def check_data_batches(self, data_batch, image_batch):
      def print_error(message):
          print(f"Error: {message}")

      #if not isinstance(data_batch, (list, tuple)):
      #    print_error("Data must be a list or tuple.")
      #    return False
      if data_batch.shape[0] != image_batch.shape[0]:
        print_error("Batches count data and images is not equal")
        return False

      if not np.any(np.asarray(data_batch)):
        print_error("Data is empty.")
        return False


      if len(data_batch) != len(image_batch):
          print_error("Length mismatch: Data and image batch must have the same length.")
          return False

      return True

  def landmark_batch_image_joints_show(self, image_batch, landmark_batch):
    try:
      limit = min(len(image_batch), self.limit)

      num_landmarks = len(landmark_batch[0])
      image_batch = np.clip(image_batch[:limit], 0, 1)
      landmark_batch = landmark_batch[:limit]

      if not self.check_data_batches(landmark_batch, image_batch):
              return

      fig, axs = plt.subplots(1, len(image_batch), figsize = self.fig_size)
       # Check if axs is not iterable and convert it to a list
      if not isinstance(axs, (list, np.ndarray)):
          axs = np.array([axs])

      for i, (image, landmarks) in enumerate(zip(image_batch, landmark_batch)):
        image = np.clip(image, 0, 1)
        if len(landmarks) != num_landmarks:
                  print("Length mismatch: Landmarks and num_landmarks must have the same length.")
                  continue


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
        #cbar = fig.colorbar(im, ax=axs[i], shrink = 1.0)

      plt.show()
    except Exception as e:
      print(f"An error occurred: {e}")
      print(f"Shape of image_batch: {np.shape(image_batch)}")
      print(f"Shape of heatmap_batch: {np.shape(landmark_batch)}")


  def heatmap_batch_vstacked_show(self,heatmap_batch):
    try:
      fig, axs = plt.subplots(1, len(heatmap_batch), figsize = (25, 35))

      # Check if axs is not iterable and convert it to a list
      if not isinstance(axs, (list, np.ndarray)):
          axs = np.array([axs])

      heatmaps = None
      for i, heatmap in enumerate(heatmap_batch):
        heatmaps = np.transpose(heatmap, (2,0,1))
        axs[i].imshow(np.vstack(heatmaps))
      plt.show()
    except Exception as e:
      print(f"An error occurred: {e}")
      print(f"Shape of heatmap_batch: {np.shape(heatmap_batch)}")


  def heatmap_batch_onimage_show(self,image_batch, heatmap_batch,  background_weight = 0.3, heatmap_weight = 0.8, show_cbar = False):
    #if not self.check_data_batches(heatmap_batch, image_batch):
    #  return
    limit = min(len(image_batch), self.limit)

    image_batch = np.clip(image_batch[:limit], 0, 1)
    heatmap_batch = np.clip(heatmap_batch[:limit], 0, 1)

    try:
      fig, axs = plt.subplots(1, len(heatmap_batch), figsize = self.fig_size)

      # Check if axs is not iterable and convert it to a list
      if not isinstance(axs, (list, np.ndarray)):
          axs = np.array([axs])

      heatmaps = None
      for i, (image, heatmap) in enumerate(zip(image_batch, heatmap_batch)):
          bimg = image.copy()
          composed_image = self.draw_heatmap_on_image(bimg, heatmap.copy(), background_weight = background_weight, heatmap_weight = heatmap_weight)
          im = axs[i].imshow(
            composed_image,
          )
          if (show_cbar):
            cbar = fig.colorbar(im, ax=axs[i], shrink = 1.0)


      plt.show()
    except Exception as e:
      print(f"An error occurred: {e}")
      print(f"Shape of image_batch: {np.shape(image_batch)}")
      print(f"Shape of heatmap_batch: {np.shape(heatmap_batch)}")

  def draw_heatmap_on_image(self, img, heatmap, background_weight = 0.3, heatmap_weight = 0.8):
    if (img.shape[-2] != heatmap.shape[-2] ):
      #print("Image shape:", img.shape)
      #print("Heatmap shape:", heatmap.shape)
      heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))


     # Sum across channels
    summed_heatmap = np.sum(heatmap, axis=-1, keepdims=True)
    summed_heatmap = np.tile(summed_heatmap,3)

    # Normalize to the range [0, 1]
    normalized_heatmap = (summed_heatmap - np.min(summed_heatmap)) / (np.max(summed_heatmap) - np.min(summed_heatmap))

    # Overlay the heatmap on the image
    overlaid_image = (img  * background_weight + normalized_heatmap * heatmap_weight )   # Adjust this blending based on your preference
    overlaid_image = np.clip(overlaid_image, 0, 1)
    #print(overlaid_image.min(), overlaid_image.max())
    return overlaid_image
