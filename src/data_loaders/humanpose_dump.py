from .humanpose.py import DataSequence

class OnlyOneDataSequence(DataSequence):
    def __init__(self, images_folder, label_file,   batch_size = 8, input_size = (128, 128), heatmap_size = (64, 64), heatmap_sigma = 4, shuffle=True, augment=False, random_flip=False, clip_landmark = False, symmetry_point_ids = None ):
      super().__init__(images_folder, label_file,   batch_size = batch_size, input_size = input_size, heatmap_size = heatmap_size, heatmap_sigma = heatmap_sigma, shuffle=True, augment = augment, random_flip = random_flip, clip_landmark = clip_landmark, symmetry_point_ids = symmetry_point_ids )


    def _load_batch_data_from_anno(self, idx):
      return [self.anno[i] for i in range(self.batch_size)]
