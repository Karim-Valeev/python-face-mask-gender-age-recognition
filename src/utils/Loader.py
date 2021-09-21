import os
import cv2


class Loader:
    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            print(filename)
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        return images
