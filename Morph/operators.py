import skimage


def erosion(image, element=None):
    return skimage.morphology.erosion(image, element)
