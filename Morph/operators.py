import skimage


def erosion(image, element=None):
    return skimage.morphology.erosion(image, element)


def dilation(image, element=None):
    return skimage.morphology.dilation(image, element)
