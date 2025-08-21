"""
Morphological Operators

This module provides low-level morphological operations for image processing,
including basic morphological operations (erosion, dilation, opening, closing),
geodesic operations, morphological reconstruction, and specialized functions
for propagation analysis and connected component labeling.
"""

import numpy
import scipy
import skimage


def erosion(image, element=None):
    """
    Perform morphological erosion on binary or grayscale image.
    
    Erosion shrinks bright regions and enlarges dark regions. For binary images,
    it removes pixels from object boundaries.
    
    Args:
        image (numpy.ndarray): Input binary or grayscale image
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Eroded image
    """
    return skimage.morphology.erosion(image, element)


def dilation(image, element=None):
    """
    Perform morphological dilation on binary or grayscale image.
    
    Dilation enlarges bright regions and shrinks dark regions. For binary images,
    it adds pixels to object boundaries.
    
    Args:
        image (numpy.ndarray): Input binary or grayscale image
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Dilated image
    """
    return skimage.morphology.dilation(image, element)


def opening(image, element=None):
    """
    Perform morphological opening (erosion followed by dilation).
    
    Opening removes small bright objects and noise while preserving the shape
    and size of larger objects. It's useful for noise removal and separation
    of connected objects.
    
    Args:
        image (numpy.ndarray): Input binary or grayscale image
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Opened image
    """
    return skimage.morphology.opening(image, element)


def closing(image, element=None):
    """
    Perform morphological closing (dilation followed by erosion).
    
    Closing fills small holes and gaps in objects while preserving their shape
    and size. It's useful for connecting nearby objects and filling holes.
    
    Args:
        image (numpy.ndarray): Input binary or grayscale image
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Closed image
    """
    return skimage.morphology.closing(image, element)


def geodesic_erosion(marker_image, mask_image, element=None):
    """
    Perform geodesic erosion constrained by a mask.
    
    Geodesic erosion performs erosion on the marker image while ensuring
    the result remains above (or equal to) the mask image at each pixel.
    
    Args:
        marker_image (numpy.ndarray): Marker image to be eroded
        mask_image (numpy.ndarray): Mask image providing lower bound
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Geodesically eroded image
    """
    image = erosion(marker_image, element)
    return numpy.maximum(mask_image, image)


def geodesic_dilation(marker_image, mask_image, element=None):
    """
    Perform geodesic dilation constrained by a mask.
    
    Geodesic dilation performs dilation on the marker image while ensuring
    the result remains below (or equal to) the mask image at each pixel.
    
    Args:
        marker_image (numpy.ndarray): Marker image to be dilated
        mask_image (numpy.ndarray): Mask image providing upper bound
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Geodesically dilated image
    """
    image = dilation(marker_image, element)
    return numpy.minimum(mask_image, image)


def reconstruction_by_erosion(marker_image, mask_image, element=None):
    """
    Perform morphological reconstruction by erosion.
    
    Reconstruction by erosion iteratively applies geodesic erosion until
    convergence, effectively "reconstructing" the marker image within the
    constraints of the mask image using erosion operations.
    
    Args:
        marker_image (numpy.ndarray): Starting marker image
        mask_image (numpy.ndarray): Mask image constraining reconstruction
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Reconstructed image
    """
    method = 'erosion'
    image = skimage.morphology.reconstruction(
        marker_image, mask_image, method, element)
    dtype = marker_image.dtype
    return numpy.astype(image, dtype)


def reconstruction_by_dilation(marker_image, mask_image, element=None):
    """
    Perform morphological reconstruction by dilation.
    
    Reconstruction by dilation iteratively applies geodesic dilation until
    convergence, effectively "reconstructing" the marker image within the
    constraints of the mask image using dilation operations.
    
    Args:
        marker_image (numpy.ndarray): Starting marker image
        mask_image (numpy.ndarray): Mask image constraining reconstruction
        element (numpy.ndarray, optional): Structuring element. If None, uses default
        
    Returns:
        numpy.ndarray: Reconstructed image
    """
    method = 'dilation'
    image = skimage.morphology.reconstruction(
        marker_image, mask_image, method, element)
    dtype = marker_image.dtype
    return numpy.astype(image, dtype)


def propagation_function(image, element=None):
    """
    Compute propagation distances within connected components.
    
    For each pixel in a connected component, this function computes the minimum
    number of geodesic dilation steps required to reach that pixel from the
    component boundary. This creates a distance-like function that measures
    how "deep" each pixel is within its connected component.
    
    Args:
        image (numpy.ndarray): Binary or labeled image with connected components
        element (numpy.ndarray, optional): Structuring element for dilation. If None, uses default
        
    Returns:
        numpy.ndarray: Propagation distance image where each pixel value represents
                      the minimum steps from component boundary
    """
    points = numpy.where(image)
    propagation = numpy.zeros_like(image)
    for x, y in zip(*points):
        dilated = numpy.zeros_like(image)
        dilated[x, y] = 1
        reconstructed = reconstruction_by_dilation(dilated, image, element)
        i = 0
        while not numpy.array_equal(dilated, reconstructed):
            dilated = geodesic_dilation(dilated, image, element)
            i += 1
        propagation[x, y] = i
    return propagation


def labeling(image, element=None):
    """
    Label connected components in binary image.
    
    Assigns unique integer labels to each connected component in a binary image.
    Connected components are determined by the specified connectivity structure.
    
    Args:
        image (numpy.ndarray): Binary input image
        element (numpy.ndarray, optional): Connectivity structure. If None, uses default
        
    Returns:
        numpy.ndarray: Labeled image where each connected component has a unique integer label
    """
    image, _ = scipy.ndimage.label(image, element)
    return image
