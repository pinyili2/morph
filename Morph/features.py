"""
Morphological Feature Extraction Module

This module provides feature extraction capabilities for morphological analysis of images,
including center detection, distance transforms, layer analysis, shape measurements, and size calculations.
The module is designed to work with labeled images and provides support for tissue masking and 
specialized methods like Visium spatial transcriptomics analysis.

Classes:
    Center: Methods for finding centers of objects using geodesic and ultimate erosion techniques
    Distance: Distance transform calculations with minimum and maximum distance features
    Layer: Layer-based morphological analysis with erosion-based layer counting
    Shape: Shape analysis including roundness calculations
    Size: Size measurements including object counting

The module includes various helper functions for image processing operations and supports
both standard morphological operations and specialized spatial analysis methods.
"""

import numpy
import scipy

import Morph.operators

PI = numpy.pi


def _unique(image):
    """
    Extract unique non-zero values from an image.
    
    Args:
        image (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Array of unique non-zero values from the image
    """
    ar = image[image != 0]
    return numpy.unique(ar)


def _minimum(input_, labels, index):
    """
    Find minimum values of input array for each labeled region.
    
    Args:
        input_ (numpy.ndarray): Input array to find minimums in
        labels (numpy.ndarray): Label array defining regions
        index (array-like): Labels for which to compute minimums
        
    Returns:
        numpy.ndarray: Minimum values for each specified label
    """
    return scipy.ndimage.minimum(input_, labels, index)


def _nonzero(a):
    """
    Return indices of non-zero elements.
    
    Args:
        a (numpy.ndarray): Input array
        
    Returns:
        tuple: Tuple of arrays containing indices of non-zero elements
    """
    return numpy.nonzero(a)


def _any(a):
    """
    Test whether any array element is True.
    
    Args:
        a (numpy.ndarray): Input array
        
    Returns:
        bool: True if any element is True, False otherwise
    """
    return numpy.any(a)


def _isin(element, test_elements):
    """
    Test whether each element of element is in test_elements.
    
    Args:
        element (numpy.ndarray): Input array
        test_elements (array-like): Values against which to test each element
        
    Returns:
        numpy.ndarray: Boolean array of same shape as element
    """
    return numpy.isin(element, test_elements)


def _padding_func(vector, iaxis_pad_width, iaxis, kwargs):
    """
    Custom padding function for numpy.pad operation.
    
    This function is used as a callback for numpy.pad to apply custom padding
    behavior, particularly for distance transform operations.
    
    Args:
        vector (numpy.ndarray): 1D array to be padded
        iaxis_pad_width (tuple): Tuple of (pad_before, pad_after) for current axis
        iaxis (int): Current axis being padded
        kwargs (dict): Additional keyword arguments containing 'pad' values
    """
    pad = kwargs.get('pad')
    vector[:iaxis_pad_width[0]] = pad[0]
    vector[-iaxis_pad_width[1]:] = pad[-1]
    pad[0] = 1 - pad[0]
    pad[-1] = 1 - pad[-1]


def _distance(image, sampling, border_value, tissue, method):
    """
    Compute distance transform with optional tissue masking and specialized methods.
    
    This function computes Euclidean distance transforms with support for tissue masking
    and specialized sampling methods like Visium spatial transcriptomics.
    
    Args:
        image (numpy.ndarray): Binary or labeled image for distance computation
        sampling (float or list): Pixel spacing for distance calculation
        border_value (int): Value to use for border/background pixels
        tissue (numpy.ndarray, optional): Binary tissue mask. If None, uses entire image
        method (str, optional): Special method for distance calculation ('visium' for 
                               Visium spatial transcriptomics hexagonal grid)
    
    Returns:
        numpy.ndarray: Distance transform of the input image
    """
    if tissue is None:
        tissue = numpy.ones_like(image)
    image[tissue == 0] = border_value
    if method == 'visium':
        shape = 2 * image.shape[0] - image.shape[1], image.shape[1]
        unmapped = numpy.ones(shape, bool)
        grid = numpy.indices(shape)
        checkerboard = 1 - sum(grid) % 2
        x, y = _nonzero(checkerboard)
        unmapped[x, y] = image[(x + y) // 2, y]
        image = unmapped
        sampling = [50, 50 * 3**0.5]
    pad_width = 1
    pad = [0, 1] if method == 'visium' and not border_value else [border_value]
    input_ = numpy.pad(image, pad_width, _padding_func, pad=pad)
    distances = scipy.ndimage.distance_transform_edt(input_, sampling)
    distances = distances[pad_width:-pad_width, pad_width:-pad_width]
    if method == 'visium':
        shape = sum(distances.shape) // 2, distances.shape[1]
        mapped = numpy.zeros(shape)
        mapped[(x + y) // 2, y] = distances[x, y]
        distances = mapped
    distances[tissue == 0] = 0
    return distances


def _layer(image, structure, border_value, tissue):
    """
    Compute morphological layers using iterative erosion.
    
    This function creates layers by iteratively eroding the image until no pixels remain,
    counting the number of erosion steps required to remove each pixel.
    
    Args:
        image (numpy.ndarray): Binary input image
        structure (numpy.ndarray, optional): Structuring element for erosion
        border_value (int): Value to use for border pixels during erosion
        tissue (numpy.ndarray, optional): Binary tissue mask. If None, uses entire image
    
    Returns:
        numpy.ndarray: Layer image where each pixel value represents the layer number
    """
    if tissue is None:
        tissue = numpy.ones_like(image)
    image[tissue == 0] = border_value
    layers = image * 1
    while _any(image):
        layers += scipy.ndimage.binary_erosion(image,
                                               structure,
                                               output=image,
                                               border_value=border_value)
    layers[tissue == 0] = 0
    return layers


def _count(image):
    """
    Count occurrences of each unique non-zero value in image.
    
    Args:
        image (numpy.ndarray): Input image with labeled regions
        
    Returns:
        tuple: (unique_values, counts) - arrays of unique non-zero values and their counts
    """
    x = image[image != 0]
    return numpy.unique_counts(x)


def _maximum(input_, labels, index):
    """
    Find maximum values of input array for each labeled region.
    
    Args:
        input_ (numpy.ndarray): Input array to find maximums in
        labels (numpy.ndarray): Label array defining regions
        index (array-like): Labels for which to compute maximums
        
    Returns:
        numpy.ndarray: Maximum values for each specified label
    """
    return scipy.ndimage.maximum(input_, labels, index)


class Center():
    """
    Center detection methods for morphological objects.
    
    This class provides methods to find centers of labeled objects using different
    morphological techniques including geodesic centers and ultimate erosion centers.
    """
    
    def geodesic(self, image, element=None):
        """
        Find geodesic centers of labeled objects.
        
        Geodesic centers are points within each object that have the minimum
        propagation distance from the object boundary.
        
        Args:
            image (numpy.ndarray): Labeled image with distinct objects
            element (numpy.ndarray, optional): Structuring element for morphological operations
            
        Returns:
            dict: Dictionary mapping label values to sets of center coordinates (y, x)
        """
        propagation = Morph.operators.propagation_function(image, element)
        index = _unique(image)
        minimum = _minimum(propagation, image, index)
        centers = {}
        for i, m in zip(index, minimum):
            points = _nonzero((image == i) & (propagation == m))
            centers[i] = set(zip(*points))
        return centers

    def ultimate(self, image, element=None):
        """
        Find ultimate erosion centers of labeled objects.
        
        Ultimate centers are points that survive the longest during iterative erosion
        with reconstruction, representing the "skeleton" or medial axis of objects.
        
        Args:
            image (numpy.ndarray): Labeled image with distinct objects
            element (numpy.ndarray, optional): Structuring element for morphological operations
            
        Returns:
            dict: Dictionary mapping label values to sets of center coordinates (y, x)
        """
        index = _unique(image)
        centers = {i: set() for i in index}
        while _any(image):
            eroded = Morph.operators.erosion(image, element)
            reconstructed = Morph.operators.reconstruction_by_dilation(
                eroded, image, element)
            points = _nonzero(image - reconstructed)
            for point in zip(*points):
                i = image[point]
                centers[i].add(point)
            image = eroded
        return centers


class Distance:
    """
    Distance transform analysis for morphological features.
    
    This class provides methods to compute signed distance transforms that measure
    distances to object boundaries, with support for tissue masking and specialized
    spatial analysis methods.
    """
    
    def minimum(self, image, index=None, tissue=None, d=1, method=None):
        """
        Compute minimum distance transform (distance from outside to object boundary).
        
        This method computes the signed distance transform where positive values
        represent distances outside objects and negative values represent distances
        inside objects.
        
        Args:
            image (numpy.ndarray): Labeled image or binary image
            index (array-like, optional): Specific label values to consider. 
                                        If None, uses all non-zero values
            tissue (numpy.ndarray, optional): Binary tissue mask to restrict analysis
            d (float): Pixel spacing for distance calculation
            method (str, optional): Special method ('visium' for spatial transcriptomics)
            
        Returns:
            numpy.ndarray: Signed distance transform image
        """
        image = image != 0 if index is None else _isin(image, list(index))
        distances = _distance(~image, d, 0, tissue, method)
        distances -= _distance(image, d, 1, tissue, method)
        return distances

    def maximum(self, image, index=None, tissue=None, d=1, method=None):
        """
        Compute maximum distance transform (distance from inside to object boundary).
        
        This method computes the signed distance transform where negative values
        represent distances outside objects and positive values represent distances
        inside objects.
        
        Args:
            image (numpy.ndarray): Labeled image or binary image
            index (array-like, optional): Specific label values to consider.
                                        If None, uses all non-zero values
            tissue (numpy.ndarray, optional): Binary tissue mask to restrict analysis
            d (float): Pixel spacing for distance calculation
            method (str, optional): Special method ('visium' for spatial transcriptomics)
            
        Returns:
            numpy.ndarray: Signed distance transform image
        """
        image = image != 0 if index is None else _isin(image, list(index))
        distances = _distance(~image, d, 1, tissue, method)
        distances -= _distance(image, d, 0, tissue, method)
        return distances


class Layer():
    """
    Morphological layer analysis using erosion-based techniques.
    
    This class provides methods to compute morphological layers that represent
    the distance from object boundaries measured in terms of erosion steps.
    """
    
    def minimum(self, image, index=None, tissue=None, element=None):
        """
        Compute minimum layer transform (layers from outside to object boundary).
        
        This method computes signed layer values where positive values represent
        layers outside objects and negative values represent layers inside objects.
        
        Args:
            image (numpy.ndarray): Labeled image or binary image
            index (array-like, optional): Specific label values to consider.
                                        If None, uses all non-zero values
            tissue (numpy.ndarray, optional): Binary tissue mask to restrict analysis
            element (numpy.ndarray, optional): Structuring element for erosion operations
            
        Returns:
            numpy.ndarray: Signed layer transform image
        """
        image = image != 0 if index is None else _isin(image, list(index))
        layers = _layer(~image, element, 0, tissue)
        layers -= _layer(image, element, 1, tissue)
        return layers

    def maximum(self, image, index=None, tissue=None, element=None):
        """
        Compute maximum layer transform (layers from inside to object boundary).
        
        This method computes signed layer values where negative values represent
        layers outside objects and positive values represent layers inside objects.
        
        Args:
            image (numpy.ndarray): Labeled image or binary image
            index (array-like, optional): Specific label values to consider.
                                        If None, uses all non-zero values
            tissue (numpy.ndarray, optional): Binary tissue mask to restrict analysis
            element (numpy.ndarray, optional): Structuring element for erosion operations
            
        Returns:
            numpy.ndarray: Signed layer transform image
        """
        image = image != 0 if index is None else _isin(image, list(index))
        layers = _layer(~image, element, 1, tissue)
        layers -= _layer(image, element, 0, tissue)
        return layers


class Shape():
    """
    Shape analysis methods for morphological objects.
    
    This class provides methods to compute shape descriptors and measurements
    for labeled objects in images.
    """
    
    def roundness(self, image, element=None):
        """
        Compute roundness measure for each labeled object.
        
        Roundness is calculated as 4 * area / (π * max_propagation²), where
        max_propagation is the maximum propagation distance within each object.
        A perfect circle has roundness = 1, while elongated shapes have values < 1.
        
        Args:
            image (numpy.ndarray): Labeled image with distinct objects
            element (numpy.ndarray, optional): Structuring element for propagation function
            
        Returns:
            dict: Dictionary mapping label values to roundness measurements
        """
        propagation = Morph.operators.propagation_function(image, element)
        index, size = _count(image)
        maximum = _maximum(propagation, image, index)
        shape = 4 * size / (PI * maximum**2)
        return dict(zip(index, shape))


class Size():
    """
    Size analysis methods for morphological objects.
    
    This class provides methods to compute size measurements and statistics
    for labeled objects in images.
    """
    
    def count(self, image):
        """
        Count the number of pixels in each labeled object.
        
        Args:
            image (numpy.ndarray): Labeled image with distinct objects
            
        Returns:
            dict: Dictionary mapping label values to pixel counts (areas)
        """
        index, size = _count(image)
        return dict(zip(index, size))
