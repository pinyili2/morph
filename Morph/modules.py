"""
Data Processing Modules

This module provides a collection of processing classes for spatial transcriptomics data,
including coordinate mapping, data counting, image multiplexing, morphological filtering,
thresholding, algebraic filtering, and labeling operations. Each class supports multiple
methods and custom implementations.
"""

import numpy
import skimage

import Morph.operators


def _floor(x):
    """
    Convert input to integer type (floor operation).
    
    Args:
        x: Input value or array
        
    Returns:
        Integer version of input
    """
    return numpy.astype(x, int)


def _count(data, G, method):
    """
    Count data points for each gene/feature at spatial locations.
    
    Args:
        data (dict): Spatial data with 'g' (genes), 'x', 'y' coordinates
        G (set): Set of gene/feature identifiers to process
        method (str): Counting method ('naive' removes genes after first count)
        
    Returns:
        dict: Dictionary mapping gene IDs to 2D count arrays
    """
    G = G & set(data['g'])
    image = {}
    X = max(data['x'])
    Y = max(data['y'])
    shape = (X + 1, Y + 1)
    for g in G:
        image[g] = numpy.zeros(shape, int)
    for g, x, y in zip(data['g'], data['x'], data['y']):
        if g in G:
            image[g][x, y] += 1
            if method == 'naive':
                G.remove(g)
    return image


def _point_wise_maximum(image):
    """
    Compute element-wise maximum across multiple images.
    
    Args:
        image (dict): Dictionary of 2D arrays
        
    Returns:
        numpy.ndarray: Element-wise maximum of all input images
    """
    array = [image[i] for i in image]
    return numpy.maximum.reduce(array)


def _area_opening(image, area_threshold):
    """
    Remove connected components smaller than area threshold.
    
    Args:
        image (numpy.ndarray): Input binary or labeled image
        area_threshold (int): Minimum area for component retention
        
    Returns:
        numpy.ndarray: Filtered image with small components removed
    """
    return skimage.morphology.area_opening(image, area_threshold)


def _area_closing(image, area_threshold):
    """
    Fill holes smaller than area threshold in connected components.
    
    Args:
        image (numpy.ndarray): Input binary or labeled image
        area_threshold (int): Maximum area for hole filling
        
    Returns:
        numpy.ndarray: Filtered image with small holes filled
    """
    return skimage.morphology.area_closing(image, area_threshold)


class Mapper:
    """
    Coordinate mapping methods for different spatial transcriptomics platforms.
    
    This class provides methods to transform coordinates from platform-specific
    formats to standardized grids suitable for image analysis.
    """
    
    def naive(self, data):
        """
        Pass-through mapping with no coordinate transformation.
        
        Args:
            data (dict): Input spatial data
            
        Returns:
            dict: Unmodified input data
        """
        return data

    def visium(self, data):
        """
        Map Visium hexagonal coordinates to rectangular grid.
        
        Visium uses a hexagonal spot arrangement. This method transforms
        the coordinates to a rectangular grid for standard image processing.
        
        Args:
            data (dict): Input data with 'g', 'x', 'y' keys
            
        Returns:
            dict: Data with transformed coordinates
        """
        x = (data['x'] + data['y']) // 2
        return {'g': data['g'], 'x': x, 'y': data['y']}

    def xenium(self, data, d):
        """
        Bin Xenium sub-cellular coordinates to regular grid.
        
        Xenium provides sub-cellular resolution coordinates. This method
        bins them to a regular grid with spacing d.
        
        Args:
            data (dict): Input data with 'g', 'x', 'y' keys
            d (float): Grid spacing for binning
            
        Returns:
            dict: Data with binned coordinates
        """
        x = _floor(data['x'] / d)
        y = _floor(data['y'] / d)
        return {'g': data['g'], 'x': x, 'y': y}

    def custom(self, data, mapper, *args):
        """
        Apply custom coordinate mapping function.
        
        Args:
            data (dict): Input spatial data
            mapper (callable): Custom mapping function
            *args: Additional arguments for mapper function
            
        Returns:
            dict: Mapped data as returned by mapper function
        """
        return mapper(data, *args)


class Counter:
    """
    Data counting and aggregation methods for spatial transcriptomics data.
    
    This class provides methods to count and aggregate molecular detections
    at spatial locations for specified gene sets.
    """
    
    def naive(self, data, G):
        """
        Count each gene only once per location (first occurrence).
        
        Args:
            data (dict): Spatial data with gene identifiers and coordinates
            G (set): Set of gene identifiers to count
            
        Returns:
            dict: Dictionary mapping gene IDs to 2D count arrays
        """
        return _count(data, G, Counter.naive.__name__)

    def total(self, data, G):
        """
        Count all occurrences of each gene at each location.
        
        Args:
            data (dict): Spatial data with gene identifiers and coordinates
            G (set): Set of gene identifiers to count
            
        Returns:
            dict: Dictionary mapping gene IDs to 2D count arrays
        """
        return _count(data, G, Counter.total.__name__)

    def custom(self, data, counter, *args):
        """
        Apply custom counting function.
        
        Args:
            data (dict): Input spatial data
            counter (callable): Custom counting function
            *args: Additional arguments for counter function
            
        Returns:
            Result of custom counter function
        """
        return counter(data, *args)


class Muxer:
    """
    Image multiplexing methods for combining multiple gene expression images.
    
    This class provides methods to combine multiple 2D arrays (gene expression
    images) into a single composite image.
    """
    
    def naive(self, image):
        """
        Return arbitrary single image from the collection.
        
        Args:
            image (dict): Dictionary of 2D arrays
            
        Returns:
            numpy.ndarray: Single arbitrarily selected image
        """
        _, image = image.popitem()
        return image

    def maximum(self, image):
        """
        Compute element-wise maximum across all images.
        
        Args:
            image (dict): Dictionary of 2D arrays
            
        Returns:
            numpy.ndarray: Element-wise maximum of all input images
        """
        return _point_wise_maximum(image)

    def custom(self, image, muxer, *args):
        """
        Apply custom multiplexing function.
        
        Args:
            image (dict): Dictionary of 2D arrays
            muxer (callable): Custom multiplexing function
            *args: Additional arguments for muxer function
            
        Returns:
            Result of custom muxer function
        """
        return muxer(image, *args)


class MorphologicalFilter:
    """
    Morphological filtering operations for image processing.
    
    This class provides standard morphological operations including opening,
    closing, and their combinations for noise reduction and shape enhancement.
    """
    
    def naive(self, image):
        """
        Pass-through with no morphological filtering.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Unmodified input image
        """
        return image

    def opening(self, image, element):
        """
        Apply morphological opening (erosion followed by dilation).
        
        Opening removes small bright objects and noise while preserving
        the overall shape and size of larger objects.
        
        Args:
            image (numpy.ndarray): Input binary or grayscale image
            element (numpy.ndarray): Structuring element
            
        Returns:
            numpy.ndarray: Opened image
        """
        return Morph.operators.opening(image, element)

    def closing(self, image, element):
        """
        Apply morphological closing (dilation followed by erosion).
        
        Closing fills small holes and gaps while preserving the overall
        shape and size of objects.
        
        Args:
            image (numpy.ndarray): Input binary or grayscale image
            element (numpy.ndarray): Structuring element
            
        Returns:
            numpy.ndarray: Closed image
        """
        return Morph.operators.closing(image, element)

    def open_close(self, image, element):
        """
        Apply opening followed by closing.
        
        This combination removes noise and fills holes in sequence.
        
        Args:
            image (numpy.ndarray): Input binary or grayscale image
            element (numpy.ndarray): Structuring element
            
        Returns:
            numpy.ndarray: Processed image
        """
        image = Morph.operators.opening(image, element)
        return Morph.operators.closing(image, element)

    def close_open(self, image, element):
        """
        Apply closing followed by opening.
        
        This combination fills holes and removes noise in sequence.
        
        Args:
            image (numpy.ndarray): Input binary or grayscale image
            element (numpy.ndarray): Structuring element
            
        Returns:
            numpy.ndarray: Processed image
        """
        image = Morph.operators.closing(image, element)
        return Morph.operators.opening(image, element)

    def custom(self, image, morphological_filter, *args):
        """
        Apply custom morphological filtering function.
        
        Args:
            image (numpy.ndarray): Input image
            morphological_filter (callable): Custom filtering function
            *args: Additional arguments for filter function
            
        Returns:
            Result of custom filter function
        """
        return morphological_filter(image, *args)


class Thresholder:
    """
    Intensity thresholding methods for image binarization.
    
    This class provides methods to convert grayscale images to binary
    images based on intensity thresholds.
    """
    
    def naive(self, image):
        """
        Pass-through with no thresholding.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Unmodified input image
        """
        return image

    def binary(self, image, tau):
        """
        Apply binary thresholding with specified threshold value.
        
        Pixels with values >= tau are set to 1, others to 0.
        
        Args:
            image (numpy.ndarray): Input grayscale image
            tau (float): Threshold value
            
        Returns:
            numpy.ndarray: Binary image (0s and 1s)
        """
        return (image >= tau) * 1

    def custom(self, image, thresholder, *args):
        """
        Apply custom thresholding function.
        
        Args:
            image (numpy.ndarray): Input image
            thresholder (callable): Custom thresholding function
            *args: Additional arguments for thresholder function
            
        Returns:
            Result of custom thresholder function
        """
        return thresholder(image, *args)


class AlgebraicFilter:
    """
    Algebraic filtering operations based on connected component properties.
    
    This class provides filtering methods that use geometric properties
    like area to remove or modify connected components.
    """
    
    def naive(self, image):
        """
        Pass-through with no algebraic filtering.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Unmodified input image
        """
        return image

    def area_opening(self, image, lambda_):
        """
        Remove connected components smaller than specified area.
        
        This is a connected component filter that removes objects
        with area less than the threshold lambda_val.
        
        Args:
            image (numpy.ndarray): Input binary or labeled image
            lambda_ (int): Minimum area threshold
            
        Returns:
            numpy.ndarray: Filtered image with small components removed
        """
        return _area_opening(image, lambda_)

    def area_closing(self, image, lambda_):
        """
        Fill holes smaller than specified area in connected components.
        
        This filter fills holes (background regions) that are smaller
        than the threshold lambda_val.
        
        Args:
            image (numpy.ndarray): Input binary or labeled image
            lambda_ (int): Maximum hole size to fill
            
        Returns:
            numpy.ndarray: Filtered image with small holes filled
        """
        return _area_closing(image, lambda_)

    def custom(self, image, algebraic_filter, *args):
        """
        Apply custom algebraic filtering function.
        
        Args:
            image (numpy.ndarray): Input image
            algebraic_filter (callable): Custom filtering function
            *args: Additional arguments for filter function
            
        Returns:
            Result of custom filter function
        """
        return algebraic_filter(image, *args)


class Labeler:
    """
    Connected component labeling methods for image segmentation.
    
    This class provides methods to assign unique labels to connected
    components in binary images.
    """
    
    def naive(self, image):
        """
        Pass-through with no labeling.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Unmodified input image
        """
        return image

    def blob(self, image, element):
        """
        Label connected components (blobs) in binary image.
        
        Assigns unique integer labels to each connected component
        using the specified connectivity structure.
        
        Args:
            image (numpy.ndarray): Input binary image
            element (numpy.ndarray): Connectivity structure
            
        Returns:
            numpy.ndarray: Labeled image with unique integer labels
        """
        return Morph.operators.labeling(image, element)

    def custom(self, image, labeler, *args):
        """
        Apply custom labeling function.
        
        Args:
            image (numpy.ndarray): Input image
            labeler (callable): Custom labeling function
            *args: Additional arguments for labeler function
            
        Returns:
            Result of custom labeler function
        """
        return labeler(image, *args)
