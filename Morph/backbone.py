"""
Morphological Analysis Pipeline Backbone

This module provides the core processing pipeline that chains together various
morphological analysis steps in a configurable sequence. The backbone function
orchestrates the entire analysis workflow from data mapping through labeling.
"""

import Morph.modules

def backbone(data, mapper, counter, muxer, morphological_filter, thresholder, algebraic_filter, labeler):
    """
    Execute the complete morphological analysis pipeline.
    
    This function processes input data through a series of configurable steps:
    1. Mapping: Transform coordinate systems
    2. Counting: Aggregate data points
    3. Muxing: Combine multiple data streams
    4. Morphological filtering: Apply morphological operations
    5. Thresholding: Apply intensity thresholds
    6. Algebraic filtering: Apply algebraic filters
    7. Labeling: Assign labels to connected components
    
    Args:
        data (dict): Input spatial data containing coordinates and identifiers
        mapper (list): [method_name, args] for coordinate mapping
        counter (list): [method_name, args] for data counting/aggregation
        muxer (list): [method_name, args] for data multiplexing
        morphological_filter (list): [method_name, args] for morphological operations
        thresholder (list): [method_name, args] for intensity thresholding
        algebraic_filter (list): [method_name, args] for algebraic filtering
        labeler (list): [method_name, args] for connected component labeling
        
    Returns:
        numpy.ndarray: Labeled image with morphologically processed regions
        
    Example:
        >>> result = backbone(
        ...     data,
        ...     ['visium'],
        ...     ['total', gene_set],
        ...     ['maximum'],
        ...     ['opening', structuring_element],
        ...     ['binary', threshold_value],
        ...     ['area_opening', min_area],
        ...     ['blob', connectivity]
        ... )
    """
    data = getattr(Morph.modules.Mapper(), mapper[0])(data, *(mapper[1:]))
    data = getattr(Morph.modules.Counter(), counter[0])(data, *(counter[1:]))
    data = getattr(Morph.modules.Muxer(), muxer[0])(data, *(muxer[1:]))
    data = getattr(Morph.modules.MorphologicalFilter(), morphological_filter[0])(data, *(morphological_filter[1:]))
    data = getattr(Morph.modules.Thresholder(), thresholder[0])(data, *(thresholder[1:]))
    data = getattr(Morph.modules.AlgebraicFilter(), algebraic_filter[0])(data, *(algebraic_filter[1:]))
    return getattr(Morph.modules.Labeler(), labeler[0])(data, *(labeler[1:]))
