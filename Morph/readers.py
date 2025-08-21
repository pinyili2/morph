"""
Data Reading Module

This module provides functions to read spatial transcriptomics data from various
file formats, particularly those used by platforms like Xenium. It supports
reading both transcript-level data and cell-level data from compressed CSV files.
"""

import csv
import gzip
import numpy


def transcripts(file):
    """
    Read transcript-level spatial transcriptomics data from compressed CSV file.
    
    This function reads individual transcript detections with their spatial coordinates
    and gene identities from a gzip-compressed CSV file, typically used for
    transcript-level analysis in spatial transcriptomics.
    
    Args:
        file (str): Path to gzip-compressed CSV file containing transcript data.
                   Expected columns: 'feature_name', 'x_location', 'y_location'
                   
    Returns:
        dict: Dictionary containing:
            - 'g': list of gene/feature names for each transcript
            - 'x': numpy array of x-coordinates
            - 'y': numpy array of y-coordinates
            
    Example:
        >>> data = transcripts('transcripts.csv.gz')
        >>> print(f"Found {len(data['g'])} transcripts")
        >>> print(f"Gene types: {set(data['g'])}")
    """
    g = []
    x = []
    y = []
    with gzip.open(file, 'rt') as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            g.append(row['feature_name'])
            x.append(float(row['x_location']))
            y.append(float(row['y_location']))
    return {'g': g, 'x': numpy.array(x), 'y': numpy.array(y)}


def cells(file):
    """
    Read cell-level spatial transcriptomics data from compressed CSV file.
    
    This function reads cell centroid locations and identifiers from a gzip-compressed
    CSV file, typically used for cell-level analysis in spatial transcriptomics.
    
    Args:
        file (str): Path to gzip-compressed CSV file containing cell data.
                   Expected columns: 'cell_id', 'x_centroid', 'y_centroid'
                   
    Returns:
        dict: Dictionary containing:
            - 'g': list of cell identifiers  
            - 'x': numpy array of cell centroid x-coordinates
            - 'y': numpy array of cell centroid y-coordinates
            
    Example:
        >>> data = cells('cells.csv.gz')
        >>> print(f"Found {len(data['g'])} cells")
        >>> print(f"X range: {data['x'].min():.1f} - {data['x'].max():.1f}")
    """
    g = []
    x = []
    y = []
    with gzip.open(file, 'rt') as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            g.append(row['cell_id'])
            x.append(float(row['x_centroid']))
            y.append(float(row['y_centroid']))
    return {'g': g, 'x': numpy.array(x), 'y': numpy.array(y)}
