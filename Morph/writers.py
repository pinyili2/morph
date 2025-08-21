"""
Data Writing Module

This module provides functions to write morphological analysis results to files,
particularly for spatial transcriptomics platforms like Xenium. It supports
writing cell group assignments and feature measurements to CSV format.
"""

import csv


def xenium(file, image, data):
    """
    Write cell group assignments to CSV file in Xenium format.
    
    This function writes cell identifiers and their corresponding group assignments
    based on image segmentation results to a CSV file compatible with Xenium data format.
    
    Args:
        file (str): Path to output CSV file
        image (numpy.ndarray): Segmented image where pixel values represent group assignments
        data (dict): Dictionary containing cell data with keys:
                    - 'g': list of cell identifiers
                    - 'x': array of x-coordinates
                    - 'y': array of y-coordinates
    """
    with open(file, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=['cell_id', 'group'])
        dict_writer.writeheader()
        for g, x, y in zip(data['g'], data['x'], data['y']):
            dict_writer.writerow({'cell_id': g, 'group': image[x, y]})


def xenium_dict(file, feature):
    """
    Write feature measurements to CSV file in dictionary format.
    
    This function writes group-feature pairs to a CSV file, where each row
    contains a group identifier and its corresponding feature measurement.
    
    Args:
        file (str): Path to output CSV file
        feature (dict): Dictionary mapping group identifiers to feature values
    """
    with open(file, 'w') as f:
        dict_writer = csv.DictWriter(f, fieldnames=['group', 'feature'])
        dict_writer.writeheader()
        for f in feature:
            dict_writer.writerow({'group': f, 'feature': feature[f]})
