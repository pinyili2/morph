"""
Morph Package

A comprehensive morphological analysis package for spatial data processing and feature extraction.
This package provides tools for reading spatial transcriptomics data, performing morphological
operations, extracting features, and writing results.

Modules:
    backbone: Core processing pipeline for morphological analysis
    modules: Data processing modules including mapping, filtering, and labeling
    readers: Data input functions for various spatial transcriptomics formats
    writers: Data output functions for saving analysis results
    features: Feature extraction methods for morphological analysis
    operators: Low-level morphological operations

The package is designed to work with spatial transcriptomics data from platforms
like Xenium and Visium, providing a complete workflow from data ingestion to
feature extraction and result export.
"""

from Morph.backbone import backbone
import Morph.modules
import Morph.readers
import Morph.writers
