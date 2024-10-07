# Text Generation and Classification

This repository contains code for generating text using various models and classifying the generated texts using a BERT-based classifier.

## Files and Directories

- **Generate_Data**: This file contains code for generating 2000 texts for each of the 8 models.
- **classifier_bert.py**: This script is used to train a classifier based on the BERT model for identifying which model generated the text.
- **test_bert.py**: This script is used to test the trained BERT model on a test dataset to evaluate its performance.

## Instructions

1. **Generate Data**: Run the `Generate_Data` script to generate the text data.
2. **Train Classifier**: Use `classifier_bert.py` to train the classifier on the generated data.
3. **Test Classifier**: After training, use `test_bert.py` to test the performance of the classifier.

## Requirements

To run the scripts, ensure you have the following libraries installed:

- Transformers
- PyTorch
- Scikit-learn
- Numpy
- Pandas
