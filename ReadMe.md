# MeatPhenotype
## General
This repository has the code for MetaPhenotype 
## Abstract
Single-cell mass spectrometry (SCMS) is an emerging tool for studying cell heterogeneity according to the variation of molecular species in single cells. Although it has become increasingly common to employ machine learning models in SCMS data analysis, such as the classification of cell phenotype, the existing machine learning models often suffer from low adaptability and transferability. In addition, SCMS studies of rare cells can be restricted by limited cell samples. To overcome these limitations, here we proposed a meta-learning-based model, MetaPhenotype, for classifying cell phenotypes using pairs of primary and metastatic melanoma cancer cell lines. Our results show that, compared with standard transfer learning models, MetaPhenotype can more rapidly achieve a high accuracy of over 90% with fewer new training samples. Overall, our work opens the possibility of accurate cell phenotype classification based on fewer SCMS samples, thus lowering the demand for training sample acquisition.
## Instructions for runing MetaPhenotype
### Input Data
The input data were included in the Input data folder which named dataset.csv

### Running the model
Using our dataset to run the MetaPhenotype model.
```
python base.py
```
dataloade.py file will read dataset.csv file.
