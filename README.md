# Prediction of Individual Brain Ages from 3D MRI Datasets

This is an independent project by Zach Lyu.

## Abstract

Index terms: deep learning; brain age estimation; MRI.

Deep learning can accurately predict healthy individuals’ chronological age from T1-weighted MRI brain images. By feeding novel data into the model, the resulting bio-marker, termed brain age, has the potential to help investigate brain maturation and degeneration, as well as detect brain diseases in early phases. In this project, in order to evaluate the robustness of the brain age estimation system, four Convolutional Neural Network models were developed using different datasets. The training efficiency of the model is excellent compared to previous attempts.

Images were selected from an archive containing 2072 samples to form four datasets. By training separately on the datasets, the model’s performance with or without rotated images, with or without non-brain tissue interference, with or without dropout during testing, and with different data quality was compared. Input data were raw MRI images from individuals without known brain diseases with the mean image of the training data subtracted.

With the best performance model, the correlation between brain age and chronological age is evaluated as r=0.92, RMSE=9.66 years. With a smaller dataset, the model achieved 0.96 correlation. The model is capable of dealing with images with different orientations but performs poorly when non-brain tissues are involved. Removing dropout during testing phase shrinks the estimated chronological age, with strong correlation intact, suggesting the estimated value is a sum of several indexes. The image quality of 65×65×55 is enough for the sound performance of the model.

By integrating with deep learning techniques such as feature visualization and inversion, brain age has the potential to deepen our understanding of brain development processes. Furthermore, data from individuals with known brain diseases can turn the biomarker into a valid tool for clinical evaluation.

## Detailed Explanation

For detailed information please refer to the [report.pdf](https://github.com/bijiuni/brain_age/blob/master/Lv%20Ruyi%20Final%20Report.pdf) file

The table of content is shown below

```
Abstract
1. Introduction
  1.1 Need for Accurate Brain Age Estimation
    1.1.1 Detection of Diseases
    1.1.2 Neural Degeneration Evaluation
    1.1.3 Bio-marker Reliability and Time Efficiency
  1.2 Previous Attempts
  1.3 Introduction to Deep Learning
  1.4 Introduction to Convolutional Neural Network
2. Methodology
  2.1 Deep Learning
    2.1.1 The Neural Viewpoint and Backpropagation
    2.1.2 Cost Function
    2.1.3 Optimization
    2.1.4 Activation Functions and Weights Initialization
    2.1.5 Regularization
    2.1.6 3D Convolution and Regression
    2.1.7 CNN Structures
  2.2 Data Acquisition
  2.3 Data Preprocessing
  2.4 Software and Hardware
  2.5 Evaluation of Performance
3. Results
  3.1 Data Visualization
  3.2 Best Performance Model
  3.3 Image Rotation
  3.4 Non-brain Tissue
  3.5 Dropout Interference
  3.6 Sample Size
  3.7 Overfitting Alleviation and Learning rate
  3.8 Post statistical Test
4. Discussion
  4.1 Rotation and Non-brain Tissue: Robustness of the Model
  4.2 Dropout: Clues for Brain Age Estimation
  4.3 Limitations
  4.4 Future Directions
    4.4.1 Feature Visualization and Inversion
    4.4.2 Disease Detection
5. Conclusion
```

### Prerequisites

All codes were written in Python (tensorflow). Due to the high requirement of computational ability of deep learning, the training part of the project was run on google cloud platform. The following versions of the software were used: tensorflow-gpu-1.2.0, CUDA 8.0, cuDNN 5.1.

The hardware is extremely important for the speed of training. Since one of the main objectives of this project is to develop a high training speed model, the hardware was carefully chosen. 8 vCPUs were used with 52 G memory. The GPU was NVIDIA Tesla P100 in the operating system Ubuntu 16.04 LTS.

## Running the tests

The explanations are divided into data track and coding track

### Data Explanation

The raw data is connected through International Data Sharing Initiatives and OASIS brain.
You can get access to the raw data here:
[INDI](http://fcon_1000.projects.nitrc.org/indi/summerofsharing2012.html)
[OASIS](https://www.oasis-brains.org/)

In raw data is in nifti or Analyze format
After some preprocessing, the data are (more details in the report):
```
1: data656555noxrotate
2: data656555no|x|
3: data656555b16regression
4: data130130110no|x|
```


### Source files explanation

[Preliminary Preprocess](https://github.com/bijiuni/brain_age/tree/master/Codes/Preliminary%20Preprocess) (for different repositories): output numpy files

[preprocess.py](https://github.com/bijiuni/brain_age/blob/master/Codes/preprocess.py) (visualize.py): rotate, resample, crop, pad, shuffle, combine into batches

[setupcheck.ipynb](https://github.com/bijiuni/brain_age/blob/master/Codes/setupcheck.ipynb): check the setup of ipython tensorflow

[shell.ipynb](https://github.com/bijiuni/brain_age/blob/master/Codes/shell.ipynb): check GPU realtime usage

[regression_model.ipynb](https://github.com/bijiuni/brain_age/blob/master/Codes/regression_model.ipynb): train the neural network and save the results

[regression_result.ipynb](https://github.com/bijiuni/brain_age/blob/master/Codes/regression_restore.ipynb): restore and input new data (plot the results)
<br>

## Methodology

The final structure selected is shown below. It is a few repetitions of the combination of a 3×3×3 convolutional layer and a 2×2×2 max-pooling layer. After each convolutional layer, a ReLU function is implemented and a dropout layer with a dropout rate of 0.5-0.9 ensues. The final output is a scalar calculated using mean square error. Two types of sample size were selected: 65×65×55 or 130×130×110.

![CNN Structure](https://github.com/bijiuni/brain_age/blob/master/img/structure.JPG)

All MRI brain imaging data is collected from four repositories: Autism Brain Imaging Data
Exchange (ABIDE), Autism Brain Imaging Data Exchange Second Phase (ABIDEII), Open
Access Series of Imaging Studies (OASIS), and IXI. All data was approved by the sharing
scheme of international neuroimaging data-sharing initiative (INDI) or OASIS site. 

The raw data is in ANALYZE 7.5 format or NIFTI format. The mprage images of the subjects were selected and transformed into numpy array format for further processing. All images fed into the model are T1-weighted. As the assumption of the model is that healthy individuals’ chronological ages are close to their brain ages, all brain images from individuals with autism and Alzheimer’s disease were excluded. These images, however, may be helpful in further analyzing the model’s efficiency in detecting brain-related diseases. The data information is shown below.

![Repositories](https://github.com/bijiuni/brain_age/blob/master/img/repository.JPG)
<br>

## Sample Input and Results

![Sample Input](https://github.com/bijiuni/brain_age/blob/master/img/sample_input.JPG)
![Sample Result 1](https://github.com/bijiuni/brain_age/blob/master/img/sample_result1.JPG)
![Sample Result 2](https://github.com/bijiuni/brain_age/blob/master/img/sample_result2.JPG)

## Built With

* [Tensorflow](https://www.tensorflow.org/) - An open source machine learning framework
* [Numpy](http://www.numpy.org/) - fundamental package for scientific computing
* [Pandas](https://pandas.pydata.org/) - data structures and data analysis tools


## Authors

* **Zach Lyu** -  [bijiuni](https://github.com/bijiuni)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

Raw MRI brain data in this project is licensed under the [INDI(International Neuroimaging Data-sharing Intiative)](http://fcon_1000.projects.nitrc.org/) sharing scheme and [OASIS](https://www.oasis-brains.org/) sharing scheme.

## Acknowledgments

* Prof. Ed Wu for advising
