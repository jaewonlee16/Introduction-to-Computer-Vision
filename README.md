# Introduction to Computer Vision 2024 Spring
Assignments from SNU ECE Introduction to Computer Vision (M2608.001900)

# Assignment #1
[Notebook](https://github.com/jaewonlee16/Introduction-to-Computer-Vision/blob/master/assignment1/assignment1.ipynb) / [Report](https://github.com/jaewonlee16/Introduction-to-Computer-Vision/blob/master/assignment1/Computer_Vision_assignment1.pdf)

## Overview

This assignment focuses on camera calibration, specifically determining the camera projection matrix using various methods.

## Tasks

1. **Camera Calibration**: 
   - Derive the camera projection matrix \( P \) using Singular Value Decomposition (SVD).
   - Implement an alternative method using the pseudo-inverse.

2. **Steps**:
   - **SVD Method**: Show that the solution \( p \) is the last column of \( V \) corresponding to the smallest singular value of \( A \).
   - **Pseudo-Inverse Method**: Determine \( P \) using the pseudo-inverse approach.

## Implementation

- **Language**: Python
- **Data**: Provided in the assignment
- **Output**: Camera projection matrix \( P \)

## References

- Tutorials on MATLAB and Image Processing Toolbox
- OpenCV and OpenCV-Python Tutorials


# Assignment #2
[Notebook](https://github.com/jaewonlee16/Introduction-to-Computer-Vision/blob/master/assignment2/GHT.ipynb) / [Report](https://github.com/jaewonlee16/Introduction-to-Computer-Vision/blob/master/assignment2/Computer_Vision_assignment2.pdf)

## Overview

This assignment focuses on deriving mathematical representations and implementing the Generalized Hough Transform (GHT) algorithm for detecting arbitrary shapes in images.

## Tasks

### 1. Math

#### a. Polar Representation of Lines
- **Task**: Derive the 2D polar line representation: \( \rho = x \cos \theta + y \sin \theta \).

#### b. Polar Representation of Planes
- **Task**: Derive the polar representation of planes in 3D.

### 2. Generalized Hough Transform (GHT)

- **Paper to Read**: D.H. Ballard, "Generalizing the Hough Transform to Detect Arbitrary Shapes", Pattern Recognition, Vol.13, No.2, p.111-122, 1981.

#### Implementation Steps:
1. **GHT Algorithm**: Write your own program for the Generalized Hough Transform algorithm.
2. **Edge Detection**: Convert both the template image and the target image into edge images using the OpenCV (Python) Canny edge detection algorithm.
3. **Detection Results**: Show your detection results by overlaying the edge template at the detected locations on the target image and analyze them.
4. **Invariance**: Make your GHT algorithm scale and rotation invariant, if possible, and show the results.

## Implementation

- **Language**: Python
- **Tools**: OpenCV (for edge detection and other image processing tasks)

## References
- [OpenCV](https://opencv.org/)
- [OpenCV-Python Tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html)



# Assignment #3
[Notebook](https://github.com/jaewonlee16/Introduction-to-Computer-Vision/blob/master/assignment3/Assignment3.ipynb) / [Report](https://github.com/jaewonlee16/Introduction-to-Computer-Vision/blob/master/assignment3/Computer_vision_assignment3.pdf)

## Overview

This assignment involves various tasks related to least square line fitting and image stitching using homography and RANSAC. The goal is to understand the concepts and implement the algorithms to achieve the tasks described below.


## Assignment Tasks

### 1. Least Square Line Fitting

#### a. Visualize Data Distribution
- Given \(n\) 2D points \( \mathbf{x}_i = \begin{pmatrix} x_i \\ y_i \end{pmatrix}, i = 1 \ldots n \), visualize the data distribution.

#### b. Calculate Second Moment and Eigenvalues
- Compute the second moment of the data.
- Calculate the eigenvalues and eigenvectors.

#### c. Geometric Interpretation
- Visualize the geometric interpretation of the eigenvalues and eigenvectors with respect to the data distribution.

#### d. Optimal Line Fitting
- Find the optimal line \( \ell: ax + by = d \) that fits the data in the total least square sense.
- Visualize the optimal line on the data distribution.

### 2. Another Interpretation of Least Square Line Fitting

#### a. Projection Line and Variance
- Interpret least-square fitting as finding the optimal projection line \( \ell \) that maximizes the variance of the projected samples of the original data.

#### b. Principal Component Analysis
- Show that the vector \( \mathbf{v} \) that maximizes the variance of the projected samples is the principal component.
- Establish the relationship between \( \Sigma \) and the second moment matrix \( \mathbf{U}^T \mathbf{U} \).
- Discuss the relationship between \( \mathbf{v} \) and the line normal vector \( \mathbf{n} \) obtained from the second moment matrix.

### 3. Homography and Image Stitching

#### a. Taking Panoramic Pictures
- Obtain three images (I1, I2, I3) of a scene from different views by rotating a smartphone camera.
- Ensure the same center of projection and adequate overlap between neighboring views.
- Resize images to 256x256 pixels.

#### b. Feature Extraction
- Convert color images to grayscale.
- Detect feature points using SIFT or its alternatives (OpenCV, Python, MATLAB versions).
- Ensure features are evenly distributed and avoid multiple detections at the same point.

#### c. Feature Matching
- Find putative matches between images \(I_i\) and \(I_{i+1}\) using the ratio of distances between the best and second-best matches.
- Display detected feature correspondences.

#### d. Homography Estimation using RANSAC
- Estimate the 3x3 Homography matrix \( H_{ij} \) using RANSAC and DLT methods.

##### Steps for RANSAC:
1. Compute interest points in each image.
2. Compute a set of interest point matches based on a similarity measure.
3. Repeat for N samples:
   - Select 4 correspondences and compute \( H \) using DLT.
   - Calculate the distance for each putative match.
   - Compute the number of inliers consistent with \( H \).
   - Choose \( H \) with the most inliers.
4. Optionally re-estimate \( H \) from all inliers.
5. Optionally iterate the last two steps until convergence.

##### Steps for DLT:
1. For each correspondence, compute \( A_i \).
2. Assemble into a single matrix \( A \).
3. Obtain SVD of \( A \). The solution is the last column of \( V \).

#### e. Warping Images
- Implement a warping function to transform images I1 and I3 to the center image I2 using homographies \( H_{12} \) and \( H_{32} \).

#### f. Creating the Mosaic
- Stitch the warped images to create a panoramic image.

## Implementation Details

### Language and Tools
- **Language**: Python
- **Libraries**: OpenCV for image processing tasks


## References
- [OpenCV SIFT Tutorial](https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html)
- [Python SIFT](https://github.com/rmislam/PythonSIFT)
- [SIFT++](https://github.com/davidstutz/vedaldi2006-siftpp)
- [VLFeat SIFT](http://www.vlfeat.org/overview/sift.html)
- [OpenCV Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [VLFeat UBCMatch](http://www.vlfeat.org/matlab/vl_ubcmatch.html)



