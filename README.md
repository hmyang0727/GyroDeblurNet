## Gyro-based Neural Single Image Deblurring (CVPR 2025)


## News
- [2025.02.27] GyroDeblurNet is accepted to CVPR 2025!
- [2025.07.06] Model and testing code have been released.

## Download

| Dataset| Link |
| :-----: | :--:  | 
| GyroBlur-Synth | [Google Drive](https://drive.google.com/drive/folders/1Bv-A2biucSXWS8EwwoLVzREDcwoDrjzi?usp=drive_link)|
| GyroBlur-Real  | [Google Drive](https://drive.google.com/drive/folders/1l7TJ9qQCmLAY8FVV4t13x4rahcczw_Lz?usp=drive_link) |

<details>
<summary><strong>How to use the datasets</strong> (click) </summary>

### GyroBlur-Synth dataset
```md
GyroBlur-Synth
├── train (Training data)
│   ├── avg_blur: Blurred images 
│   ├── sat_mask: Saturation masks for RSBlur pipeline (For detail, refer to the RSBlur paper)
│   ├── sharp: Ground-truth images
│   └── GyroBlur-Synth_train_starting_point.txt: Starting point of images in the raw gyro data sequence
├── test (Test data)
│   ├── avg_blur: Blurred images 
│   ├── sat_mask: Saturation masks for RSBlur pipeline (For detail, refer to the RSBlur paper). For the test dataset, saturations have already been added to the blurred images.
│   ├── sharp: Ground-truth images
│   └── GyroBlur-Synth_test_starting_point.txt: Starting point of each images in the raw gyro data sequence
├── gyro_train.txt: Raw gyro data sequence for the training data
└── gyro_test.txt: Raw gyro data sequence for the test data
```
Each line  of `GyroBlur-Synth_*_starting_point.txt` denotes the starting point of gyro data that corresponds to the image.  
For example, n-th line of the file corresponds to the n-th image when the images are sorted in ascending order.  
If the n-th line of the file is 1234, then the gyro data of n-th image can be retrieved from the raw gyro data sequence by slicing its 1234-th line and 1244-th line since we use 11 gyro data to generate a blurred image.

### GyroBlur-Real dataset
Description will be added

</details>

## How to use
### Generating camera motion field
`generate_camera_motion_field.py` file contains code for generating camera motion field.  
Note that the code generates camera motion field that corresponds to the given gyro data without any perturbation.  
If you want to add some perturbations to the camera motion field, you need to implement your own routine.  
The code contains camera intrinsic matrix of the 4KRD dataset, which is the source of GyroBlur-Synth sharp images.  
If you apply the code to other images, you need to change the intrinsic.

## TODO
- [ ] Code & Dataset release
