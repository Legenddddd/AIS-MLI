# Angular Isotonic Loss Guided Multi-Layer Integration for Few-shot Fine-grained Image Classification


## Instruction with Data Preparation

<!-- Please download the dataset before you run the code.

CUB_200_2011: [CUB_200_2011 download link](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)

Stanford-Dogs : [Stanford-Dogs download link](http://vision.stanford.edu/aditya86/ImageNetDogs/)

iNaturalist2017 : [[iNaturalist2017 Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017), [iNaturalist2017 Download Data](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz), [iNaturalist2017 Download Annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip)\]

 -->
 The following datasets are used in our paper:

CUB_200_2011: [Dataset Page](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

Stanford-Dogs: [Dataset Page](http://vision.stanford.edu/aditya86/ImageNetDogs/)

iNaturalist2017: [Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017)

Please proceed with the setting up data by referring to [FRN Github](http://github.com/Tsingularity/FRN#setting-up-data).

After setting up few-shot datasets, the following folders will exist in your `data_path`:
- `CUB_fewshot_cropped`: 100/50/50 classes for train/validation/test, using bounding-box cropped images as input
- `CUB_fewshot_raw`: class split same as above, using raw un-cropped images as input
- `StanfordDogs_fewshot`: 70/20/30 classes for train/validation/test
- `meta_iNat`: 908/227 classes for train/test. <!-- Holds softlinks to images in `inat2017_84x84` -->
- `tiered_meta_iNat`: 781/354 classes for train/test, split by superclass. <!-- Holds softlinks to images in `inat2017_84x84`  -->

Under each folder, images are organized into `train`, `val`, and `test` folders. In addition, you may also find folders named `val_pre` and `test_pre`, which contain validation and testing images pre-resized to 84x84 for the sake of speed.

## Instruction with Setup

This code requires Pytorch 1.7.0 and torchvision 0.8.0 or higher with cuda support. 

### Requirement
All the requirements to run the code are in requirements.txt.

You can download requirements by running below script.
```
pip install -r requirements.txt
```

### Setting up data
Change the data_path in config.yml.

This should be the absolute path of the folder where you plan to store all the data.
```
data_path: #your_dataset_directory
```


## Train and Test
Running the shell script ```scripts/run.sh``` will train and evaluate the model with hyperparameters matching our paper.

We have provided the checkpoints in ```checkpoints/```, and running the shell script ```scripts/test.sh``` can directly load the trained model to evaluate the code.

