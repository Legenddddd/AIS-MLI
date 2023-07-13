# Angular Isotonic Loss guided Multi-Layer Integration for Few-shot Fine-grained Image Classification


## Data Preparation

<!-- Please download the dataset before you run the code.

CUB_200_2011: [CUB_200_2011 download link](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)

Stanford-Dogs : [Stanford-Dogs download link](http://vision.stanford.edu/aditya86/ImageNetDogs/)

iNaturalist2017 : [[iNaturalist2017 Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017), [iNaturalist2017 Download Data](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz), [iNaturalist2017 Download Annotations](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip)\]

 -->
 The following datasets are used in our paper:

CUB_200_2011: [Dataset Page](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

Stanford-Dogs: [Dataset Page](http://vision.stanford.edu/aditya86/ImageNetDogs/)

iNaturalist2017: [Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017)

Please proceed with the setting up data by referring to [TDM Github](https://github.com/leesb7426/CVPR2022-Task-Discrepancy-Maximization-for-Fine-grained-Few-Shot-Classification).



## Usage

### Requirement
All the requirements to run the code are in requirements.txt.

You can download requirements by running below script.
```
pip install -r requirements.txt
```

<!-- ### Dataset directory
Change the data_path in config.yml.
```
dataset_path: #your_dataset_directory
```
 -->

### Train and test
Running the shell script ```run.sh``` will train and evaluate the model with hyperparameters matching our paper.

