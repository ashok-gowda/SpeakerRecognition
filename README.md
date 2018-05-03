# Speaker Recognition using LPCC & MFCC

Traditional speaker recognition systems use speech features or neural networks to train the model and they recognize new users if they are trained with sufficiently long samples. We introduce a neural network and SVM model which is trained on using LPCC & MFCC features on a CNN. The new samples are passed through the CNN and trained on an SVM and we classify the speakers with very short audio samples!

## Prerequisites
* Install `python2.7` following the instructions at [Python Docs](https://www.python.org/downloads/release/python-2715/)
* Install `ffmpeg` on your machine.
* Run `pip install youtube-dl tensorflow keras scikit-learn librosa scikits.talkbox`

## Steps to run 
### Setup
* `cd SpeakerRecognition`
* `./setup.sh` Note: This may take a very long time, since the training data is in tens of GBs.
* `cd model`

### Train CNN & perform transfer learning
* There are 4 models involved:
	- Spectrogram
	- LPCC
	- MFCC
	- Hybrid (LPCC & MFCC)
* Each of the above models have 3 separate files in the `model` directory as follows:
	- **model_\<name\>.py** - Contains the model implementation (Need not run this).
	- **train_\<name\>.py** - Run this file to train the CNN on the downloaded data.
	- **transfer_\<name\>.py** - Run this file to transfer the neural network weights and to train SVM on new set of speakers.

## Directories
* **data_pro**: Contains the python files for the data preprocessing.
* **download_links**: Contains files with libriVoX and YouTube audio sample download links.
* **model**: Implementation of the model, in Python.
* **notebooks**: Jupyter Notebook files for each of the **train_*.py** and **transfer_*.py** files in `model` directory. You need to install jupyter notebook on your machine to run these.

## Contact
* Ashok Gowda: ashokgovindagowda@tamu.edu
* Kishan Sheshagiri: kishan.sheshagiri@tamu.edu

