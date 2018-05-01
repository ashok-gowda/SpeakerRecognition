import glob
import os
import shutil
import subprocess

import numpy as np

# Cleanup the 'split' directory which contains 'wav' files and spectrograms.
def cleanup_split(root):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            shutil.rmtree(os.path.join(subdir, directory, "split"), ignore_errors = True)
        break

# Experimental function to print the list of directories which
# have more than the intended number of files.
def check_data_spect(root, split):
    for subdir, dirs, files in os.walk(root, split):
        for directory in dirs:
            file_path = os.path.join(subdir, directory, "split", split, "spect")
            f_path, f_dirs, f_files = os.walk(file_path).next()
            if len(f_files) != 2:
                print directory
        break

# API to remove the stored spectrogram files.
def remove_spect(root, split, count):
    for subdir, dirs, files in os.walk(root, split):
        for directory in dirs:
            file_path = os.path.join(subdir, directory, "split", split, "spect")
            f_path, f_dirs, f_files = os.walk(file_path).next()
            for f in f_files:
                fname = os.path.basename(f)
                if int(os.path.splitext(os.path.splitext(fname)[0])[0]) > int(count):
                    os.remove(os.path.join(subdir, directory, "split", split, "spect", f))

            file_path = os.path.join(subdir, directory, "split", split, "wav")
            f_path, f_dirs, f_files = next(os.walk(file_path))
            for f in f_files:
                fname = os.path.basename(f)
                if int(os.path.splitext(os.path.splitext(fname)[0])[0]) > int(count):
                    os.remove(os.path.join(subdir, directory, "split", split, "wav", f))
        break

# Remove all MP3/WAV files that are generated.
def cleanup_merged(root):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            for f in glob.glob(os.path.join(subdir, directory, "*_merged*.*")):
                os.remove(f)
        break

# Remove all generated NumPy data files
def cleanup_npy(root):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            for f in glob.glob(os.path.join(subdir, directory, "*.npy")):
                os.remove(f)
        break

# Rename the directories by removing everything after the first '_'
def rename_samples(root):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            shutil.move(os.path.join(subdir, directory), os.path.join(subdir, directory.split("_")[0]))
        break

# Remove additional samples for a particular speaker
def remove_extra_samples(root, count):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            name_o = directory.split(".") 
            if len(name_o) == 2 and int(name_o[1]) > count:
                shutil.rmtree(os.path.join(subdir, directory))
        break

# Distribute each sample of a speaker into different directories
def distribute_samples(root):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            if len(directory.split(".")) != 1:
                continue
            for d_subdir, d_dirs, d_files in os.walk(os.path.join(root, directory)):
                for i, sample in enumerate(d_files):
                    os.makedirs(os.path.join(subdir, directory + "." + str(i + 1)))
                    shutil.move(os.path.join(subdir, directory, sample), \
                                os.path.join(subdir, directory + "." + str(i + 1), sample))
                break
        remove_extra_samples(root)
        break

# Combine distributed samples back into a single directory
def combine_samples(root):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            if not os.path.isdir(os.path.join(subdir, directory.split(".")[0])):
                os.makedirs(os.path.join(subdir, directory.split(".")[0]))
            else:
                for f in glob.glob(os.path.join(subdir, directory.split(".")[0], "*.npy")):
                    os.remove(f)
                    shutil.rmtree(os.path.join(subdir, directory.split(".")[0], "split"), ignore_errors = True)
            for d_subdir, d_dirs, d_files in os.walk(os.path.join(root, directory)):
                mp3_files = glob.glob(os.path.join(subdir, directory, "*.mp3"))
                for i, sample in enumerate(mp3_files):
                    shutil.move(os.path.join(subdir, directory, os.path.basename(sample)), \
                                os.path.join(subdir, directory.split(".")[0], os.path.basename(sample)))
                if len(directory.split(".")) != 1:
                    shutil.rmtree(os.path.join(subdir, directory))
                break
        break

