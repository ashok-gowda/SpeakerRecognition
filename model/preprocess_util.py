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

