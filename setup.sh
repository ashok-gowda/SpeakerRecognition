#!/bin/bash

# Create directories
mkdir -p audio-train-new audio-train-transfer neural-net-weights

# Download training data for CNN
xargs -n 1 -P 20 wget -P audio-train-new < download_links/urls.txt

# Unzip all the files
unzip 'audio-train-new/*.zip' -d audio-train-new

# Download testing data from YouTube
cd data_pro && python yt_download.py && cd -

