from __future__ import unicode_literals

import os
import sys
sys.path.insert(0, os.path.join('..', 'model'))

from preprocess_util import *
import youtube_dl

YT_DL_FILE = os.path.join('..', 'download_links', 'yt_list.txt')
SAVE_DIR = os.path.join('..', 'audio-train-transfer')

with open(YT_DL_FILE, 'r') as f:
    vids = [line.strip() for line in f]

profile_list = {}
profile = 0

for line in vids:
    if line.startswith("http") is False:
        profile = line
        profile_list[profile] = []
    else:
        profile_list[profile].append(line)

for key in profile_list:
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'outtmpl': SAVE_DIR + '/' + key + '/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    for URL in profile_list[key]:
        with youtube_dl.YoutubeDL(ydl_opts) as yt_dl:
            yt_dl.download([URL])

distribute_samples(SAVE_DIR)

