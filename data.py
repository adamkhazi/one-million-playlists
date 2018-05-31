import os
import json

class Data(object):
    DATASET_DIR = "C:/Users/Adam Khazi/Dropbox/University College London/MSc/Dissertation/Datasets/data/"

    # 1000 playlists per file
    def load(self, nr_files=1):
        dataFileNames = os.listdir(self.DATASET_DIR)
        playlistData = []
        for i in range(nr_files):
            with open(self.DATASET_DIR + dataFileNames[i]) as f:
                data = json.load(f)
                playlistData.append(data)
        return playlistData