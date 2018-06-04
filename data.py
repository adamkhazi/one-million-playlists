import os
import json

class Data(object):
    def getDatasetPath(self):
        paths = open("../dataset_paths.config").read().split('\n')
        return paths[0]

    # 1000 playlists per file
    def load(self, nr_files=1):
        dataDir = self.getDatasetPath()
        dataFileNames = os.listdir(dataDir)
        playlistData = []
        for i in range(nr_files):
            with open(dataDir + dataFileNames[i]) as f:
                data = json.load(f)
                playlistData.append(data)
        return playlistData

    def loadTracks(self, nr_files=1):
        dataDir = self.getDatasetPath()
        dataFileNames = os.listdir(dataDir)
        trackData = []
        for i in range(nr_files):
            with open(dataDir + dataFileNames[i]) as f:
                data = json.load(f)
                for playlist in data['playlists']:
                    for track in playlist['tracks']:
                        trackData.append(track)
        return trackData