import sys
import os
import json
import csv
from random import sample
from tqdm import tqdm
sys.path.append(os.path.abspath('../'))
from data import Data

import pdb

class Submission:
    def create(self):
        d = Data()
        c, db = d.getDB()

        print("fetching unique URIs...")
        trackURIs =  list(map(lambda x: list(x.values())[0], db.uniqueTrackURIs.find({}, {"_id": False})))
        challengeSetPath = dict(line.strip().split('=') for line in open("../project.config"))["CHALLENGE_SET"]
        with open(challengeSetPath) as f:
            challengeSet = json.load(f)

        print("fetching seed tracks...")
        seedTracks = {track["track_uri"]:True for playlist in challengeSet["playlists"] for track in playlist["tracks"]}

        trackURIs = [uri for uri in trackURIs if uri not in seedTracks]

        print("sampling...")
        res = []
        for playlist in tqdm(challengeSet["playlists"]):
            subRes = []
            subRes.append(playlist["pid"])
            subRes.extend(sample(trackURIs, 500))
            res.append(subRes)

        print("saving...")
        with open(dict(line.strip().split('=') for line in open("../project.config"))["CHALLENGE_SET_DEST"], 'w') as f:
            wr = csv.writer(f)
            firstLine = ['team_info', 'main', 'ucl-msc', 'adam.khazi.17@ucl.ac.uk']
            wr.writerow(firstLine)
            wr.writerows(res)



if __name__ == "__main__":
    s = Submission()
    s.create()
    print("done")