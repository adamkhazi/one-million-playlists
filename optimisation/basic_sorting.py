import array
import random
import json

import numpy

import numpy as np

import sys
import os
import math

from sklearn.preprocessing import Normalizer
from tqdm import tqdm

sys.path.append(os.path.abspath('../'))
from data import Data
from evaluation import Evaluation

import pdb

d = Data()
SET_SIZE = 10000
#NR_FEATURES = trackFeatures.shape[1]
NR_FEATURES = 12
FEATURE_INDICES = [i for i in range(13)]

#cons, testTrackURIs = d.getGoldSetAvgCons('Fun Run 150–165 BPM')
cons, testTrackURIs = d.getUGSetAvgCons(31) # running 2.0 ug playlist

trackFeatures, trackNames, trackURIs = d.getTrackFeaturesWNames( testTrackURIs, 10 )
#trackFeatures = trackFeatures[:,FEATURE_INDICES]

cons, trackFeatures = d.convertFeaturesToMatrix(cons, trackFeatures)

# features have equal importance
scaler = Normalizer()
trackFeatures = scaler.fit_transform(trackFeatures)

#pdb.set_trace()
cons = np.reshape(cons, (-1,13))
cons = scaler.transform(cons)
cons = cons[0]

#ideal = [0.458, 0.591, 5, -5.621, 1, 0.0326, 0.568, 0.0, 0.286, 0.654, 50.558, 161187, 3]
ideal = [cons[i] for i in FEATURE_INDICES]

trackDiffs = []
for track in trackFeatures:
    trackDiffSum = 0
    for cIdx, cIdeal in tqdm(enumerate(cons)):
        trackDiffSum += (cIdeal - track[cIdx])**2
    trackDiffs.append(trackDiffSum/len(cons))

trackDiffs, trackURIs = tqdm(zip(*sorted(zip(trackDiffs, trackURIs))))

#for i in range(20):
    #print(trackDiffs[i], trackNames[i])

eval = Evaluation()
print(" set scores ", eval.exactSetMatches(testTrackURIs, trackURIs[:SET_SIZE]))

pdb.set_trace()