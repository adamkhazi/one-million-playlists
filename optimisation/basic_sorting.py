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
FEATURE_INDICES = [i for i in range(14)]

print('getting all features...')

cons, testTrackURIs = d.getGoldSetAvgCons('Happy Hits!')
#cons, testTrackURIs = d.getUGSetAvgCons(31) # running 2.0 ug playlist

trackFeatures, trackNames, trackURIs = d.getTrackFeaturesWNames( testTrackURIs, 300000 )

testTrackURIs = [i for i in testTrackURIs if i in trackURIs]
print("nr of songs in editorial playlist:", len(testTrackURIs))

#pdb.set_trace()

#trackFeatures = trackFeatures[:,FEATURE_INDICES]

cons, trackFeatures = d.convertFeaturesToMatrix(cons, trackFeatures)

# features have equal importance
scaler = Normalizer()
trackFeatures = scaler.fit_transform(trackFeatures)

#pdb.set_trace()
cons = np.reshape(cons, (-1, 14))
cons = scaler.transform(cons)
cons = cons[0]

trackFeatures, trackNames, trackURIs = d.sortTrackFeaturesUsingCons(trackFeatures, trackNames, trackURIs, cons)

#ideal = [0.458, 0.591, 5, -5.621, 1, 0.0326, 0.568, 0.0, 0.286, 0.654, 50.558, 161187, 3]
ideal = [cons[i] for i in FEATURE_INDICES]

def UnweightedSum(ideal, trackFeatures):
    trackDiffs = []
    for track in tqdm(trackFeatures):
        trackDiffSum = 0
        for cIdx, cIdeal in enumerate(ideal):
            trackDiffSum += (cIdeal - track[cIdx])**2
        trackDiffs.append(trackDiffSum/len(cons))
    return trackDiffs

def WeightedSum(ideal, trackFeatures, weights):
    '''
    keyOrder = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', \
         'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'popularity', 'release_date']
    '''
    trackDiffs = []
    wSum = sum(weights)
    for track in tqdm(trackFeatures):
        trackDiffSum = 0
        for cIdx, cIdeal in enumerate(ideal):
            trackDiffSum += (weights[cIdx]/wSum) * abs(cIdeal - track[cIdx])
        trackDiffs.append(trackDiffSum/len(cons))
    return trackDiffs

trackDiffs = UnweightedSum(ideal, trackFeatures)

print('sorting')
trackDiffs, trackURIs, trackNames, trackFeatures = zip(*sorted(zip(trackDiffs, trackURIs, trackNames, trackFeatures)))

#for i in range(20):
    #print(trackDiffs[i], trackNames[i])

eval = Evaluation()
print(" set scores ", eval.exactSetMatches(testTrackURIs, trackURIs[:SET_SIZE]))

pdb.set_trace()