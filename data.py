import os
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize
import numpy as np
from pymongo import MongoClient
import json
from joblib import Parallel, delayed
from more_itertools import chunked
import time
from random import randint

from api import API
import pdb

class Data(object):
    def __init__(self):
        self.__config = dict(line.strip().split('=') for line in open("../project.config"))

    def getTrackDfPath(self):
        return self.__config['ONE_MILLION_PLAYLISTS_TRACKS_FORMATTED']

    def getPlaylistDfPath(self):
        return self.__config['ONE_MILLION_PLAYLISTS_FORMATTED']

    def getDatasetPath(self):
        return self.__config['ONE_MILLION_PLAYLISTS_DATASET']

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
        for i in tqdm(range(nr_files)):
            with open(dataDir + dataFileNames[i]) as f:
                data = json.load(f)
                for playlist in data['playlists']:
                    for track in playlist['tracks']:
                        trackData.append(track)
        return trackData

    def loadFormattedTracks(self, nr_files=1):
        dataDir = self.getDatasetPath()
        dataFileNames = os.listdir(dataDir)

        trackDataDF = pd.DataFrame(columns=['album_name', 'album_uri', 'artist_name', 
        'artist_uri', 'duration_ms', 'pos', 'track_name', 'track_uri'])
        
        trackRows = []
        for i in tqdm(range(nr_files)):
            with open(dataDir + dataFileNames[i]) as f:
                data = json.load(f)
                trackRows.extend([t for playlist in data['playlists'] for t in playlist['tracks']])

                if i % 250 == 0 and i > 0:
                    trackDataDF = trackDataDF.append(trackRows, ignore_index=True)
                    trackRows = None
                    trackRows = []

        trackDataDF = trackDataDF.append(trackRows, ignore_index=True)
        trackRows = None

        return trackDataDF

    def loadFormattedPlaylists(self, nr_files=1):
        dataDir = self.getDatasetPath()
        dataFileNames = os.listdir(dataDir)

        playlistDataDF = pd.DataFrame(columns=['num_artists', 'num_albums', 'num_tracks',
        'num_followers', 'num_edits', 'modified_at', 'modified_at_month', 'modified_at_year',
        'modified_at_day', 'duration_ms', 'duration_sec', 'duration_min', 'collaborative'])

        for i in tqdm(range(nr_files)):
            with open(dataDir + dataFileNames[i]) as f:
                playlistSet = json.load(f)
                playListSetRows = []

                origLen = len(playlistSet['playlists'])
                for _ in range(origLen):
                    row = dict()
                    row['num_artists'] = playlistSet["playlists"][0]['num_artists']
                    row['num_albums'] = playlistSet["playlists"][0]['num_albums']
                    row['num_tracks'] = playlistSet["playlists"][0]['num_tracks']
                    row['num_followers'] = playlistSet["playlists"][0]['num_followers']
                    row['num_edits'] = playlistSet["playlists"][0]['num_edits']
                    row['modified_at'] = playlistSet["playlists"][0]['modified_at']
                    row['modified_at_month'] = int(datetime.fromtimestamp(playlistSet["playlists"][0]['modified_at']).strftime('%m'))
                    row['modified_at_year'] = int(datetime.fromtimestamp(playlistSet["playlists"][0]['modified_at']).strftime('%Y'))
                    row['modified_at_day'] = int(datetime.fromtimestamp(playlistSet["playlists"][0]['modified_at']).strftime('%d'))
                    row['modified_at_week_day'] = datetime.fromtimestamp(playlistSet["playlists"][0]['modified_at']).strftime('%A')
                    row['duration_ms'] = playlistSet["playlists"][0]['duration_ms']
                    row['duration_sec'] = playlistSet["playlists"][0]['duration_ms'] / 1000
                    row['duration_min'] = (playlistSet["playlists"][0]['duration_ms'] / 1000)/60
                    row['collaborative'] = playlistSet["playlists"][0]['collaborative']
                    
                    del playlistSet["playlists"][0]
                    playListSetRows.append(row)

                playlistDataDF = playlistDataDF.append(playListSetRows, ignore_index=True)
        return playlistDataDF

    def loadFormattedPlaylistsAndTracks(self, nr_files=1):
        dataDir = self.getDatasetPath()
        dataFileNames = os.listdir(dataDir)

        sets = []
        for i in tqdm(range(nr_files)):
            sets.append(pd.io.json.json_normalize(json.load(open(dataDir + dataFileNames[i]))['playlists'], 'tracks', ['collaborative', 'description', 'duration_ms',
                'modified_at', 'name', 'num_albums', 'num_artists', 'num_edits', 'num_followers', 
                    'num_tracks', 'pid'], errors='ignore', meta_prefix='playlist.'))
        return pd.concat(sets)

    def getDB(self):
        client = MongoClient(self.__config['MONGO_DB_ADDR'], int(self.__config['MONGO_DB_PORT']), connect=False)
        return client, client[self.__config['MONGO_DB_NAME']]

    def loadFormattedPlaylistFilesDB(self):
        dataDir = self.getDatasetPath()
        dataFileNames = os.listdir(dataDir)
        client, db = self.getDB()

        for i in tqdm(range(len(dataFileNames))):
            playlistDf = pd.io.json.json_normalize(json.load(open(dataDir + dataFileNames[i]))['playlists'], 'tracks', ['collaborative', 'description', 'duration_ms',
                'modified_at', 'name', 'num_albums', 'num_artists', 'num_edits', 'num_followers', 
                    'num_tracks', 'pid'], errors='ignore', meta_prefix='playlist_')

            playlistDf = json.loads(playlistDf.T.to_json()).values()
            db.tracks.insert(playlistDf)

    def clearTracksDB(self):
        c, db =  self.getDB()
        db.tracks.remove({})

    def fillTrackDB(self):
        client, db = self.getDB()
        uniqTracks = db.uniqueTrackURIs.find({})
        uniqTrackURIs = {u['track_uri']: True for u in uniqTracks}

        tracksCatalog = db.tracksCatalog.aggregate([{'$group': {"_id": {"uri": '$uri'}}}], allowDiskUse=True)
        doneSoFar = [u['_id']['uri'] for u in tracksCatalog]
        client.close()
        print(len(doneSoFar), '/', len(uniqTrackURIs))

        for done in doneSoFar:
            uniqTrackURIs[done] = False
        uniqTrackURIs = [k for k,v in uniqTrackURIs.items() if v]
        uniqTrackURIs = list(chunked(uniqTrackURIs, 50))

        Parallel(n_jobs=-1, verbose=70)(delayed(self.updateTrackCatalog)(c) for c in uniqTrackURIs)

    def updateTrackCatalog(self, uris):
        client, db = self.getDB()
        a = API()
        while True:
            try:
                features = a.getTrackInfo(uris)
                print(features)
            except (ConnectionError, ValueError('TransportableException')) as e:
                time.sleep(10) # rate-limiting
                continue
            
            res = []
            for f in features['tracks']:
                if f:
                    res.append(f) 
            if res:
                inID = db.tracksCatalog.insert_many(res).inserted_ids

            break

    def fillTrackFeaturesDB(self):
        client, db = self.getDB()
        uniqTracks = db.uniqueTrackURIs.find({})
        uniqTrackURIs = {u['track_uri']: True for u in uniqTracks}

        featureCache = db.tracksFeatureCache.aggregate([{'$group': {"_id": {"uri": '$uri'}}}], allowDiskUse=True)
        doneSoFar = [u['_id']['uri'] for u in featureCache]
        client.close()

        for done in doneSoFar:
            uniqTrackURIs[done] = False
        uniqTrackURIs = [k for k,v in uniqTrackURIs.items() if v]
        uniqTrackURIs = list(chunked(uniqTrackURIs, 50))

        for i in tqdm(uniqTrackURIs):
            self.updateTrackFeatureData(i)

    def updateTrackFeatureData(self, trackURI):
        c, db = self.getDB()

        while True:
            try:
                a = API()
                while True:
                    try:
                        features = a.getTrackFeatures(trackURI)
                        res = []
                        for f in features:
                            if f:
                                res.append(f) 
                        inID = db.tracksFeatureCache.insert_many(res).inserted_ids
                        break  
                    except ConnectionError:
                        pass
                break
            except:
                pass

    def insertDistinctURIs(self):
        c, db = self.getDB()
        uniqTracks = db.tracks.aggregate([{'$group': {"_id": {"track_uri":'$track_uri'}}}], allowDiskUse=True)
        uniqTrackURIs = [u['_id']['track_uri'] for u in uniqTracks]

        for u in uniqTrackURIs:
            db.uniqueTrackURIs.insert_one({"track_uri": u}).inserted_id
        
    def savePlaylistDf(self, df):
        path = self.getPlaylistDfPath()
        df.to_pickle(path)
    
    def loadPlaylistDf(self):
        path = self.getPlaylistDfPath()
        return pd.read_pickle(path)

    def saveTrackDf(self, df):
        path = self.getTrackDfPath()
        df.to_pickle(path)
    
    def loadTrackDf(self):
        path = self.getTrackDfPath()
        return pd.read_pickle(path)

    def addTrackAPIFields(self, trackDf):
        spotifyAPI = API()
        tqdm.pandas()

        def apiFields(uri):

            features = spotifyAPI.getTrackFeatures(uri)
            return pd.Series([spotifyAPI.getTrackInfo(uri)['popularity'], features['danceability'], features['energy'],
                features['key'], features['loudness'], features['mode'], features['speechiness'], features['acousticness'],
                    features['instrumentalness'], features['liveness'], features['valence'], features['tempo']])

        trackDf[['popularity', 'danceability', 'energy',
                'key', 'loudness', 'mode', 'speechiness', 
                'acousticness', 'instrumentalness','liveness',
                'valence','tempo']] = trackDf['track_uri'].progress_apply(apiFields)

        return trackDf

    def getAllUniqueTrackFeatures(self):
        c, db = self.getDB()
        trackFeatures = db.tracksFeatureCache.aggregate([
         {"$lookup": {"from":"tracksCatalog", "localField":"uri", "foreignField": "uri", "as": "catalog"}},
         {'$project': {"_id": 0, "uri": 1, "danceability": 1, "energy": 1, "key": 1, "loudness": 1, "mode": 1, "speechiness": 1, "acousticness": 1,
                    "instrumentalness": 1, "liveness": 1, "valence": 1, "tempo": 1, "time_signature": 1, "catalog.popularity": 1}}
        ], allowDiskUse=True)

        # $lookup for popularity

        return pd.DataFrame(list(trackFeatures))

    def createPlaylistAvgFeatures(self):
        c, db = self.getDB()
        db.tracks.aggregate([
            {"$lookup": {"from": "tracksFeatureCache", "localField": "track_uri", "foreignField": "uri", "as": "trackFeatures"}},
            {"$unwind": "$trackFeatures" },
            {"$group": {"_id": "$playlist_pid", "avgEnergy": {"$avg": "$trackFeatures.energy" },
             "avgDanceability": {"$avg": "$trackFeatures.danceability" },
             "avgKey": {"$avg": "$trackFeatures.key"}, 
             "avgLoudness": {"$avg": "$trackFeatures.loudness"},
             "avgMode": {"$avg": "$trackFeatures.mode"},
             "avgSpeechiness": {"$avg": "$trackFeatures.speechiness"},
             "avgAcousticness": {"$avg": "$trackFeatures.acousticness"},
             "avgInstrumentalness": {"$avg": "$trackFeatures.instrumentalness"},
             "avgLiveness": {"$avg": "$trackFeatures.liveness"},
             "avgValence": {"$avg": "$trackFeatures.valence"},
             "avgTempo": {"$avg": "$trackFeatures.tempo"},
             "avgTimeSig": {"$avg": "$trackFeatures.time_signature"}}}, 
            #{"$project": {"_id": 0, "playlist_pid": 1, "avgEnergy"} },
            {"$out": "playlistAvgFeatures"}
        ], allowDiskUse=True) 

    def createPlaylistMaxFeatures(self):
        c, db = self.getDB()
        db.tracks.aggregate([
            {"$lookup": {"from": "tracksFeatureCache", "localField": "track_uri", "foreignField": "uri", "as": "trackFeatures"}},
            {"$unwind": "$trackFeatures" },
            {"$group": {"_id": "$playlist_pid", 
             "maxEnergy": {"$max": "$trackFeatures.energy"},
             "maxDanceability": {"$max": "$trackFeatures.danceability" },
             "maxKey": {"$max": "$trackFeatures.key"}, 
             "maxLoudness": {"$max": "$trackFeatures.loudness"},
             "maxMode": {"$max": "$trackFeatures.mode"},
             "maxSpeechiness": {"$max": "$trackFeatures.speechiness"},
             "maxAcousticness": {"$max": "$trackFeatures.acousticness"},
             "maxInstrumentalness": {"$max": "$trackFeatures.instrumentalness"},
             "maxLiveness": {"$max": "$trackFeatures.liveness"},
             "maxValence": {"$max": "$trackFeatures.valence"},
             "maxTempo": {"$max": "$trackFeatures.tempo"},
             "maxTimeSig": {"$max": "$trackFeatures.time_signature"}}}, 
            #{"$project": {"_id": 0, "playlist_pid": 1, "avgEnergy"} },
            {"$out": "playlistMaxFeatures"}
        ], allowDiskUse=True)

    def getTrackFeatures(self, limNr=0):
        c, db = self.getDB()
        trackFeatures =  list(map(lambda x: list(x.values()), db.tracksFeatureCache.find({}, {"_id": False, "type": False, "id": False, "uri": False, "track_href": False, "analysis_url": False}).limit( limNr )))
        res = np.array(trackFeatures)
        return res
    
    def getPlaylistAvgFeatures(self, limNr=0):
        c, db = self.getDB()
        playlistFeatures =  list(map(lambda x: list(x.values()), db.playlistAvgFeatures.find({}, {"_id": False}).limit( limNr )))
        res = np.array(playlistFeatures)
        return res

    def getPlaylistMaxFeatures(self, limNr=0):
        c, db = self.getDB()
        playlistFeatures =  list(map(lambda x: list(x.values()), db.playlistMaxFeatures.find({}, {"_id": False}).limit( limNr )))
        res = np.array(playlistFeatures)
        return res

    def downloadedEditorialPlaylists(self):
        c, db = self.getDB()
        a = API()
        editorial = a.getFeaturedPlaylists()
        for i in editorial['playlists']['items']:
            ep = a.getPlaylist(i['id'])
            if db.editorialPlaylists.find({'id': ep['id']}).count() == 0:
                db.editorialPlaylists.insert_one(ep)

    def downloadedSpecificEditorialPlaylist(self, uri):
        c, db = self.getDB()
        a = API()
        ep = a.getPlaylist(uri)
        if db.editorialPlaylists.find({'id': ep['id']}).count() == 0:
            db.editorialPlaylists.insert_one(ep)