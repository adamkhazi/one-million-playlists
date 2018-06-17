import os
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from pandas.io.json import json_normalize
from pymongo import MongoClient
import json
from joblib import Parallel, delayed
from more_itertools import chunked
import time

from api import API
import pdb

def unwrap_self(arg, **kwarg):
    return Data.updateTrackFeatureData(*arg, **kwarg)

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

    def fillTrackFeaturesDB(self):
        #uniqTrackURIs = list(chunked(uniqTrackURIs), 50)
        client, db = self.getDB()
        uniqTracks = db.uniqueTrackURIs.find({})
        uniqTrackURIs = {u['track_uri']: True for u in uniqTracks}

        featureCache = db.tracksFeatureCache.aggregate([{'$group': {"_id": {"uri":'$uri'}}}], allowDiskUse=True)
        doneSoFar = [u['_id']['uri'] for u in featureCache]
        #uniqueTrackURIs = db.uniqueTrackURIs.aggregate([{"$match": { 'track_uri': {'$nin': featureCache}}}], allowDiskUse=True)
        #uniqueTrackURIs = db.uniqueTrackURIs.find({"uri": {"$nin": featureCache}})
        #tracksNotDownloaded = [u['track_uri'] for u in uniqueTrackURIs] 
        #pdb.set_trace()
        client.close()

        for done in doneSoFar:
            uniqTrackURIs[done] = False
        uniqTrackURIs = [k for k,v in uniqTrackURIs.items() if v]
        uniqTrackURIs = list(chunked(uniqTrackURIs, 50))

        #for i in tqdm(uniqTrackURIs):
        #    self.updateTrackFeatureData(i)

        Parallel(n_jobs=-1)(delayed(self.updateTrackFeatureData)(uri) for uri in zip([self]*len(uniqTrackURIs), uniqTrackURIs))

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
                        c.close()
                        time.sleep(60)
                        c, db = self.getDB()
                break
            except:
                c.close()
                time.sleep(60)
                c, db = self.getDB()
        c.close()

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