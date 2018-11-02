from data import Data

from tqdm import tqdm

import pdb

class MongoToSql:
    def moveUGTrackFeatures(self):
        d = Data()
        c, db = d.getDB()
        trackFeatures = db.tracksFeatureCache.aggregate([{'$match': {} }], allowDiskUse=True)


        conn, cur = d.getSQLDB()
        for t in tqdm(trackFeatures):
            if "popularity" in t:
                pop = t["popularity"]
            else:
                pop = 0
            if "release_date" in t:
                release_date = t["release_date"]
            else:
                release_date = 0
            if t["valence"]:
                valence = t["valence"]
            else:
                valence = 0
            
            cur.execute("INSERT INTO ug_track_features (acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, mode, popularity, release_date, speechiness, tempo, time_signature, valence, track_uri) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (t["acousticness"],  t["danceability"], t["duration_ms"], t["energy"], t["instrumentalness"], t["key"], t["liveness"], t["loudness"], t["mode"], pop, release_date, t["speechiness"], t["tempo"], t["time_signature"], t["valence"], t["uri"]))

        conn.commit()

        cur.close()
        conn.close()