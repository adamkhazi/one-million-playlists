import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import shelve

class API:
    def __init__(self):
        self.__config = dict(line.strip().split('=') for line in open("../project.config"))
        client_credentials_manager = SpotifyClientCredentials(
            client_id=self.__config['SPOTIFY_API_CLIENT_ID'],
                client_secret=self.__config['SPOTIFY_API_CLIENT_SECRET'])
        self.__sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        self.__sp.trace = False
        self.__sp.trace_out = False

    def getTrackInfo(self, trackURI):
        if not hasattr(self, '__trackCache'):
            self.__trackCache = shelve.open(self.__config['TRACK_CACHE'])
        
        if trackURI not in self.__trackCache:
            track = self.__sp.track(trackURI)
            self.__trackCache[trackURI] = track

        print("track cache size:", len(self.__trackCache))

        return self.__trackCache[trackURI]

    def getAlbumInfo(self, albumURI):
        album = self.__sp.album(albumURI)
        return album

    def getArtistInfo(self, artistURI):
        artist = self.__sp.artist(artistURI)
        return artist

    def getTrackAnalysis(self, trackURI):
        if not hasattr(self, '__trackAnalysisCache'):
            self.__trackAnalysisCache = dict()
        
        if trackURI not in self.__trackAnalysisCache:
            track = self.__sp.audio_analysis(trackURI)
            self.__trackAnalysisCache[trackURI] = track

        return self.__trackAnalysisCache[trackURI]

    def getTrackFeatures(self, trackURI):
        if not hasattr(self, '__trackFeaturesCache'):
            self.__trackFeaturesCache = shelve.open(self.__config['TRACK_FEATURES_CACHE'])
        
        if trackURI not in self.__trackFeaturesCache:
            track = self.__sp.audio_features(trackURI)[0]
            self.__trackFeaturesCache[trackURI] = track
        
        print("track feature cache size:", len(self.__trackFeaturesCache))

        return self.__trackFeaturesCache[trackURI]
    
    def closeTrackCache(self):
        self.__trackCache.close()

    def closeTrackFeatureCache(self):
        self.__trackFeaturesCache.close()