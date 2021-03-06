import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class API:
    def __init__(self):
        self.__config = dict(line.strip().split('=') for line in open("../project.config"))
        client_credentials_manager = SpotifyClientCredentials(
            client_id=self.__config['SPOTIFY_API_CLIENT_ID'],
                client_secret=self.__config['SPOTIFY_API_CLIENT_SECRET'])
        self.__sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        self.__sp.trace = False
        self.__sp.trace_out = False

    def getTrackInfo(self, trackURIs):
        tracks = self.__sp.tracks(trackURIs)
        return tracks

    def getAlbumInfo(self, albumURI):
        album = self.__sp.album(albumURI)
        return album

    def getArtistInfo(self, artistURI):
        artist = self.__sp.artist(artistURI)
        return artist

    def getTrackAnalysis(self, trackURI):
        track = self.__sp.audio_analysis(trackURI)
        return track

    def getTrackFeatures(self, trackURIs):
        track = self.__sp.audio_features(trackURIs)        
        return track

    def getFeaturedPlaylists(self, limit=50):
        p = self.__sp.featured_playlists(limit=limit)        
        return p

    def getPlaylist(self, id):
        p = self.__sp.user_playlist('spotify', id)        
        return p
