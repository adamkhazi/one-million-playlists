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

    def getTrackInfo(self, trackURI):
        if not hasattr(self, '__trackCache'):
            self.__trackCache = dict()
        
        if trackURI not in self.__trackCache:
            track = self.__sp.track(trackURI)
            self.__trackCache[trackURI] = track

        return self.__trackCache[trackURI]

    def getAlbumInfo(self, albumURI):
        album = self.__sp.album(albumURI)
        return album

    def getArtistInfo(self, artistURI):
        artist = self.__sp.artist(artistURI)
        return artist

    def augmentTracksTable(self, trackDataDF):
        pass