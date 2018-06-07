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
        track = self.__sp.track(trackURI)
        print(track)