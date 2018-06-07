import spotipy

class API:
    def __init__(self):
        self.__config = dict(line.strip().split('=') for line in open("../project.config"))
        self.__sp = spotipy.Spotify(auth=self.__config['SPOTIFY_API_KEY'])
        self.__sp.trace = True
        self.__sp.trace_out = True

    def getTrackInfo(self, trackURI):
        track = self.__sp.track(trackURI)
        print(track)