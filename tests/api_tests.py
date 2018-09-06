import sys
import os
sys.path.append(os.path.abspath('../'))
from data import Data
from api import API

import pytest

class TestAPI:
    def test_getTrackInfo(self):
        a = API()
        track = a.getTrackInfo(['0UaMYEvWZi0ZqiDOoHU3YI'])
        
        for a in ['album', 'artists', 'available_markets', 'disc_number', 'duration_ms', 'explicit',  \
            'external_ids', 'external_urls', 'href', 'id', 'is_local', 'name', 'popularity', 'preview_url', 'track_number', 'type', 'uri']:
            assert track['tracks'][0][a] != None


    def test_getAlbumInfo(self):
        a = API()
        album = a.getAlbumInfo('6vV5UrXcfyQD1wu4Qo2I9K')
        
        for a in ['album_type', 'artists', 'available_markets', 'copyrights', 'external_ids', 'external_urls', 'genres', 'href', 'id', 'images', 'label', 'name', 'popularity', 'release_date', 'release_date_precision', 'total_tracks', 'tracks', 'type', 'uri']:
            assert album[a] != None


    def test_getArtistInfo(self):
        a = API()
        artist = a.getArtistInfo('2wIVse2owClT7go1WT98tk')

        for a in ['external_urls', 'followers', 'genres', 'href', 'id', 'images', 'name', 'popularity', 'type', 'uri']:
            assert artist[a] != None


    def test_getTrackAnalysis(self):
        a = API()
        track = a.getTrackAnalysis('0UaMYEvWZi0ZqiDOoHU3YI')

        for a in ['meta', 'track', 'bars', 'beats', 'tatums', 'sections', 'segments']:
            assert track[a] != None


    def test_getTrackFeatures(self):
        a = API()
        track = a.getTrackFeatures('0UaMYEvWZi0ZqiDOoHU3YI')

        pdb.set_trace()

        for a in ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature']:
            assert track[0][a] != None
        

    def test_getFeaturedPlaylists(self):
        a = API()
        plist = a.getFeaturedPlaylists(2)

        pdb.set_trace()

        for a in ['collaborative', 'external_urls', 'href', 'id', 'images', 'name', 'owner', 'primary_color', 'public', 'snapshot_id', 'tracks', 'type', 'uri']:
            assert plist['playlists']['items'][0][a] != None
        

    def test_getPlaylist(self):
        a = API()
        plist = a.getPlaylist('37i9dQZF1DX1tyCD9QhIWF')

        pdb.set_trace()

        for a in ['collaborative', 'description', 'external_urls', 'followers', 'href', 'id', 'images', 'name', 'owner', 'primary_color', 'public', 'snapshot_id', 'tracks', 'type', 'uri']:
            assert plist[a] != None

