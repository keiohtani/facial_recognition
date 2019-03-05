import authentication
import requests
import json
import urllib.request

UNKNOWN_FACE_PATH = 'unknown_faces'

def download_people_images():
    service = authentication.get_authenticated_service()

    url = 'https://photoslibrary.googleapis.com/v1/mediaItems:search'
    headers = {
        'Authorization': "Bearer " + service._http.request.credentials.access_token,
    }
    with open('downloader_payload.json') as f:
        payload = json.loads(f.read())
    
    nextPageToken = None
    photo_id = 1
    while True:
        response = requests.post(url=url, headers=headers, data=json.dumps(payload))
        for mediaItem in response.json()['mediaItems']:
            image_url = mediaItem['baseUrl'] # the size can be set by adding '=w2048-h1024' at the end of URL
            urllib.request.urlretrieve(image_url + '=w1024', UNKNOWN_FACE_PATH + '/' + str(photo_id) + '.jpg')
            photo_id = photo_id + 1 
        nextPageToken = response.json()['nextPageToken']
        payload['pageToken'] = nextPageToken
        if (nextPageToken == None):
            break

# if __name__ == "__main__":
#     # Google アカウントの認証を行い API 呼び出し用の service object を取得する
    
#     # showing a list of albums
#     # view_albums(service)

    # download_people_images()