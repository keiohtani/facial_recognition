import authentication
import json
import urllib.request
import face_recog


UNKNOWN_FACE_PATH = 'unknown_faces'

def download_people_images():

    service = authentication.get_authenticated_service()

    with open('downloader_payload.json') as f:
        payload = json.loads(f.read())
    
    nextPageToken = ''
    with open('id.txt') as f:
        photo_id = int(f.readline())
        print('Photo id', photo_id, 'is loaded.')
    
    while True:
        media_list = service.mediaItems().search(body=payload).execute()
        if 'mediaItems' not in media_list:  # when no items are found
            break
        for mediaItem in media_list['mediaItems']:
            image_url = mediaItem['baseUrl'] # the size can be set by adding '=w2048-h1024' at the end of URL
            urllib.request.urlretrieve(image_url + '=w1024', 'temp.jpg')
            photo_id = face_recog.save_face_from_path('temp.jpg', photo_id)
        if 'nextPageToken' not in media_list:
            break
        print('next page')
        nextPageToken = media_list['nextPageToken']
        payload['pageToken'] = nextPageToken

    with open('id.txt', 'w') as f:
        f.write(str(photo_id))
        print('Photo id', photo_id, 'is saved.')

    print('finished downloading pictures')

if __name__ == "__main__":
    download_people_images()