from urllib.request import urlopen, urlretrieve
from xml.dom import minidom
import datetime
import cv2
import re
import ssl
import json
from image_content import ImageContent

def download_bing_image(idx=0):
    # Getting the XML File
    try:
        usock = urlopen(''.join(['http://www.bing.com/HPImageArchive.aspx?format=xml&idx=',
                                 str(idx), '&n=1&mkt=ru-RU']))  # ru-RU, because they always have 1920x1200 resolution
    except Exception as e:
        print('Error while downloading #', idx, e)
        return
    try:
        xmldoc = minidom.parse(usock)
    # This is raised when there is trouble finding the image url.
    except Exception as e:
        print('Error while processing XML index #', idx, e)
        return
    # Parsing the XML File

    element = xmldoc.getElementsByTagName('copyright')[0]
    caption = element.firstChild.nodeValue

    # url tag
    element = xmldoc.getElementsByTagName('url')[0]
    url = 'http://www.bing.com' + element.firstChild.nodeValue
    # Get Current Date as fileName for the downloaded Picture
    now = datetime.datetime.now()
    date = now - datetime.timedelta(days=int(idx))
    name = date.strftime('%d-%m-%Y')
    print('Downloading: ', name, 'index #', idx)

    # Download and Save the Picture
    # Get a higher resolution by replacing the file name
    filename, _ = urlretrieve(url.replace('_1366x768', '_1920x1200'))
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    return ImageContent(img, name, caption)


def download_ng_image(idx=0):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = urlopen("https://photography.nationalgeographic.com/photography/photo-of-the-day/_jcr_content/.gallery.json", context=ctx)
    gallery_json = json.loads(url.read())

    num_images = len(gallery_json['items'])
    if idx >= num_images:
        return None
    uri = gallery_json["items"][idx]["image"]["uri"]
    title = gallery_json["items"][idx]["image"]["title"]
    credit = gallery_json["items"][idx]["image"]["credit"]
    description = gallery_json["items"][idx]["image"]["caption"].replace("<p>", "").replace("</p>\n", "")
    
    filename, _ = urlretrieve(uri)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    now = datetime.datetime.now()
    date = now - datetime.timedelta(days=int(idx))
    name = date.strftime('%d-%m-%Y')
    print('Downloading: ', name)

    return ImageContent(img, name, title, description, credit)


def test():
    download_ng_image()

if __name__ == '__main__':
    test()