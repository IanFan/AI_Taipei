from PIL import Image
from io import BytesIO

class File(object):
    def __init__(self):
        super(File, self).__init__()

    def save_bytes_image(self, raw):
        file_path = './media/' + 'user_sent' + '.jpg'
        img = Image.open(BytesIO(raw))
        img.save(file_path)
        return img, file_path
