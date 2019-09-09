import os
from flask import Flask, render_template, request, redirect
from inference import get_prediction
from commons import format_class_name
from base64 import b64encode
from PIL import Image
from werkzeug import secure_filename
from flask_caching import Cache


app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"


# prevent cached responses
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/', methods=['GET', 'POST'])
#@cache.cached(timeout=50)
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return 
        
        
        img_bytes = file.read()
        file.seek(0)
        file_name = file.filename
        filename = secure_filename(file_name)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],"showimage.tif"))
        
        #fh = open(app.config['UPLOAD_FOLDER'] + "/showimage.tif", "wb")
        #image = b64encode(img_bytes).decode("utf-8")
        #fh.write(img_bytes)
        #fh.close()

        im = Image.open(app.config['UPLOAD_FOLDER'] + "/showimage.tif")
        im.thumbnail(im.size)
        im.save(app.config['UPLOAD_FOLDER'] + "/showimage.jpg", "JPEG", quality=100)
        class_name,prob = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template( 'result.html', class_id=prob,
                               class_name=class_name,image=app.config['UPLOAD_FOLDER'] + "/showimage.jpg")
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
