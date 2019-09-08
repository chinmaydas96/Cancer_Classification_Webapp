import os

from flask import Flask, render_template, request, redirect

from inference import get_prediction
from commons import format_class_name
from base64 import b64encode
from PIL import Image

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()

       

        fh = open("static/showimage.tif", "wb")
        image = b64encode(img_bytes).decode("utf-8")
        fh.write(img_bytes)
        fh.close()

        im = Image.open("static/showimage.tif")
        im.thumbnail(im.size)
        im.save('static/showimage.jpg', "JPEG", quality=100)



        class_name,prob = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)

        
        #return render_template("show_a.html", obj=obj, image=image)

        return render_template( 'result.html', class_id=prob,
                               class_name=class_name,image=image)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
