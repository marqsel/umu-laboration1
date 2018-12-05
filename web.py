from flask import Flask, render_template, request
import flask_settings as settings
import base64
import io
from keras.models import load_model
from keras import backend as K
import numpy as np
import utils
app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html', **{
        'canvas_width': settings.CANVAS_WIDTH,
        'canvas_height': settings.CANVAS_HEIGHT
    })


@app.route("/ajax/test", methods=['POST'])
def ajax_test():

    _, imgstring = request.data.decode("utf-8").split(',')
    img = io.BytesIO(base64.b64decode(imgstring))
    img = utils.transform_image(img, settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, True, 'test.png')

    K.clear_session()

    the_model = load_model(settings.MODEL_FILE)

    result = the_model.predict(np.array([img]))
    prediction = np.argmax(result)
    probability = result[0][prediction]

    return 'It\'s a %d (%.0f%%)' % (prediction, probability * 100)


if __name__ == '__main__':
    app.run(debug=settings.DEBUG)
