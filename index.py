from flask import Flask, render_template, request, url_for, send_from_directory
from flask_bootstrap import Bootstrap
import config
from werkzeug.utils import secure_filename, redirect
import os
# import logging
app = Flask(__name__)
app.config.from_object(config)
bootstrap = Bootstrap(app)
# log = logging.getLogger('werkzeug')
# log.disabled = True

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

@app.route('/')
def user():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

import Kersa_result
#
@app.route('/result/<int:id>')
def get_result(id):
    Kersa_result.get_result(id)
    return render_template('result.html',id=id)

# @app.route('/result/<int:id>')
# def get_result(id):
#     # Kersa3.get_result(id,type)
#     return render_template('result.html',id=id)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            id = Kersa_result.test_input_img_paths.index('static/DataSet/test/images/' + filename)
    return redirect(url_for('get_result',id=id))


@app.route("/download", methods=['GET'])
def download_file():
    # 此处的filepath是文件的路径，但是文件必须存储在static文件夹下， 比如images\test.jpg
    return send_from_directory("static/supply",filename="supply.zip",as_attachment=True)

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        # host='0.0.0.0',
        port=5001,
        debug=True
    )
