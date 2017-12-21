import numpy as np
import tensorflow as tf
import vec_mappings as vecmp
import constants
from flask import Flask, request
from flask_cors import CORS

sess = tf.Session()
saver = tf.train.import_meta_graph('models/model_data_07_2016_init_0.1.ckpt.meta')
saver.restore(sess,'models/model_data_07_2016_init_0.1.ckpt')
myX = np.zeros([1, constants.IMG_HEIGHT*constants.IMG_WIDTH])

graph = tf.get_default_graph()
x_mean_t = graph.get_tensor_by_name('x_mean:0')
x_std_t = graph.get_tensor_by_name('x_std:0')
x_mean, x_std = sess.run([x_mean_t, x_std_t])
x = graph.get_tensor_by_name('x:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
pred = graph.get_tensor_by_name('pred:0')

def tf_predict(request):
    content = request.files.get('pic').read()
    success, X0 = vecmp.load_single_from_binary(content)
    X0 = (X0 - x_mean) / (x_std + 0.00001)
    myX[0,:] = X0
    pp = sess.run(pred, feed_dict={x: myX, keep_prob: 1.})
    p = tf.reshape(pp, [-1, constants.CAPTCHA_LENGTH, 63])
    max_idx_p = tf.argmax(p, 2).eval(session=sess)
    predicted_word = vecmp.map_vec_pos2words(max_idx_p[0, :])
    return predicted_word

app = Flask(__name__)
cors = CORS(app)
@app.route("/api/predict", methods=['POST'])
def predict():
    return tf_predict(request)

app.run(port=9987)
