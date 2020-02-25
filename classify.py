from bottle import request, response, route, run
import tensorflow as tf
import sys, os, errno, urllib, uuid
import requests



WORKING_DIRECTORY = "tf_files"
TMP_DIRECTORY = "tmp"
TRAINED_LABELS = "%s/retrained_labels_Apple.txt" % (WORKING_DIRECTORY)
RETRAINED_GRAPH = "%s/retrained_graph_Apple.pb" % (WORKING_DIRECTORY)

@route('/classify_image/', method='POST')
def hello():
    json = {}
    lien = request.json['lien']
    print(lien)
    f = open('00000001.jpg','wb')
    f.write(requests.get(lien).content)
    f.close()
    #path = download_image(info['path'], info['ext'])
    json[lien] = score('00000001.jpg')
    os.remove('00000001.jpg')
    return json


@route('/status/', method='GET')
def status():
    return 'online'


def download_image(url, extension):
    filename = TMP_DIRECTORY + '/' + uuid.uuid4().hex + extension
    urllib.urlretrieve(url, filename)
    return filename

def create_tmp(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def score(image_path):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(TRAINED_LABELS)]

    data = []

    with tf.gfile.FastGFile(RETRAINED_GRAPH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s:%.5f' % (human_string, score))
            data.append('%s:%.5f' % (human_string, score))
        return data


create_tmp('tmp')
run(host='10.13.25.15', port=8989, debug=True)
