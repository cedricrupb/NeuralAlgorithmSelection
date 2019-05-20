import flask
from flask import Flask, url_for, request
from flask_restful import Resource, Api, abort
from werkzeug.utils import secure_filename

import os
from os.path import isfile
import subprocess
from multiprocessing import Process
import re
from subprocess import PIPE
from glob import glob
import uuid
import json
import datetime


UPLOAD_FOLDER = "./upload/"
ALLOWED_EXTENSIONS = set(['.c', '.i'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

api = Api(app)

GRAPH_PROCESSES = {}

__pesco_path__ = "./pesco/"
__ram__ = "10g"

if not os.path.isdir("./upload"):
    os.mkdir("./upload")


def allowed_file(filename):
    return os.path.splitext(filename)[1] in ALLOWED_EXTENSIONS


def run_graphgen(file, out_path, timeout=900):

    global __pesco_path__
    global __ram__

    if not os.path.isfile(file):
        return "File \"%s\" does not exists." % file

    if not os.path.isdir(__pesco_path__):
        return "PeSCo is not initialised!"

    cpash_path = os.path.join(__pesco_path__, 'scripts', 'cpa.sh')
    output_path = os.path.join(out_path, "graph.json")

    try:
        proc = subprocess.run([cpash_path,
                               '-graphgen',
                               '-heap', __ram__,
                               '-skipRecursion',
                               '-outputpath', out_path,
                               '-setprop', 'neuralGraphGen.output=%s' % output_path,
                               file
                               ],
                              check=False, stdout=PIPE, stderr=PIPE,
                              timeout=timeout)
        match_vresult = re.search(r'Verification\sresult:\s([A-Z]+)\.', str(proc.stdout))
        if match_vresult is None:
            raise ValueError('Invalid output of CPAChecker.')
        if match_vresult.group(1) != 'TRUE':
            raise ValueError('ASTCollector Analysis failed:')
        if not isfile(output_path):
            raise ValueError('Invalid output of CPAChecker: Missing graph output')
    except ValueError as err:
        timestamp = str(datetime.datetime.now())
        return "\n".join(["ERROR %s" % timestamp, str(err),
                          str(proc.args),
                          proc.stdout.decode('utf-8'),
                          proc.stderr.decode('utf-8')])

    return "\n".join([timestamp, str(proc.args),
                      proc.stdout.decode('utf-8'),
                      proc.stderr.decode('utf-8')])


def run_save_graphgen(file, out_path, timeout=900):
    text = run_graphgen(file, out_path, timeout)

    with open(os.path.join(out_path, "response.txt"), "w") as o:
        o.write(text)

    return 1 if text.startswith("ERROR") else 0


class VerificationFile(Resource):

    def get(self, id, jsonFormat=False):

        UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
        upload = os.path.join(UPLOAD_FOLDER, id)
        if not os.path.isdir(upload):
            abort(400, message="%s is unknown." % id)

        c = list(glob(os.path.join(upload, "*.c")))
        c.extend(glob(os.path.join(upload, "*.i")))

        if len(c) != 1:
            abort(400, message="%s is unknown." % id)

        c = c[0]

        if jsonFormat == "json":
            with open(c, "r") as i:
                return {"file_id": id, "content": i.read()}
        elif jsonFormat:
            abort(400, message="%s is not supported." % str(jsonFormat))

        c = os.path.basename(c)

        return flask.send_from_directory(upload, c)

    def post(self, id, jsonFormat=None):

        if id != "create":
            abort(404)

        uid = str(uuid.uuid1())
        upload = os.path.join(app.config['UPLOAD_FOLDER'], uid)

        if os.path.isdir(upload):
            abort(500, "UUID generation failed")

        os.mkdir(upload)

        return {
            'id': uid,
            'contentURL': url_for("api.verification", id=uid)
        }

    def put(self, id, jsonFormat=None):

        UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
        upload = os.path.join(UPLOAD_FOLDER, id)
        if not os.path.isdir(upload):
            abort(400, message="%s is unknown." % id)

        if 'file' not in request.files:
            abort(400, message="No file part")

        c = list(glob(os.path.join(upload, "*.c")))
        c.extend(glob(os.path.join(upload, "*.i")))

        if len(c) > 0:
            abort(400, message="File %s must be unique." % id)

        file = request.files['file']

        if file.filename == '':
            abort(400, message="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload, filename))
            return {"id": id, "created": 1}, 201

        abort(400, message="File is rejected")


class GraphGenRun(Resource):

    def get(self, id):
        UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
        upload = os.path.join(UPLOAD_FOLDER, id)
        if not os.path.isdir(upload):
            abort(400, message="%s is unknown." % id)

        c = list(glob(os.path.join(upload, "*.c")))
        c.extend(glob(os.path.join(upload, "*.i")))

        if len(c) != 1:
            abort(400, message="%s is unknown." % id)

        start = os.path.join(upload, "start.json")
        if not os.path.isfile(start):
            return {
                'id': id,
                'status': 'NOT_STARTED',
                'code': 0
            }, 200

        with open(start, "r") as i:
            start = json.load(i)

        if start['code'] == 2:
            response = os.path.join(upload, 'response.txt')
            if os.path.isfile(response):
                with open(response, "r") as i:
                    data = i.read()

                timestamp = data.split("\n")[0]
                if id in GRAPH_PROCESSES:
                    del GRAPH_PROCESSES[id]

                if data.startswith("ERROR"):
                    start = {
                        'id': id,
                        'start_time': start['start_time'],
                        'end_time': timestamp[6:],
                        'status': 'ERROR',
                        'code': -1,
                    }
                else:
                    start = {
                        'id': id,
                        'start_time': start['start_time'],
                        'end_time': timestamp,
                        'status': 'FINISHED',
                        'code': 3,
                    }

                start_path = os.path.join(upload, "start.json")
                with open(start_path, "w") as o:
                    json.dump(start, o)

                return start, 200

        return start, 200

    def put(self, id):
        UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
        upload = os.path.join(UPLOAD_FOLDER, id)
        if not os.path.isdir(upload):
            abort(400, message="%s is unknown." % id)

        c = list(glob(os.path.join(upload, "*.c")))
        c.extend(glob(os.path.join(upload, "*.i")))

        if len(c) != 1:
            abort(400, message="%s is unknown." % id)

        c = c[0]

        start = os.path.join(upload, "start.json")
        if os.path.isfile(start):
            with open(start, "r") as i:
                start = json.load(i)
            return {
                'id': id,
                'status': 'STARTED',
                'start_time': start['start_time'],
                'code': 1
            }, 200

        timestamp = str(datetime.datetime.now())
        p = Process(target=run_save_graphgen, args=(c, upload,))
        GRAPH_PROCESSES[id] = p
        print("Start graphgen")
        p.start()

        with open(start, "w") as o:
            json.dump({
                'id': id,
                'start_time': timestamp,
                'status': 'RUNNING',
                'code': 2
            }, o)

        return {
            'id': id,
            'start_time': timestamp,
            'status': 'STARTED',
            'code': 1
        }, 200


class GraphGenInfo(Resource):

    CMDS = set(['file', 'stats', 'log', 'json'])

    def get(self, id, cmd="file"):

        if len(cmd) == 0:
            cmd = "file"

        if cmd not in GraphGenInfo.CMDS:
            abort(400)

        UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
        upload = os.path.join(UPLOAD_FOLDER, id)
        if not os.path.isdir(upload):
            abort(400, message="%s is unknown." % id)

        if cmd == 'json':
            graph = os.path.join(upload, "graph.json")

            if os.path.isfile(graph):
                return flask.send_from_directory(upload, "graph.json")
            else:
                abort(400, message="Maybe an error occured?",
                      url=url_for('api.info', id=id))

        if cmd == 'file':
            graph = os.path.join(upload, "graph.json")
            response = os.path.join(upload, "response.txt")
            if os.path.isfile(graph):
                return {'id': id, 'status': 1,
                        'url': url_for('api.info', id=id, cmd='json')}
            elif os.path.isfile(response):
                return {'id': id, 'status': -1,
                        'url': url_for('api.info', id=id, cmd='log')}
            else:
                return {'id': id, 'status': 0,
                        'url': url_for('api.run', id=id)}

        if cmd == 'log':
            response = os.path.join(upload, "response.txt")

            if os.path.isfile(response):
                with open(response, "r") as i:
                    response = i.read()
                return {
                    'id': id,
                    'status': 1,
                    'content': response
                }
            else:
                return {
                    'id': id,
                    'status': 0
                }

        if cmd == 'stats':
            response = os.path.join(upload, "Statistics.txt")

            if os.path.isfile(response):
                with open(response, "r") as i:
                    response = i.read()
                return {
                    'id': id,
                    'status': 1,
                    'content': response
                }
            else:
                return {
                    'id': id,
                    'status': 0
                }

        abort(400)


api.add_resource(VerificationFile,
                 "/task/<string:id>",
                 "/task/<string:id>/",
                 "/task/<string:id>/<jsonFormat>",
                 "/task/<string:id>/<jsonFormat>/",
                 endpoint="api.verification")

api.add_resource(
    GraphGenRun,
    "/run/<string:id>",
    "/run/<string:id>/",
    endpoint="api.run"
)

api.add_resource(
    GraphGenInfo,
    "/graph/<string:id>",
    "/graph/<string:id>/",
    "/graph/<string:id>/<string:cmd>",
    "/graph/<string:id>/<string:cmd>/",
    endpoint='api.info'
)


if __name__ == '__main__':
    app.run(debug=True)
