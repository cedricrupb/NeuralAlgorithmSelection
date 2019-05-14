from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):

    def get(self, name):
        return {"message": "Hello %s" % name}


api.add_resource(HelloWorld, "/<string:name>")


if __name__ == '__main__':
    app.run(debug=True)
