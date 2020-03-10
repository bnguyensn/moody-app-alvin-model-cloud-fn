import os
from pathlib import Path
import pickle
from flask import Flask, request, make_response
from markupsafe import escape
from main import moody, load_models
from flask_cors import CORS, cross_origin

# Define environment variables
os.environ['EMBEDDINGS_FILENAME'] = 'model/embeddings.txt'
os.environ['EMBEDDINGS_POSTDOC_FILENAME'] = 'model/embeddings_postdoc.pickle'
os.environ['INITIAL_MODEL_FILENAME'] = 'model/initial_model.sav'
os.environ['FINAL_MODEL_FILENAME'] = 'model/final_model.sav'
os.environ['IS_CLOUD_FN'] = 'false'


# Load models
# def load_models():
#     try:
#         print('Loading initial model...')
#         f1 = open(Path(os.environ['INITIAL_MODEL_FILENAME']), 'rb')
#         model_initial = pickle.load(f1)
#         f1.close()
#
#         print('Loading final model...')
#         f2 = open(Path(os.environ['FINAL_MODEL_FILENAME']), 'rb')
#         model_final = pickle.load(f2)
#         f2.close()
#
#         print('Loading embeddings...')
#         f3 = open(Path(os.environ['EMBEDDINGS_POSTDOC_FILENAME']), 'rb')
#         embeddings = pickle.load(f3)
#         f3.close()
#
#         return model_initial, model_final, embeddings
#     except Exception as err:
#         print(err)


# Initialize the Flask server
def create_app():
    # Create the application
    app = Flask(__name__)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'

    # Load model files
    # model_files = load_models()
    load_models()

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def ping():
        res = make_response('Flask server running normally')
        res.headers.add("Access-Control-Allow-Origin", "*")
        res.headers.add('Access-Control-Allow-Headers', "*")
        res.headers.add('Access-Control-Allow-Methods', "*")
        return res

    @app.route('/message')
    def get_sentiment():
        message = request.args.get('m')

        try:
            if message is None or len(message) == 0:
                res = make_response('Message is empty')
                res.headers.add("Access-Control-Allow-Origin", "*")
                res.headers.add('Access-Control-Allow-Headers', "*")
                res.headers.add('Access-Control-Allow-Methods', "*")
                return res, 401

            # Run the message through the sentiment function, then return the
            # sentiment
            res = make_response({
                'data': moody(message)
            })
            res.headers.add("Access-Control-Allow-Origin", "*")
            res.headers.add('Access-Control-Allow-Headers', "*")
            res.headers.add('Access-Control-Allow-Methods', "*")
            return res
        except Exception as err:
            res = make_response(f'{err}')
            res.headers.add("Access-Control-Allow-Origin", "*")
            res.headers.add('Access-Control-Allow-Headers', "*")
            res.headers.add('Access-Control-Allow-Methods', "*")
            return res, 401

    return app


app = create_app()
