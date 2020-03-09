import os
from pathlib import Path
import pickle
from flask import Flask, request
from markupsafe import escape
from main import moody, load_models

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

    # Load model files
    # model_files = load_models()
    load_models()

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def ping():
        return 'Flask server running normally'

    @app.route('/message')
    def get_sentiment():
        message = request.args.get('m')

        try:
            if message is None or len(message) == 0:
                return 'Message is empty', 401

            # Run the message through the sentiment function, then return the
            # sentiment
            return {
                'data': moody(message)
            }
        except Exception as err:
            return f'{err}', 401

    return app


app = create_app()

