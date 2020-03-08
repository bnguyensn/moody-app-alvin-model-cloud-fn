from flask import Flask, request
from markupsafe import escape

app = Flask(__name__)


@app.route('/')
def ping():
    return 'Flask server running normally'


@app.route('/message')
def get_sentiment():
    message = request.args.get('m')

    try:
        if message is None or len(message) == 0:
            return 'Message is empty', 401

        return escape(message)

        # Run the message through the sentiment function
        # sentiment = get_sentiment(message)
        # return sentiment
    except Exception as err:
        return f'{err}', 401

