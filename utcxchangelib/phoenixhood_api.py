from flask import Flask, request, Response, jsonify
from flask_sse import sse
from flask_cors import CORS
from redis import Redis
import json
import logging
import time
import traceback

def create_api(client, symbology):
    """
    API for react application to interact with the XChangeClient struct.

    This way all trades will go through the one client that has been connected to the exchange.
    """

    app = Flask(__name__)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    CORS(app, supports_credentials=True)

    app.config["REDIS_URL"] = "redis://localhost"
    redis = Redis.from_url(app.config["REDIS_URL"], port=6379)

    app.register_blueprint(sse, url_prefix='/jms')

    try:
        redis.ping()
        log.info("Connected to Redis!")
    except redis.ConnectionError as e:
        log.error("Redis connection error:", e)

    def news_stream():

        pubsub = redis.pubsub(ignore_subscribe_messages=False)
        pubsub.subscribe("news")

        while True:
            message = pubsub.get_message()
            if message and message['type'] == 'subscribe':
                break
            time.sleep(0.001)

        while True:
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                data = message['data'].decode('utf-8')
                yield f"data: {data}\n\n"
            time.sleep(0.001)

    @app.route("/news")
    def stream_news():

        return Response(news_stream(), mimetype='text/event-stream')


    def positions_stream():

        pubsub = redis.pubsub(ignore_subscribe_messages=False)

        for symb in symbology:
            pubsub.subscribe(f"{symb}_positions")

        pubsub.subscribe("Cash_positions")

        while True:
            message = pubsub.get_message()
            if message and message['type'] == 'subscribe':
                break
            time.sleep(0.001)

        while True:
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                data = message['data'].decode('utf-8')
                yield f"data: {data}\n\n"
            time.sleep(0.001)

    @app.route("/positions_update")
    def stream_positions_update():

        return Response(positions_stream(), mimetype='text/event-stream')


    def book_stream():

        pubsub = redis.pubsub(ignore_subscribe_messages=False)

        for symb in symbology:
            pubsub.subscribe(symb)

        while True:
            message = pubsub.get_message()
            if message and message['type'] == 'subscribe':
                break
            time.sleep(0.01)

        while True:
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                data = message['data'].decode('utf-8')
                yield f"data: {data}\n\n"
            time.sleep(0.01)

    @app.route("/book_update")
    def stream_book_update():

        return Response(book_stream(), mimetype='text/event-stream')

    @app.route("/place_order", methods=['POST'])
    async def handle_place_order():
        """
        Handler such as this should handle requests from the react app.
        Maybe implement some kind of security here to ensure it is only
        coming from the app.
        """

        if request.method == 'POST':

            data = request.get_json()

            try:
                data['px'] = int(data['px'])
                data['qty'] = int(data['qty'])

                # Creates a market order if price is -1.
                tmp = {}
                for key in data.keys():
                    if key != 'px' or data[key] != -1:
                        tmp[key] = data[key]

                to_put = {'type': "Order", "data": tmp}

                client.to_exchange_queue.put(to_put)

                return jsonify({"status": "success"})

            except Exception as e:
                logging.error(traceback.format_exc())

    @app.route("/place_swap", methods=['POST'])
    def handle_place_swap():
        """
        Will place a swap order as a result of manual c/r.
        """

        if request.method == 'POST':

            data = request.get_json()

            try:

                to_put = {'type': "Swap", "data": data}
                
                client.to_exchange_queue.put(to_put)

                return jsonify({"status": "success"})

            except Exception as e:
                logging.error(traceback.format_exc())

    @app.route("/cancel_orders", methods=['POST'])
    def handle_cancel_orders():
        """"
        Will cancel all open orders if receives a POST request.
        """

        if request.method == 'POST':

            try:

                to_put = {'type': "Cancel"}

                client.to_exchange_queue.put(to_put)

                return jsonify({"status": "success"})
            
            except Exception as e:
                logging.error(traceback.format_exc())
            
    @app.route("/updates", methods=['POST'])
    def handle_updates():
        """
        This should be the only handler needed for information coming from
        the client. All updates can be handled here, the specific type being
        specified in the update_type parameter of the POST request.
        """

        if request.method == 'POST':

            data = request.get_json()

            if data['update_type'] == "news_release":

                redis.publish(channel='news', message=json.dumps(data['data']))

                return jsonify({"status": "success"})

            if data['update_type'] == "position_update":

                symb = data["symbol"]
                if symb != "Cash":
                    pos = client.positions[symb]
                else:
                    pos = client.positions['cash']
                positions = {"symb": symb, "position": pos }
                redis.publish(channel=f"{symb}_positions", message=json.dumps(positions))

                return jsonify({"status": "success"})

            if data['update_type'] == "position_snapshot":

                for symb in symbology:

                    pos = client.positions[symb]
                    positions = {"symb": symb, "position": pos }
                    redis.publish(channel=f"{symb}_positions", message=json.dumps(positions))

                    print(positions)

                pos = client.positions['cash']
                positions = {"symb": "Cash", "position": pos }
                redis.publish(channel=f"Cash_positions", message=json.dumps(positions))

                return jsonify({"status": "success"})
            
            if data['update_type'] == "book_update":

                symb = data['symbol']
                is_bid = data['is_bid']
                book = client.order_books[symb].bids if is_bid else client.order_books[symb].asks

                message_json = {"symb": symb, "is_bid": is_bid, "book": book}

                redis.publish(channel=symb, message=json.dumps(message_json))

                return jsonify({"status": "success"})
            
            if data['update_type'] == "book_snapshot":

                symb = data['symbol']
                bids = {"symb": symb, "is_bid": True, "book": client.order_books[symb].bids}
                asks = {"symb": symb, "is_bid": False, "book": client.order_books[symb].asks}
                redis.publish(channel=symb, message=json.dumps(bids))
                redis.publish(channel=symb, message=json.dumps(asks))

                return jsonify({"status": "success"})

    return app