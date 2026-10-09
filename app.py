from flask import Flask, send_from_directory, jsonify
import os
import json
import urllib.request

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

DATA_URL = os.environ.get("DATA_PAYLOAD_URL", "")

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/d2tf_scout_payload.json")
def get_payload():
    payload_path = os.path.join("outputs", "d2tf_scout_payload.json")
    
    # If a remote URL is provided, fetch it dynamically
    if DATA_URL and not os.path.exists(payload_path):
        try:
            os.makedirs("outputs", exist_ok=True)
            urllib.request.urlretrieve(DATA_URL, payload_path)
        except Exception as e:
            print(f"Error downloading remote payload: {e}")

    if os.path.exists(payload_path):
        with open(payload_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
            
    return jsonify({
        "meta": {"mode_message": "Waiting for dataset payload connection...", "score_label": "Score scale 0-1"},
        "players": [],
        "map": []
    })

@app.route("/<path:path>")
def static_proxy(path):
    target = os.path.join("frontend", path)
    if os.path.exists(target):
        return send_from_directory("frontend", path)
    return send_from_directory("outputs", path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
