from flask import Flask, send_from_directory, jsonify
import os
import json

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/api/summary")
def get_summary():
    summary_path = os.path.join("outputs", "d2tf_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Summary not found"})

@app.route("/api/card/<card_type>/<int:nfl_id>")
def get_card(card_type, nfl_id):
    # card_type can be 'scouting' or 'research'
    folder_name = "scouting_cards" if card_type == "scouting" else "research_cards"
    card_path = os.path.join("outputs", folder_name, f"{nfl_id}.json")
    if os.path.exists(card_path):
        with open(card_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Card not found"}), 404

@app.route("/d2tf_scout_payload.json")
def get_payload():
    payload_path = os.path.join("outputs", "d2tf_scout_payload.json")
    if os.path.exists(payload_path):
        with open(payload_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"meta": {"mode_message": "Payload building..."}, "players": [], "map": []})

@app.route("/<path:path>")
def static_proxy(path):
    target = os.path.join("frontend", path)
    if os.path.exists(target):
        return send_from_directory("frontend", path)
    return send_from_directory("outputs", path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
