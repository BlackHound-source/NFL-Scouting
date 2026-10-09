from flask import Flask, send_from_directory, jsonify
import os
import json
import zipfile

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

# Automatically extract any zip files found in outputs/ on startup
OUTPUTS_DIR = "outputs"
if os.path.exists(OUTPUTS_DIR):
    for item in os.listdir(OUTPUTS_DIR):
        if item.endswith(".zip"):
            zip_path = os.path.join(OUTPUTS_DIR, item)
            extract_to = os.path.join(OUTPUTS_DIR, item.replace(".zip", ""))
            if not os.path.exists(extract_to):
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                print(f"Extracted {item} to {extract_to}")

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/api/card/<card_type>/<int:nfl_id>")
def get_card(card_type, nfl_id):
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
