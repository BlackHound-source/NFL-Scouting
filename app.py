from flask import Flask, send_from_directory, jsonify
import os
import json
import zipfile

app = Flask(__name__, static_folder="frontend", template_folder="frontend")

OUTPUTS_DIR = "outputs"

# Automatically extract zip files if present
if os.path.exists(OUTPUTS_DIR):
    for item in os.listdir(OUTPUTS_DIR):
        if item.endswith(".zip"):
            zip_path = os.path.join(OUTPUTS_DIR, item)
            extract_to = os.path.join(OUTPUTS_DIR, item.replace(".zip", ""))
            if not os.path.exists(extract_to):
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)

def load_all_players():
    players = []
    seen_ids = set()
    
    # 1. Try loading from individual card folders (scouting_cards / research_cards)
    for folder_name in ["scouting_cards", "research_cards"]:
        folder_path = os.path.join(OUTPUTS_DIR, folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            card_data = json.load(f)
                            # Ensure it has basic player identifiers
                            pid = card_data.get("nfl_id")
                            if pid and pid not in seen_ids:
                                seen_ids.add(pid)
                                # Normalize structure if it's a raw card format
                                if "pred_rr" not in card_data:
                                    card_data["pred_rr"] = 0.75  # fallback score
                                if "position" not in card_data:
                                    card_data["position"] = "UNK"
                                if "name" not in card_data:
                                    card_data["name"] = f"Player {pid}"
                                if "kind" not in card_data:
                                    card_data["kind"] = "prospect"
                                if "coverage" not in card_data:
                                    card_data["coverage"] = 0.8
                                if "confidence" not in card_data:
                                    card_data["confidence"] = "medium"
                                if "pos_rank" not in card_data:
                                    card_data["pos_rank"] = 1
                                if "pos_n" not in card_data:
                                    card_data["pos_n"] = 10
                                if "duel_metrics" not in card_data:
                                    card_data["duel_metrics"] = {}
                                players.append(card_data)
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    # 2. Fallback to d2tf_scout_payload.json if folders are empty
    payload_path = os.path.join(OUTPUTS_DIR, "d2tf_scout_payload.json")
    if not players and os.path.exists(payload_path):
        try:
            with open(payload_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception:
            pass

    return {
        "meta": {
            "mode": "validated",
            "mode_message": f"Loaded {len(players)} player cards dynamically from repository folders.",
            "score_label": "Score scale 0-1"
        },
        "players": players,
        "map": []
    }

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/d2tf_scout_payload.json")
def get_payload():
    return jsonify(load_all_players())

@app.route("/<path:path>")
def static_proxy(path):
    target = os.path.join("frontend", path)
    if os.path.exists(target):
        return send_from_directory("frontend", path)
    return send_from_directory("outputs", path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
