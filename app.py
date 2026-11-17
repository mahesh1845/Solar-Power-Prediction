"""Flask backend for the Solar Power Generation Prediction System.
Run:  python app.py   then open http://127.0.0.1:5000"""
import csv
import io
import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.pkl")
HISTORY_PATH = os.path.join(BASE, "history.csv")
HISTORY_COLS = ["timestamp", "temperature", "humidity", "irradiation", "wind_speed",
                "power_kw", "daily_kwh", "efficiency_pct", "savings", "co2_kg"]

PANEL_AREA_M2 = 28.0      # used for efficiency estimate
CO2_KG_PER_KWH = 0.82     # approx. grid emission factor (India)

app = Flask(__name__, static_folder=None)  # only whitelisted files are served

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("model.pkl not found - run: python train.py")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# Always answer with JSON (even on errors) so the browser never sees an empty body.
@app.errorhandler(Exception)
def handle_error(e):
    code = getattr(e, "code", 500)
    return jsonify(error=str(e)), code if isinstance(code, int) else 500


# ---------- pages ----------
@app.route("/")
@app.route("/index.html")
def home():
    return send_from_directory(BASE, "index.html")


@app.route("/style.css")
@app.route("/script.js")
def assets():
    return send_from_directory(BASE, request.path.lstrip("/"))


@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard():
    return send_from_directory(BASE, "dashboard.html")


@app.route("/about")
@app.route("/about.html")
def about():
    return send_from_directory(BASE, "about.html")


# ---------- helpers ----------
def num(data, key, lo, hi, default=None):
    raw = data.get(key, default)
    if raw in (None, ""):
        raise ValueError(f"'{key}' is required")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}")
    return v


def read_history():
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=HISTORY_COLS)
    return pd.read_csv(HISTORY_PATH)


def alerts_for(temp, irr, power, capacity):
    out = []
    if power < 0.2 * capacity:
        out.append({"level": "warn", "text": "Low solar generation expected - consider shifting heavy loads to the grid or battery."})
    if temp >= 40:
        out.append({"level": "danger", "text": "High temperature warning - panel efficiency drops about 0.4% per degree C above 25."})
    if irr >= 800 and temp < 40:
        out.append({"level": "ok", "text": "Strong irradiation - excellent conditions for generation."})
    return out


def recommendation(power, capacity, temp):
    ratio = power / capacity if capacity else 0
    if ratio >= 0.6:
        return "Great generation window: run washing machines, pumps and EV charging now."
    if ratio >= 0.3:
        return "Moderate generation: run essential appliances during 11:00-15:00."
    if temp >= 40:
        return "Output is limited by heat. Keep panels clean and ensure airflow behind them."
    return "Low generation: rely on grid/battery and re-check when irradiation improves."


# ---------- API ----------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        temp = num(data, "temperature", -20, 70)
        hum = num(data, "humidity", 0, 100)
        irr = num(data, "irradiation", 0, 1500)
        wind = num(data, "wind_speed", 0, 60)
        peak = num(data, "peak_sun_hours", 0, 14, 6)
        tariff = num(data, "tariff", 0, 1000, 8)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    b = get_bundle()
    X = pd.DataFrame([[temp, hum, irr, wind]], columns=b["features"])
    power = max(0.0, float(b["model"].predict(X)[0]))
    daily = power * peak
    eff = (power * 1000) / (PANEL_AREA_M2 * irr) * 100 if irr > 0 else 0.0
    savings = daily * tariff
    co2 = daily * CO2_KG_PER_KWH

    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "temperature": temp, "humidity": hum, "irradiation": irr, "wind_speed": wind,
           "power_kw": round(power, 3), "daily_kwh": round(daily, 2),
           "efficiency_pct": round(eff, 2), "savings": round(savings, 2), "co2_kg": round(co2, 2)}
    new_file = not os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    return jsonify(power_kw=row["power_kw"], daily_kwh=row["daily_kwh"],
                   efficiency_pct=row["efficiency_pct"], savings=row["savings"],
                   co2_kg=row["co2_kg"], capacity_kw=b["capacity_kw"],
                   best_time="11:00 - 15:00 (peak sun)",
                   alerts=alerts_for(temp, irr, power, b["capacity_kw"]),
                   recommendation=recommendation(power, b["capacity_kw"], temp))


@app.route("/api/history")
def history():
    df = read_history().tail(50).iloc[::-1]
    return jsonify(df.to_dict("records"))


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    return jsonify(ok=True)


@app.route("/api/metrics")
def metrics():
    b = get_bundle()
    return jsonify(metrics=b["metrics"], comparison=b["comparison"], capacity_kw=b["capacity_kw"])


@app.route("/api/analytics")
def analytics():
    df = read_history()
    if df.empty:
        return jsonify(summary={"predictions": 0, "total_kwh": 0, "avg_kw": 0, "avg_eff": 0,
                                "avg_temp": 0, "avg_irr": 0, "savings": 0, "co2_kg": 0},
                       daily=[], weekly=[], monthly=[], series=[])
    df["ts"] = pd.to_datetime(df["timestamp"])
    summary = {"predictions": int(len(df)), "total_kwh": round(df["daily_kwh"].sum(), 2),
               "avg_kw": round(df["power_kw"].mean(), 3), "avg_eff": round(df["efficiency_pct"].mean(), 2),
               "avg_temp": round(df["temperature"].mean(), 1), "avg_irr": round(df["irradiation"].mean(), 1),
               "savings": round(df["savings"].sum(), 2), "co2_kg": round(df["co2_kg"].sum(), 2)}

    def group(freq):
        g = df.set_index("ts")["daily_kwh"].resample(freq).sum().round(2)
        return [{"label": k.strftime("%Y-%m-%d"), "kwh": float(v)} for k, v in g.items()]

    series = [{"label": t.strftime("%H:%M:%S"), "kw": float(p)} for t, p in zip(df["ts"].tail(20), df["power_kw"].tail(20))]
    return jsonify(summary=summary, daily=group("D")[-14:], weekly=group("W")[-8:],
                   monthly=group("MS")[-12:], series=series)


@app.route("/api/report.csv")
def report():
    df = read_history()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=solar_prediction_report.csv"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
