from flask import Flask, render_template, request, jsonify
from hybrid_model import hybrid_predict
from language_api import translate
from scan_api import scan_predict

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json.get("features", [])
        lang = request.json.get("lang", "en")

        
        pred = hybrid_predict(data)

       
        if pred < 0.4:
            result = "Low Risk ❤️"
        elif pred < 0.7:
            result = "Medium Risk ⚠️"
        else:
            result = "High Risk 🚨"

       
        result = translate(result, lang)

        return jsonify({
            "result": result,
            "risk": round(pred * 100, 2)
        })

    except Exception as e:
       return jsonify({
    "scan_result": result,
    "risk": round(pred*100,2)
})




@app.route("/scan", methods=["POST"])
def scan():

    try:
        file = request.files.get("image")

        if not file:
            return jsonify({"scan_result": "No Image Uploaded"})

        pred = scan_predict(file)

      
        if pred < 0.4:
            result = "Low Risk ❤️"
        elif pred < 0.7:
            result = "Medium Risk ⚠️"
        else:
            result = "High Risk 🚨"

        return jsonify({
            "scan_result": result,
            "risk": round(pred * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "scan_result": "Scan Error",
            "error": str(e)
        })




if __name__ == "__main__":
    app.run(debug=True)
