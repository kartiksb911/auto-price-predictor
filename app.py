from flask import Flask, render_template, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        brand = request.form['brand']
        model = request.form['model']
        vehicle_age = int(request.form['vehicle_age'])
        km_driven = int(request.form['km_driven'])
        seller_type = request.form['seller_type']
        fuel_type = request.form['fuel_type']
        transmission_type = request.form['transmission_type']
        mileage = float(request.form['mileage'])
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = int(request.form['seats'])

        custom_data = CustomData(brand, model, vehicle_age, km_driven, seller_type, fuel_type, 
                                 transmission_type, mileage, engine, max_power, seats)
        features = custom_data.get_data_as_dataFrame()

        prediction = PredictPipeline().predict(features)

        return render_template('index.html', prediction_text=f'Predicted Selling Price: ₹{prediction[0]:,.2f}')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        brand = data.get('brand')
        model = data.get('model')
        vehicle_age = int(data.get('vehicle_age'))
        km_driven = int(data.get('km_driven'))
        seller_type = data.get('seller_type')
        fuel_type = data.get('fuel_type')
        transmission_type = data.get('transmission_type')
        mileage = float(data.get('mileage'))
        engine = float(data.get('engine'))
        max_power = float(data.get('max_power'))
        seats = int(data.get('seats'))

        custom_data = CustomData(brand, model, vehicle_age, km_driven, seller_type, fuel_type, 
                                 transmission_type, mileage, engine, max_power, seats)
        features = custom_data.get_data_as_dataFrame()

        prediction = PredictPipeline().predict(features)

        return jsonify({"predicted_selling_price": f"₹{prediction[0]:,.2f}"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
