from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from math import radians, sin, cos, sqrt, atan2
#from tabulate import tabulate

app = Flask(__name__)

# Haversine function to calculate distance in miles
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c * 0.621371  # Convert to miles

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Load uploaded files
        training_file = request.files['training_data']
        zip_file = request.files['us_zips']
        
        # Read data from files
        training_data = pd.read_csv(training_file)
        us_zips = pd.read_csv(zip_file)

        # Initialize the scaler and fit it on the 'Distance' column of the uploaded training data
        scaler = StandardScaler()
        training_data['Distance'] = scaler.fit_transform(training_data[['Distance']])

        # Prepare categorical feature list and one-hot encode them
        categorical_features = ['DayOfWeek', 'TimeOfDay', 'Season', 'ItemID']
        data_encoded = pd.get_dummies(training_data, columns=categorical_features)

        # Train models for each shipping option
        shipping_models = {}
        model_mae = {}  # Store MAE for each model
        for option in ['Standard', 'Expedited']:
            option_data = data_encoded[data_encoded['ShippingOption'] == option]
            features = option_data.drop(['CustomerZipCode', 'StoreZipCode', 'DaysToDeliver', 'ShippingOption'], axis=1, errors='ignore')
            labels = option_data['DaysToDeliver']
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            # Initialize models
            gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            gbr.fit(X_train, y_train)
            xgb.fit(X_train, y_train)

            # Store models and calculate MAE
            shipping_models[option] = {'gbr': gbr, 'xgb': xgb}
            model_mae[option] = {
                'gbr': mean_absolute_error(y_test, gbr.predict(X_test)),
                'xgb': mean_absolute_error(y_test, xgb.predict(X_test))
            }
        
        # Handle form inputs for prediction
        requested_item = request.form['item_id']
        requested_quantity = int(request.form['quantity'])
        customer_zip_code = int(request.form['customer_zip_code'])
        shipping_option = request.form['shipping_option']

        # Get customer location
        customer_location = us_zips[us_zips['zip'] == customer_zip_code][['lat', 'lng']].iloc[0]
        nearest_store = None
        nearest_distance = float('inf')

        # Find the nearest store with sufficient inventory for the requested item
        for _, row in training_data[(training_data['ItemID'] == requested_item) & 
                                    (training_data['Inventory'] >= requested_quantity)].iterrows():
            store_zip = row['StoreZipCode']
            store_location = us_zips[us_zips['zip'] == store_zip][['lat', 'lng']].iloc[0]
            distance = haversine(customer_location['lat'], customer_location['lng'], store_location['lat'], store_location['lng'])
            
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_store = row

        # Prepare prediction table data
        today = datetime.now()
        table_data = []

        if nearest_store is not None:
            # Transform the distance of the nearest store using the scaler
            test_data = nearest_store.drop(['CustomerZipCode', 'StoreZipCode', 'DaysToDeliver', 'ShippingOption']).to_frame().T
            test_data['Distance'] = scaler.transform([[nearest_distance]])
            test_data = pd.get_dummies(test_data)
            test_data = test_data.reindex(columns=features.columns, fill_value=0)

            # Make predictions and add results to the table
            if shipping_option in shipping_models:
                gbr_days = shipping_models[shipping_option]['gbr'].predict(test_data)[0]
                xgb_days = shipping_models[shipping_option]['xgb'].predict(test_data)[0]
                gbr_delivery_date = today + timedelta(days=int(round(gbr_days)))
                xgb_delivery_date = today + timedelta(days=int(round(xgb_days)))
                table_data.append({
                    "Model": "GBR",
                    "MAE": model_mae[shipping_option]['gbr'],
                    "Customer Zip Code": customer_zip_code,
                    "Store Zip Code": nearest_store['StoreZipCode'],
                    "Inventory": nearest_store['Inventory'],
                    "Distance (miles)": nearest_distance,
                    "Predicted Days to Deliver": gbr_days,
                    "PDD": gbr_delivery_date.strftime("%Y-%m-%d")
                })
                table_data.append({
                    "Model": "XGBoost",
                    "MAE": model_mae[shipping_option]['xgb'],
                    "Customer Zip Code": customer_zip_code,
                    "Store Zip Code": nearest_store['StoreZipCode'],
                    "Inventory": nearest_store['Inventory'],
                    "Distance (miles)": nearest_distance,
                    "Predicted Days to Deliver": xgb_days,
                    "PDD": xgb_delivery_date.strftime("%Y-%m-%d")
                })
        else:
            table_data.append({"Error": "No store with sufficient inventory found"})

        return render_template('index.html', prediction_data=table_data)

    return render_template('index.html')

if __name__ == "__main__":
       app.run(debug=True, host='0.0.0.0', port=5001)
