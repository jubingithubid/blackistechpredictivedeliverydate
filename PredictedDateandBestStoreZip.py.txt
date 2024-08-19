import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
from colorama import Fore, Style
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = 'C:/Personal/AI/PredictiveDeliveryDateAndRouteOptimization/trainingData.csv'
data = pd.read_csv(data_path)
us_zips = pd.read_csv('C:/Personal/AI/PredictiveDeliveryDateAndRouteOptimization/uszips.csv')

# Plotting histograms for all numeric variables
numeric_columns = data.select_dtypes(include=np.number).columns
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Heatmap for correlation analysis
numeric_data = data.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# One-hot encode the categorical features and prepare the data
categorical_features = ['DayOfWeek', 'TimeOfDay', 'Season', 'ItemID', 'ShippingOption']
data_encoded = pd.get_dummies(data, columns=categorical_features)
data_encoded = data_encoded.astype(float)

# Prepare the features and labels
features = data_encoded.drop(['CustomerZipCode', 'StoreZipCode', 'FulfillmentSuccess'], axis=1)
labels = data_encoded['FulfillmentSuccess'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.values, labels.values, test_size=0.2, random_state=42)
X_train = np.ascontiguousarray(X_train)
X_test = np.ascontiguousarray(X_test)

# Initialize and train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Function to calculate delivery date based on shipping option, distance, and prediction
def estimate_delivery_date(current_date, shipping_option, distance, prediction):
    if shipping_option == 'Overnight':
        return current_date + timedelta(days=1)
    elif shipping_option == 'Expedited':
        return current_date + timedelta(days=2)  # Adjusted from 3 to 2 days for expedited shipping
    else:  # Standard shipping
        if prediction > 0.5:  # Adjust this threshold as needed
            if distance <= 50:  # Adjust this threshold as needed
                return current_date + timedelta(days=3)
            elif distance <= 100:
                return current_date + timedelta(days=4)
            else:
                return current_date + timedelta(days=5)
        else:
            if distance <= 50:  # Adjust this threshold as needed
                return current_date + timedelta(days=4)  # Adjusted from 5 to 4 days
            elif distance <= 100:
                return current_date + timedelta(days=5)  # Adjusted from 6 to 5 days
            else:
                return current_date + timedelta(days=6)  # Adjusted from 7 to 6 days


# Specify the quantity of Item2
requested_quantity = 100  # You can change this to the desired quantity

# Input shipping option (modify as needed)
shipping_option = ''

# Finding inventory for Item2 at the nearest suitable store
customer_zip_code = 80517
customer_location = us_zips[us_zips['zip'] == customer_zip_code][['lat', 'lng']].iloc[0].apply(radians)
us_zips_rad = us_zips[['lat', 'lng']].copy()
us_zips_rad['lat'] = us_zips_rad['lat'].map(radians)
us_zips_rad['lng'] = us_zips_rad['lng'].map(radians)
distances = haversine_distances(customer_location.to_numpy().reshape(1, -1), us_zips_rad).flatten() * 6371000 / 1609.34  # Convert to miles

# Rank stores by distance and check inventory sequentially
sorted_store_indices = np.argsort(distances)
for idx in sorted_store_indices:
    store_zip = us_zips.iloc[idx]['zip']
    inventory_filter = (data['ItemID'] == 'Item2') & (data['Inventory'] >= requested_quantity) & (data['StoreZipCode'] == store_zip)
    nearest_inventory = data[inventory_filter]
    if not nearest_inventory.empty:
        store_city = us_zips[us_zips['zip'] == store_zip]['city'].iloc[0]
        distance = distances[idx]
        print(Fore.GREEN + "Inventory Information" + Style.RESET_ALL)
        print(tabulate([["CustomerZip", "ReqQty", "AvailableQty", "StoreZip", "City", "Distance to Customer(miles)", "Shipping Option"],
                        [customer_zip_code, requested_quantity, nearest_inventory['Inventory'].iloc[0], store_zip, store_city, f"{distance:.2f}", shipping_option]], headers='firstrow', tablefmt='grid'))
        
        # Prepare test data for prediction
        test_features = pd.get_dummies(nearest_inventory.drop(['CustomerZipCode', 'StoreZipCode', 'FulfillmentSuccess'], axis=1))
        test_features = test_features.reindex(columns=features.columns, fill_value=0).astype(float)  # Align columns with training data
        test_features = np.ascontiguousarray(test_features.values)  # Ensure C-contiguity
        
        today = datetime.now()
        predictions = np.mean([model.predict_proba(test_features)[:, 1] for model in models.values()], axis=0)  # Average predictions across models
        average_prediction = np.mean(predictions)
        
        print(Fore.BLUE + "Estimated Delivery Dates and MAE for Different Models" + Style.RESET_ALL)
        table_data = []
        for name, model in models.items():
            y_pred = model.predict(test_features)
            if not shipping_option:
                # If shipping option is not provided, use the most frequent shipping option from the training data
                shipping_option = data['ShippingOption'].mode()[0]
            estimated_date = estimate_delivery_date(today, shipping_option, distance, average_prediction)  # Use input shipping option or the most frequent one
            
            # Ensure y_pred has the same length as y_test
            y_pred = np.array([y_pred] * len(y_test))
            
            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)
            table_data.append([name, estimated_date.strftime("%Y-%m-%d"), f"{mae:.2f}"])
        
        print(tabulate(table_data, headers=['Model', 'Predicted Delivery Date', 'MAE'], tablefmt='grid'))
        break
else:
    print("No available inventory for Item2 at any nearby store.")
