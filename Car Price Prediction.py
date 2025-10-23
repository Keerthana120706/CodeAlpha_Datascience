# ==========================================================
# 🚗 Car Price Prediction using Machine Learning
# Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
# ==========================================================

# 📦 Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# STEP 1: Load Dataset
# ==========================================================
print(" Loading dataset...")
df = pd.read_csv("Car data.csv")   # 👉 Change filename if needed

print("\n Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print("\nSample data:\n", df.head())

# ==========================================================
# STEP 2: Dataset Info & Missing Values
# ==========================================================
print("\nDataset Info:")
print(df.info())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Drop duplicates (if any)
df = df.drop_duplicates()
print("\nAfter cleaning, new shape:", df.shape)

# ==========================================================
# STEP 3: Encode Categorical Columns
# ==========================================================
label_enc = LabelEncoder()
cat_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']

for col in cat_cols:
    df[col] = label_enc.fit_transform(df[col])

print("\nCategorical columns encoded:", cat_cols)
print(df.head())

# ==========================================================
# STEP 4: Feature and Target Split
# ==========================================================
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# ==========================================================
# STEP 5: Train-Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split done:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==========================================================
# STEP 6: Model Training
# ==========================================================
print("\n Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)

rf_model.fit(X_train, y_train)
print(" Model training complete!")

# ==========================================================
# STEP 7: Predictions
# ==========================================================
y_pred = rf_model.predict(X_test)

# ==========================================================
# STEP 8: Model Evaluation
# ==========================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# ==========================================================
# STEP 9: Visualization
# ==========================================================
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, color="blue")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()

# Feature Importance Plot
plt.figure(figsize=(8,5))
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh', color='orange')
plt.title("Feature Importance in Car Price Prediction")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

print("\n Task Completed Successfully! ")
