import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. Load dataset
df = pd.read_csv("iris.csv")

# 2. Clean dataset: remove 'Unnamed' and 'Id' columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# 3. Detect target column automatically
target_col = None
for col in df.columns:
    if df[col].dtype == object and col.lower() in ["species", "class", "variety"]:
        target_col = col
        break

if target_col is None:
    raise ValueError("❌ Could not find target column (species/class/variety) in CSV")

# 4. Drop rows with missing values
df = df.dropna()

# 5. Encode target
le = LabelEncoder()
df["target"] = le.fit_transform(df[target_col])

# 6. Detect numeric feature columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "target"]

X = df[numeric_cols]
y = df["target"]

# 7. Split dataset and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 9. Evaluate model
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 10. Dynamic user input for prediction
print("\nEnter flower measurements to predict species:")
try:
    sepal_length = float(input("Sepal length: "))
    sepal_width  = float(input("Sepal width: "))
    petal_length = float(input("Petal length: "))
    petal_width  = float(input("Petal width: "))

    user_sample = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Pad extra numeric columns if dataset has more than 4 features
    if X.shape[1] > 4:
        user_sample[0] += [0]*(X.shape[1]-4)

    # Convert to DataFrame with same column names
    user_sample_df = pd.DataFrame(user_sample, columns=X.columns)
    user_sample_scaled = scaler.transform(user_sample_df)

    pred = model.predict(user_sample_scaled)
    print("\nPredicted Species:", le.inverse_transform(pred)[0])

except ValueError:
    print("❌ Invalid input! Please enter numeric values only.")
