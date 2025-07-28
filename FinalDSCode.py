import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    df = pd.read_excel("D://SY4THSEM/DS/augmented_dataset.xlsx")  # Use your actual path

    label_encoder = LabelEncoder()
    df["Column_Size"] = label_encoder.fit_transform(df["Column_Size"])
    X = df.drop(columns=["Column_Size"])
    y = df["Column_Size"]
    return X, y, label_encoder

X_raw, y, label_encoder = load_and_preprocess_data()

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)
X = pd.DataFrame(X_scaled, columns=X_raw.columns)
class_counts = Counter(y)
valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
mask = y.isin(valid_classes)
X_filtered = X[mask]
y_filtered = y[mask]

min_class_size = min(Counter(y_filtered).values())
k_neighbors = min(5, min_class_size - 1)
smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train stacking model
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
    ('xgb',XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, use_label_encoder=False, eval_metric='mlogloss',
                   subsample=0.8, colsample_bytree=0.7, gamma=0.2, random_state=42))
]
meta_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train, y_train)

# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y_resampled))
y_score = stacking_model.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

# Prepare ROC curve data
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve
plt.figure(figsize=(10, 7))
colors = sns.color_palette("hsv", n_classes)
for i, color in zip(range(n_classes), colors):
    label = label_encoder.inverse_transform([i])[0]
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class AUC-ROC Curve (Stacking Model)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()



# Category mapping
category_mapping = {
    "P": (100, 0), "A": (125, 0), "B": (150, 0), "C": (150, 300), "D": (150, 350),
    "F": (200, 300), "E": (200, 350), "G": (200, 400), "H": (200, 450), "N": (250, 300),
    "I": (250, 350), "J": (250, 400), "K": (350, 0), "L": (400, 0), "O": (400, 400),"M": (450,400)
}

# Database setup
def setup_database():
    conn = sqlite3.connect("D:\SY4THSEM\DS\column_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS column_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pcd INTEGER NOT NULL CHECK(pcd > 0),
            weight REAL NOT NULL CHECK(weight > 0),
            elevation INTEGER NOT NULL CHECK(elevation >= 0),
            legs INTEGER NOT NULL CHECK(legs > 0),
            wind_speed INTEGER NOT NULL CHECK(wind_speed >= 0),
            seismic_zone INTEGER NOT NULL CHECK(seismic_zone BETWEEN 1 AND 5),
            mc INTEGER NOT NULL CHECK(mc >= 0),
            apart INTEGER NOT NULL CHECK(apart >= 0)
        )
    ''')
    conn.commit()
    conn.close()


# Prediction function
def predict_and_save():
    try:
        # Extract and convert input values
        pcd = float(pcd_var.get())
        weight = float(weight_var.get())
        elevation = float(elevation_var.get())
        legs = float(legs_var.get())
        wind_speed = float(wind_speed_var.get())
        seismic_zone = float(seismic_zone_var.get())

        # Prepare for prediction
        user_input_raw = np.array([[pcd, weight, elevation, legs, wind_speed, seismic_zone]])
        user_input_scaled = scaler.transform(user_input_raw)
        prediction = stacking_model.predict(user_input_scaled)
        predicted_category = label_encoder.inverse_transform(prediction)[0]

        # Get mapped values
        mc_value, apart_value = category_mapping.get(predicted_category, (0, 0))
        mc_var.set(mc_value)
        apart_var.set(apart_value)

        # Save to database
        conn = sqlite3.connect("column_data.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO column_entries (pcd, weight, elevation, legs, wind_speed, seismic_zone, mc, apart)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (pcd, weight, elevation, legs, wind_speed, seismic_zone, mc_value, apart_value))
        conn.commit()
        conn.close()

        messagebox.showinfo("Success", f"Prediction Saved!\nMC: {mc_value}, APART: {apart_value}")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")


# Tkinter UI Setup
root = tk.Tk()
root.title("Column Type Prediction")
root.configure(bg="#1C2833")
root.geometry("700x650")

style = ttk.Style()
style.configure("TLabel", font=("Arial", 17, "bold"), background="#1C2833", foreground="white")
style.configure("TButton", font=("Arial", 17, "bold"), background="#E67E22", foreground="white")
style.configure("TEntry", font=("Arial", 17))

title_label = tk.Label(root, text="Column Type Prediction", font=("Arial", 25, "bold"), fg="#E67E22", bg="#1C2833")
title_label.pack(pady=20)

def create_label_entry(frame, text, var):
    container = tk.Frame(frame, bg="#1C2833")
    container.pack(fill="x", pady=5)
    label = ttk.Label(container, text=text)
    label.pack(side="left", padx=8)
    entry = ttk.Entry(container, textvariable=var, width=25, font=("Arial", 12), justify="center")
    entry.pack(side="right", padx=8, ipadx=5, ipady=5)
    return entry

input_frame = tk.Frame(root, bg="#1C2833")
input_frame.pack(padx=15, pady=5, fill="both")

pcd_var, weight_var, elevation_var, legs_var, wind_speed_var, seismic_zone_var = [tk.StringVar() for _ in range(6)]
mc_var, apart_var = tk.StringVar(), tk.StringVar()

create_label_entry(input_frame, "PCD (mm):", pcd_var)
create_label_entry(input_frame, "Weight (Ton):", weight_var)
create_label_entry(input_frame, "Elevation (mm):", elevation_var)
create_label_entry(input_frame, "No. of Legs:", legs_var)
create_label_entry(input_frame, "Basic Wind Speed (m/s):", wind_speed_var)
create_label_entry(input_frame, "Seismic Zone:", seismic_zone_var)

mc_entry = create_label_entry(input_frame, "MC:", mc_var)
apart_entry = create_label_entry(input_frame, "APART:", apart_var)
mc_entry.config(state="readonly")
apart_entry.config(state="readonly")

predict_button = tk.Button(root, text="Predict & Save", command=predict_and_save, font=("Arial", 22, "bold"),
                           bg="#E67E22", fg="white", padx=5, pady=5, bd=10, relief="raised")
predict_button.pack(pady=40)

root.mainloop()