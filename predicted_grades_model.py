import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

model = RandomForestRegressor()
df = pd.DataFrame()

#add funtion


# Opens a dialog for the user to select a file, then loads the file into a dataframe if the file extension is .csv, .xlsx or .xls
def load_dataset(df):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            messagebox.showinfo("Success", "Dataset loaded successfully *but did you check the script!")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

# Trains the Random Forest Regressor Model using the user input dataset, features and target
def train_model(df, features, target):
    try:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

# Makes prediction for target used in training based on specified features
def make_predictions(model, df, features):
    try:
        X_new = df[features]
        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions:\n{predictions}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# Creates GUI window
root = tk.Tk()
root.title("Student Predictive Grades")

# Creates button in GUI that calls load_dataset function
load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset(df))
load_button.pack(pady=10)

# Adds textbox and label for user to specify features
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Adds textbox and label for user to specify target
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Creates button in GUI that calls train_model function
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(df, features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Creates button in GUI that calls make_predictions function
predict_button = tk.Button(root, text="Make Predictions", command=lambda: make_predictions(model, df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Creates output text box in GUI
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Runs GUI on running program
root.mainloop()

