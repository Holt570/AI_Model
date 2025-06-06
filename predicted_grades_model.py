import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

model = RandomForestRegressor()
df = pd.DataFrame()
encoder = LabelEncoder()
scaler = MinMaxScaler()

# Preprocesses data ready for training and predictions
def preprocess_data(df):

    # calculate means to fill missing values
    social_media_mean = df["social_media_hours"].mean()
    netflix_mean = df["netflix_hours"].mean()
    attendance_mean = df["attendance_percentage"].mean()
    sleep_mean = df["sleep_hours"].mean()
    exercise_mean = df["exercise_frequency"].mean()
    mental_health_mean = df["mental_health_rating"].mean()

    #fill empty cells with avaraged or default values
    df.fillna({"age": 20}, inplace=True)
    df.fillna({"gender": "Other"}, inplace=True)
    df.fillna({"social_media_hours": social_media_mean}, inplace=True)
    df.fillna({"netflix_hours": netflix_mean}, inplace=True)
    df.fillna({"part_time_job": "No"}, inplace=True)
    df.fillna({"attendance_percentage": attendance_mean}, inplace=True)
    df.fillna({"sleep_hours": sleep_mean}, inplace=True)
    df.fillna({"diet_quality": "Fair"}, inplace=True)
    df.fillna({"exercise_frequency": exercise_mean}, inplace=True)
    df.fillna({"parental_education_level": "High School"}, inplace=True)
    df.fillna({"internet_quality": "Average"}, inplace=True)
    df.fillna({"mental_health_rating": mental_health_mean}, inplace=True)
    df.fillna({"extracurricular_participation": "No"}, inplace=True)

    # remove rows without a student ID number
    df.dropna(subset = ["student_id"], inplace = True)

    # correct invalid values
    for x in df.index:
        if df.loc[x, "study_hours_per_day"] == "varies":
            df.loc[x, "study_hours_per_day"] = 3

        if df.loc[x, "age"] == "unknown":
            df.loc[x, "age"] = 20

        if df.loc[x, "exam_score"] > 100:
            df.loc[x, "exam_score"] = 100

    df.fillna({"study_hours_per_day": 3}, inplace=True)

    # apply label encoding to non numerical values
    df['gender']= encoder.fit_transform(df['gender'])
    df['part_time_job']= encoder.fit_transform(df['part_time_job'])
    df['diet_quality']= encoder.fit_transform(df['diet_quality'])
    df['parental_education_level']= encoder.fit_transform(df['parental_education_level'])
    df['internet_quality']= encoder.fit_transform(df['internet_quality'])
    df['extracurricular_participation']= encoder.fit_transform(df['extracurricular_participation'])

    #scaler.fit(df[["age","gender","study_hours_per_day","social_media_hours","netflix_hours","part_time_job","attendance_percentage","sleep_hours","diet_quality","exercise_frequency","parental_education_level","internet_quality","mental_health_rating","extracurricular_participation"]])
    #scaled = scaler.fit_transform(df[["age","gender","study_hours_per_day","social_media_hours","netflix_hours","part_time_job","attendance_percentage","sleep_hours","diet_quality","exercise_frequency","parental_education_level","internet_quality","mental_health_rating","extracurricular_participation"]])
    #normal_df = pd.DataFrame(scaled, columns=["age","gender","study_hours_per_day","social_media_hours","netflix_hours","part_time_job","attendance_percentage","sleep_hours","diet_quality","exercise_frequency","parental_education_level","internet_quality","mental_health_rating","extracurricular_participation"])

    result_text.insert(tk.END, f"Cleaning complete.")

    for x in df.index:
        result_text.insert(tk.END, df.loc[x])

    return df
    

# Opens a dialog for the user to select a file, then loads the file into a dataframe if the file extension is .csv, .xlsx or .xls
def load_dataset(df):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            messagebox.showinfo("Success", "Dataset loaded successfully. Cleaning...")
            df = preprocess_data(df)
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

# Trains the Random Forest Regressor Model using the user input dataset, features and target
def train_model(model, df, features, target):
    try:
        x = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
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
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(model, df, features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Creates button in GUI that calls make_predictions function
predict_button = tk.Button(root, text="Make Predictions", command=lambda: make_predictions(model, df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Creates output text box in GUI
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Runs GUI on running program
root.mainloop()

