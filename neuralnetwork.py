# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the dataset
df = pd.read_csv('./student-mat.csv', sep=';')

# Convert final grades to binary Pass/Fail labels
df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Select features for the model
features = df[['G1', 'G2', 'studytime', 'failures', 'absences']]
target = df['pass_fail']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Create and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
y_pred_train = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)

# Function to predict pass/fail
def predict_pass_fail():
    try:
        # Get inputs from the user
        g1 = float(entry_g1.get())
        g2 = float(entry_g2.get())
        studytime = int(entry_studytime.get())
        past_failures = int(entry_failures.get())
        absent_days = int(entry_absences.get())

        # Prepare the input for prediction
        input_data = np.array([[g1, g2, studytime, past_failures, absent_days]])

        # Predict the pass/fail
        prediction = model.predict(input_data)

        # Show the result in a message box
        if prediction[0] == 1:
            messagebox.showinfo("Prediction Result", "The student is predicted to PASS.", icon='info')
        else:
            messagebox.showinfo("Prediction Result", "The student is predicted to FAIL.", icon='warning')

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.", icon='error')

# Create the main window
root = tk.Tk()
root.title("Student Pass/Fail Predictor")

# Set a background color
root.configure(bg='#f0f8ff')

# Create and place labels and entry fields
tk.Label(root, text="Enter G1 (First Term Grade: 0-20):", bg='#f0f8ff').grid(row=0, column=0, pady=10, padx=10)
entry_g1 = tk.Entry(root)
entry_g1.grid(row=0, column=1)

tk.Label(root, text="Enter G2 (Second Term Grade: 0-20):", bg='#f0f8ff').grid(row=1, column=0, pady=10, padx=10)
entry_g2 = tk.Entry(root)
entry_g2.grid(row=1, column=1)

tk.Label(root, text="Enter Study Time (1-4):", bg='#f0f8ff').grid(row=2, column=0, pady=10, padx=10)
entry_studytime = tk.Entry(root)
entry_studytime.grid(row=2, column=1)

tk.Label(root, text="Enter Number of Past Failures in Exams:", bg='#f0f8ff').grid(row=3, column=0, pady=10, padx=10)
entry_failures = tk.Entry(root)
entry_failures.grid(row=3, column=1)

tk.Label(root, text="Enter Number of Absent Days:", bg='#f0f8ff').grid(row=4, column=0, pady=10, padx=10)
entry_absences = tk.Entry(root)
entry_absences.grid(row=4, column=1)

# Create a button to make predictions
btn_predict = tk.Button(root, text="Predict Pass/Fail", command=predict_pass_fail, bg='#008CBA', fg='white', font=('Arial', 12, 'bold'))
btn_predict.grid(row=5, columnspan=2, pady=20)

# Display the model accuracy
accuracy_label = tk.Label(root, text=f"Model Accuracy: {accuracy * 100:.2f}%", bg='#f0f8ff', font=('Arial', 12, 'bold'))
accuracy_label.grid(row=6, columnspan=2)

# Instructions for using the tool
instructions = tk.Label(root, text="Instructions:\n"
                                     "1. Fill in the grades and student features.\n"
                                     "2. G1 and G2 should be between 0 and 20 (inclusive).\n"
                                     "3. Study time is on a scale rated from 1 (low) to 4 (high).\n"
                                     "4. Number of Past Failures in Exams and Absent Days are the actual counts.\n"
                                     "5. Click 'Predict Pass/Fail' to get the result.\n"
                                     "6. Ensure all inputs are valid numbers.", 
                                     bg='#f0f8ff', justify=tk.LEFT)
instructions.grid(row=7, columnspan=2)

# Run the application
root.mainloop()
