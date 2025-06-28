from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from bson import ObjectId
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# App Initialization
app = Flask(__name__)
app.secret_key = os.urandom(24)

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")
db = client["disease_prediction_db"]
users_collection = db["users"]

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Load and Train ML Model
training_df = pd.read_csv("my_custom_disease_dataset.csv").dropna(axis=1, how='all')
training_df.columns = training_df.columns.str.strip().str.lower().str.replace('[^a-z0-9_]', '', regex=True)

X = training_df.drop('prognosis', axis=1).fillna(0)
y = training_df['prognosis']
symptoms = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
nb_model = BernoulliNB().fit(X_train, y_train)

# Evaluate model
nb_accuracy = nb_model.score(X_test, y_test)
print(f"Naive Bayes Model Accuracy: {nb_accuracy:.2f}")

from collections import Counter

# Predict on test data
y_pred = nb_model.predict(X_test)

# Count frequency of each disease
most_common = Counter(y_test).most_common(5)
top_classes = [label for label, _ in most_common]

# Filter test data and predictions to only include top classes
filtered_indices = [i for i, label in enumerate(y_test) if label in top_classes]
filtered_y_test = [y_test.iloc[i] for i in filtered_indices]
filtered_y_pred = [y_pred[i] for i in filtered_indices]

# Recalculate metrics on filtered data
f1 = f1_score(filtered_y_test, filtered_y_pred, average='macro')
print(f"Filtered F1 Score (Top 5 diseases): {f1:.2f}")

# Classification report for filtered data
print("\nFiltered Classification Report (Top 5 diseases):")
print(classification_report(filtered_y_test, filtered_y_pred))

# Confusion matrix for filtered data
conf_mat = confusion_matrix(filtered_y_test, filtered_y_pred, labels=top_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=top_classes,
            yticklabels=top_classes)
plt.title("Confusion Matrix (Top 5 Diseases)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.tight_layout()
plt.show()


# Load Mapping
mapping_df = pd.read_csv("prognosis_mapping.csv").dropna()
states = sorted(mapping_df['state'].unique())

# User class
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.email = user_data["email"]

@login_manager.user_loader
def load_user(user_id):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        return User(user) if user else None
    except:
        return None

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        country = request.form['country']
        state = request.form['state']

        if not all([email, password, name, country, state]):
            flash("All fields are required!", "danger")
            return redirect(url_for('signup'))

        if users_collection.find_one({"email": email}):
            flash("User already exists!", "danger")
            return redirect(url_for('signup'))

        users_collection.insert_one({
            "email": email,
            "password": generate_password_hash(password),
            "name": name,
            "country": country,
            "state": state
        })
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_data = users_collection.find_one({"email": email})
        if user_data and check_password_hash(user_data['password'], password):
            login_user(User(user_data))
            return redirect(url_for('welcome'))
        flash("Invalid email or password.", "danger")
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/welcome')
@login_required
def welcome():
    return render_template('welcome.html', name=current_user.email.split('@')[0])

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        flash("Please select at least one symptom.", "warning")
        return redirect(url_for('dashboard'))

    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

    nb_probs = nb_model.predict_proba([input_vector])[0]
    nb_predictions = sorted(zip(nb_model.classes_, nb_probs), key=lambda x: x[1], reverse=True)[:3]
    predicted_disease = nb_model.classes_[nb_probs.argmax()]

    return render_template('prediction.html', symptoms=selected_symptoms,
                           nb_predictions=nb_predictions,
                           disease=predicted_disease, states=states)

@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    disease = request.form['disease']
    selected_state = request.form['state']

    match = mapping_df[
        (mapping_df['prognosis'].str.lower() == disease.lower()) &
        (mapping_df['state'].str.lower() == selected_state.lower())
    ]

    if not match.empty:
        specialty = match.iloc[0]['specialty']
        hospital = match.iloc[0]['hospital']
        place = match.iloc[0]['place']
        return render_template("recommendation.html",
                               disease=disease,
                               state=selected_state,
                               specialty=specialty,
                               hospital=hospital,
                               place=place)
    else:
        flash("No recommendation found for the selected state.", "info")
        return render_template("recommendation.html", disease=disease, state=selected_state)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
