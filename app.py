from flask import Flask, render_template, request
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data if not already present
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Flask setup
app = Flask(__name__)

# Load model and preprocessors
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
# scaler = joblib.load("model/scaler.pkl")
# label_encoder_gender = joblib.load("model/label_encoder_gender.pkl")
# label_encoder_jobrole = joblib.load("model/label_encoder_jobrole.pkl")

# Static job roles list
job_roles = ['Fitness Coach', 'Physician', 'Financial Analyst', 
       'Supply Chain Manager', 'Database Administrator', 'Architect',
       'Operations Manager', 'Cybersecurity Analyst', 'Software Engineer',
       'Urban Planner', 'Machine Learning Engineer', 'Personal Trainer',
       'Biomedical Engineer', 'Nurse', 'Systems Analyst',
       'Product Manager', 'Content Writer', 'Pharmacist', 'Chef',
       'AI Researcher', 'Data Analyst', 'Psychologist', 'Civil Engineer',
       'Accountant', 'Graphic Designer', 'Web Developer',
       'Cloud Architect', 'AI Specialist', 'Dentist', 'Pilot',
       'UX Designer', 'Teacher', 'HR Specialist', 'Veterinarian',
       'Environmental Scientist', 'Legal Consultant',
       'Sales Representative', 'Robotics Engineer', 'SEO Specialist',
       'Business Analyst', 'Customer Service Representative',
       'Marketing Manager', 'Social Worker', 'Electrician', 'Journalist',
       'Event Planner', 'Lawyer', 'Mechanical Engineer',
       'Construction Manager', 'Research Scientist', 'Creative Director']

@app.route('/')
def home():
    return render_template("index.html", result=None, job_roles=job_roles)

@app.route('/predict', methods=["POST"])
def predict():
    # age = int(request.form["age"])
    # gender = request.form["gender"].strip()
    # job_role = request.form["job_role"].strip()  # Fix: strip spaces

    # print("Age:", age)
    # print("Gender:", gender)
    # print("Job Role:", job_role)

    resume_file = request.files["resume"]
    job_desc_file = request.files["job_desc"]

    print("Resume File Name:", resume_file.filename)
    print("Job Description File Name:", job_desc_file.filename)

    # Always reset file pointer before reading
    resume_file.seek(0)
    resume_text = resume_file.read().decode("utf-8", errors="ignore")

    job_desc_file.seek(0)
    job_desc_text = job_desc_file.read().decode("utf-8", errors="ignore")

    print("Resume Text:", resume_text[:500])
    print("Job Description Text:", job_desc_text[:500])

    # Preprocess
    resume_text = preprocess_text(resume_text)
    job_desc_text = preprocess_text(job_desc_text)

    # try:
    #     gender_encoded = label_encoder_gender.transform([gender]).reshape(1, -1)
    #     job_role_encoded = label_encoder_jobrole.transform([job_role]).reshape(1, -1)
    # except Exception as e:
    #     print("Encoding Error:", e)
    #     return render_template("index.html", result="❌ Unknown Gender or Job Role!", job_roles=job_roles)

    # age_scaled = scaler.transform([[age]])
    resume_vec = vectorizer.transform([resume_text]).toarray()
    job_desc_vec = vectorizer.transform([job_desc_text]).toarray()

    # features = np.hstack((age_scaled, gender_encoded, job_role_encoded, resume_vec, job_desc_vec))
    features = np.hstack((resume_vec, job_desc_vec))
    print("features",features)

    prediction = int(model.predict(features)[0])
    print("Prediction:", prediction)

    return render_template("index.html", result=prediction, job_roles=job_roles)

if __name__ == "__main__":
    app.run(debug=True)
