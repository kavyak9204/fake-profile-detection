from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/scam_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        username_length = int(request.form['username_length'])
        username_has_number = int(request.form['username_has_number'])
        full_name_has_number = int(request.form['full_name_has_number'])
        full_name_length = int(request.form['full_name_length'])
        is_private = int(request.form['is_private'])
        is_joined_recently = int(request.form['is_joined_recently'])
        has_channel = int(request.form['has_channel'])
        is_business_account = int(request.form['is_business_account'])
        has_guides = int(request.form['has_guides'])
        has_external_url = int(request.form['has_external_url'])
        edge_followed_by = int(request.form['edge_followed_by'])  # Followers
        edge_follow = int(request.form['edge_follow'])            # Following

        features = np.array([[edge_followed_by, edge_follow, username_length,
                              username_has_number, full_name_has_number, full_name_length,
                              is_private, is_joined_recently, has_channel,
                              is_business_account, has_guides, has_external_url]])

        prediction = model.predict(features)[0]

        result = "Real Account ✅" if prediction == 1 else "Fake Account ❌"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
