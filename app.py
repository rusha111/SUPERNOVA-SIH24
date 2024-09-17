# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import instaloader
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Function to get Instagram data using Instaloader
def get_instagram_data(username):
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
        return {
            "userFollowerCount": profile.followers,
            "userFollowingCount": profile.followees,
            "userBiographyLength": len(profile.biography),
            "userMediaCount": profile.mediacount,
            "userHasProfilPic": int(not profile.is_private and profile.profile_pic_url is not None),
            "userIsPrivate": int(profile.is_private),
            "usernameDigitCount": sum(c.isdigit() for c in profile.username),
            "usernameLength": len(profile.username),
        }
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile with username '{username}' not found.")
        return None

# Load the trained model
load_model = tf.keras.models.load_model('trainedmodel')

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/advice')
def advice():
    # Render the advice.html template
    return render_template('advice.html')
@app.route('/about')
def about():
    # Render the advice.html template
    return render_template('about.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')

@app.route('/result')
def result():
    # Get parameters from the query string or use default values
    username = request.args.get('username', 'N/A')
    confidence = float(request.args.get('confidence', 'N/A'))  # Ensure confidence is a float
    behavioral_analysis = request.args.get('behavioral_analysis', 'N/A')

    return render_template('result.html', username=username, confidence=confidence, behavioral_analysis=behavioral_analysis)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the username from the form submission
        username = request.form['username']
        reasons = request.form.getlist('reasons')
        others = request.form.get('others')

        # Get Instagram data
        insta_data = get_instagram_data(username)

        if insta_data:
            # Convert Instagram data to NumPy array
            X_new = np.array([list(insta_data.values())], dtype=np.float32)

            # Make predictions
            predictions = load_model.predict(X_new)

            # Get the number of checkboxes selected
            num_checkboxes_selected = len(reasons) + (1 if others else 0)

            # Perform behavioral analysis
            behavioral_analysis_result = "Behavioral Analysis: "
            if num_checkboxes_selected > 5:
                behavioral_analysis_result += "The user exhibits suspicious behavior."
            else:
                behavioral_analysis_result += "The user's behavior seems normal."

            # Determine the result
            confidence_percentage = (1 - predictions[0][0]) * 100  # Subtract probability from 1 and multiply by 100
            result_text = f"Prediction for {username}: {'Fake' if predictions[0][0] >= 0.5 else 'Real'} " \
                          f"(Confidence: {confidence_percentage:.2f}%) - {behavioral_analysis_result}"

            # Redirect to the result page with the result as parameters
            return redirect(url_for('result', username=username, confidence=confidence_percentage, behavioral_analysis=behavioral_analysis_result))
        else:
            return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis='Profile not found.')
    except Exception as e:
        return render_template('result.html', username='N/A', confidence='N/A', behavioral_analysis=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5004)