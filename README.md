# AI-topics

🏠 Real Estate Price Prediction using Linear Regression
This project uses simple linear regression to predict real estate prices based on square footage (sqft). Built using Python, scikit-learn, Pandas, and Matplotlib, it provides a clear understanding of how linear regression works in a real-world dataset.

📁 Project Structure
real_estate_price_prediction/
├── real_estate_data.csv        # Dataset
├── linear_regression_real_estate.py  # Main script
├── requirements.txt            # Dependencies
├── .gitignore                  # Ignore venv, etc.
├── README.md                   # Project readme
🧠 What This Project Does
Reads real estate data from a CSV

Cleans and preprocesses the data (handles missing values)

Splits the data into training and testing sets

Trains a linear regression model

Visualizes actual vs predicted prices

Displays the slope & intercept of the regression line

📐 Linear Regression Formula Used
The formula of a simple linear regression line:
y=mx+b
Where:
y = predicted price
x = square footage (sqft)
m = slope (price per square foot)
b = intercept (base price when sqft = 0)
After training, the model outputs:
Slope (m) = e.g., 215.34
Intercept (b) = e.g., 163142.57
price = 215.34 × sqft + 163142.57
price=215.34×sqft+163142.57
📊 Example Output
A plot showing:
🔵 Actual prices vs square footage

🔴 Predicted prices vs square footage

✅ Fitted regression line

Print output:
Slope of Regression Line: 215.34
Intercept: 163142.57
✅ How to Run
Clone this repo:

git clone https://github.com/your-username/real-estate-price-prediction.git
cd real-estate-price-prediction
Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install requirements:
pip install -r requirements.txt
Run the script:

python linear_regression_real_estate.py
📦 Dependencies
See requirements.txt, includes:

pandas

numpy

matplotlib

scikit-learn

🚫 Ignored Files
.gitignore includes:

venv/ (Python virtual environment)

__pycache__/

.DS_Store (macOS)

.env (if added)