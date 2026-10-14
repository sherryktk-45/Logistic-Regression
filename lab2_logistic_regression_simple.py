import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# =====================================================================
# DATASET 1 : LOAN APPROVAL
# =====================================================================
print("=" * 60)
print("DATASET 1 : LOAN APPROVAL")
print("=" * 60)

df1 = pd.read_csv("loan_approval.csv")
print(df1.head())
print("Shape:", df1.shape)
print("\nMissing values:\n", df1.isnull().sum())
print("Duplicate rows:", df1.duplicated().sum())

df1 = df1.drop_duplicates()                                        
df1["income"] = df1["income"].fillna(df1["income"].median())        
df1["credit_score"] = df1["credit_score"].fillna(df1["credit_score"].median())
df1["education"] = df1["education"].fillna(df1["education"].mode()[0])   
print("\nMissing values after cleaning:", df1.isnull().sum().sum())
print("Shape after cleaning:", df1.shape)

df1["education"] = df1["education"].map({"Not Graduate": 0, "Graduate": 1})

X1 = df1.drop("loan_approved", axis=1).values
y1 = df1["loan_approved"].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

scaler1 = StandardScaler()
X1_train = scaler1.fit_transform(X1_train)
X1_test = scaler1.transform(X1_test)

model1 = LogisticRegression()
model1.fit(X1_train, y1_train)

y1_pred = model1.predict(X1_test)
cm1 = confusion_matrix(y1_test, y1_pred)
print("\nAccuracy:", accuracy_score(y1_test, y1_pred))
print("Confusion Matrix:\n", cm1)
print(classification_report(y1_test, y1_pred))


print("=" * 60)
print("DATASET 2 : CUSTOMER CHURN")
print("=" * 60)

df2 = pd.read_csv("customer_churn.csv")
print(df2.head())
print("Shape:", df2.shape)
print("\nMissing values:\n", df2.isnull().sum())
print("Duplicate rows:", df2.duplicated().sum())

df2 = df2.drop_duplicates()
df2["monthly_charges"] = df2["monthly_charges"].fillna(df2["monthly_charges"].median())
df2["tenure_months"] = df2["tenure_months"].fillna(df2["tenure_months"].median())
df2["contract_type"] = df2["contract_type"].fillna(df2["contract_type"].mode()[0])
print("\nMissing values after cleaning:", df2.isnull().sum().sum())
print("Shape after cleaning:", df2.shape)

df2["contract_type"] = df2["contract_type"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

X2 = df2.drop("churn", axis=1).values
y2 = df2["churn"].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)

model2 = LogisticRegression()
model2.fit(X2_train, y2_train)

y2_pred = model2.predict(X2_test)
cm2 = confusion_matrix(y2_test, y2_pred)
print("\nAccuracy:", accuracy_score(y2_test, y2_pred))
print("Confusion Matrix:\n", cm2)
print(classification_report(y2_test, y2_pred))


# =====================================================================
# PLOTS (matplotlib) - shown after the output
# =====================================================================
# Plots for Dataset 1
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
fig1.suptitle("Dataset 1 : Loan Approval")

df1["loan_approved"].value_counts().sort_index().plot(kind="bar", ax=axes1[0], color=["red", "green"])
axes1[0].set_title("Class Distribution (0 = Rejected, 1 = Approved)")
axes1[0].set_xlabel("loan_approved")
axes1[0].set_ylabel("Count")

axes1[1].hist(df1["credit_score"], bins=15, color="skyblue", edgecolor="black")
axes1[1].set_title("Credit Score Histogram")
axes1[1].set_xlabel("credit_score")
axes1[1].set_ylabel("Count")

ConfusionMatrixDisplay(cm1, display_labels=["Rejected", "Approved"]).plot(ax=axes1[2])
axes1[2].set_title("Confusion Matrix")
plt.tight_layout()

# Plots for Dataset 2
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("Dataset 2 : Customer Churn")

df2["churn"].value_counts().sort_index().plot(kind="bar", ax=axes2[0], color=["green", "red"])
axes2[0].set_title("Class Distribution (0 = Stay, 1 = Churn)")
axes2[0].set_xlabel("churn")
axes2[0].set_ylabel("Count")

axes2[1].hist(df2["tenure_months"], bins=15, color="orange", edgecolor="black")
axes2[1].set_title("Tenure Months Histogram")
axes2[1].set_xlabel("tenure_months")
axes2[1].set_ylabel("Count")

ConfusionMatrixDisplay(cm2, display_labels=["Stay", "Churn"]).plot(ax=axes2[2])
axes2[2].set_title("Confusion Matrix")
plt.tight_layout()

plt.show()    # close the plot windows to continue to the user interfaces


# =====================================================================
# PREDICTION FUNCTIONS (used by both Gradio and Flask)
# =====================================================================
def predict_loan(income, loan_amount, credit_score, age, employment_years, education):
    if education == "Graduate":
        edu = 1
    else:
        edu = 0
    data = scaler1.transform([[income, loan_amount, credit_score, age, employment_years, edu]])
    result = model1.predict(data)[0]
    probability = model1.predict_proba(data)[0][result] * 100
    if result == 1:
        return f"Loan Approved (confidence: {probability:.1f}%)"
    else:
        return f"Loan Rejected (confidence: {probability:.1f}%)"


def predict_churn(tenure_months, monthly_charges, support_calls, contract_type):
    contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract_type]
    data = scaler2.transform([[tenure_months, monthly_charges, support_calls, contract]])
    result = model2.predict(data)[0]
    probability = model2.predict_proba(data)[0][result] * 100
    if result == 1:
        return f"Customer will Leave - Churn (confidence: {probability:.1f}%)"
    else:
        return f"Customer will Stay (confidence: {probability:.1f}%)"


# =====================================================================
# GRADIO USER INTERFACE
# =====================================================================
loan_ui = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Income", value=90000),
        gr.Number(label="Loan Amount", value=20000),
        gr.Number(label="Credit Score", value=700),
        gr.Number(label="Age", value=35),
        gr.Number(label="Employment Years", value=8),
        gr.Dropdown(["Graduate", "Not Graduate"], label="Education", value="Graduate"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Loan Approval Prediction",
)

churn_ui = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Tenure (months)", value=12),
        gr.Number(label="Monthly Charges", value=70),
        gr.Number(label="Support Calls", value=3),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Type", value="Month-to-month"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Customer Churn Prediction",
)

gradio_app = gr.TabbedInterface([loan_ui, churn_ui], ["Loan Approval", "Customer Churn"])
gradio_app.launch(prevent_thread_lock=True)      # runs on port 7860 in the background


# =====================================================================
# FLASK USER INTERFACE
# =====================================================================
app = Flask(__name__)

STYLE = """
<style>
    body { font-family: Arial; background: #f4f6f9; padding: 30px; }
    .box { max-width: 450px; margin: auto; background: white; padding: 25px; border-radius: 10px; }
    label { display: block; margin-top: 12px; font-weight: bold; }
    input, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
    button { width: 100%; margin-top: 18px; padding: 10px; background: #2d6cdf; color: white; border: none; font-size: 16px; }
    .result { margin-top: 20px; padding: 12px; background: #e8f5e9; text-align: center; font-size: 18px; }
</style>
"""

HOME_PAGE = STYLE + """
<div class="box">
    <h2>Logistic Regression - Flask</h2>
    <p><a href="/loan">1. Loan Approval Prediction</a></p>
    <p><a href="/churn">2. Customer Churn Prediction</a></p>
</div>
"""

LOAN_PAGE = STYLE + """
<div class="box">
    <h2>Loan Approval Prediction</h2>
    <form method="post">
        <label>Income</label><input type="number" step="any" name="income" value="90000" required>
        <label>Loan Amount</label><input type="number" step="any" name="loan_amount" value="20000" required>
        <label>Credit Score</label><input type="number" step="any" name="credit_score" value="700" required>
        <label>Age</label><input type="number" step="any" name="age" value="35" required>
        <label>Employment Years</label><input type="number" step="any" name="employment_years" value="8" required>
        <label>Education</label>
        <select name="education">
            <option>Graduate</option>
            <option>Not Graduate</option>
        </select>
        <button type="submit">Predict</button>
    </form>
    {% if result %}<div class="result">{{ result }}</div>{% endif %}
    <p><a href="/">Back to Home</a></p>
</div>
"""

CHURN_PAGE = STYLE + """
<div class="box">
    <h2>Customer Churn Prediction</h2>
    <form method="post">
        <label>Tenure (months)</label><input type="number" step="any" name="tenure_months" value="12" required>
        <label>Monthly Charges</label><input type="number" step="any" name="monthly_charges" value="70" required>
        <label>Support Calls</label><input type="number" step="any" name="support_calls" value="3" required>
        <label>Contract Type</label>
        <select name="contract_type">
            <option>Month-to-month</option>
            <option>One year</option>
            <option>Two year</option>
        </select>
        <button type="submit">Predict</button>
    </form>
    {% if result %}<div class="result">{{ result }}</div>{% endif %}
    <p><a href="/">Back to Home</a></p>
</div>
"""


@app.route("/")
def home():
    return render_template_string(HOME_PAGE)


@app.route("/loan", methods=["GET", "POST"])
def loan():
    result = None
    if request.method == "POST":
        result = predict_loan(
            float(request.form["income"]),
            float(request.form["loan_amount"]),
            float(request.form["credit_score"]),
            float(request.form["age"]),
            float(request.form["employment_years"]),
            request.form["education"],
        )
    return render_template_string(LOAN_PAGE, result=result)


@app.route("/churn", methods=["GET", "POST"])
def churn():
    result = None
    if request.method == "POST":
        result = predict_churn(
            float(request.form["tenure_months"]),
            float(request.form["monthly_charges"]),
            float(request.form["support_calls"]),
            request.form["contract_type"],
        )
    return render_template_string(CHURN_PAGE, result=result)

if __name__ == "__main__":
    app.run(port=5000)
