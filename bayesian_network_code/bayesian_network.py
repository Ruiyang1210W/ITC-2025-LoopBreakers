import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, K2Score, HillClimbSearch
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, classification_report

# Load training and testing data
train_df = pd.read_csv("train_binned_outage_data.csv")
test_df = pd.read_csv("test_binned_outage_data.csv")

# Before training, group rare outage causes
train_df['Outage Cause'] = train_df['Outage Cause'].replace({
    'Animal': 'Other',
    'Operation': 'Other',
    'Third Party': 'Other',
    # Keep more frequent ones as-is
})

test_df['Outage Cause'] = test_df['Outage Cause'].replace({
    'Animal': 'Other',
    'Operation': 'Other',
    'Third Party': 'Other',
    # Keep more frequent ones as-is
})


# Learn network structure
hc = HillClimbSearch(train_df)
best_model = hc.estimate(scoring_method=K2Score(train_df))

# Define and train the Bayesian model
model = BayesianModel(best_model.edges())
model.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu")

# Set up inference engine
inference = VariableElimination(model)

# Predict on test data
true_labels = []
predicted_labels = []

for _, row in test_df.iterrows():
    evidence = row.drop('Outage Cause').to_dict()
    result = inference.map_query(variables=["Outage Cause"], evidence=evidence)
    true_labels.append(row["Outage Cause"])
    predicted_labels.append(result["Outage Cause"])

# Accuracy and performance report
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get all unique labels (sorted for cleaner display)
labels = sorted(set(true_labels + predicted_labels))

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

# Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - Outage Cause Prediction")
plt.tight_layout()
plt.show()



