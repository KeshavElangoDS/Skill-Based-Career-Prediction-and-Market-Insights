import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_random_forest_model(df1):
    """
    Trains a Random Forest classifier to predict the occupation code based on selected features, 
    with feature engineering based on RIASEC themes, model evaluation, and top-5 accuracy.

    This function performs the following steps:
    1. Feature engineering: Adds RIASEC theme columns based on the mean of related columns.
    2. Handles missing values by filling them with the column mean.
    3. Creates a heatmap of feature correlation.
    4. Splits the dataset using an 80-20 train-test split for each category.
    5. Trains a Random Forest classifier.
    6. Evaluates the model’s accuracy and top-5 accuracy.
    7. Outputs the feature importance and top-5 predictions with their confidence levels.

    Args:
        df1 (pd.DataFrame): DataFrame containing the employment data with various columns including:
                             - Features related to RIASEC themes (e.g., 'Mechanical', 
                             'Physical strength and stamina')
                             - Target column: '2023 National Employment Matrix code'

    Returns:
        None: The function prints evaluation metrics including accuracy, 
        feature importance, and top-5 accuracy.
    """

    riasec_mapping = {
        'Realistic': ['Mechanical', 'Physical strength and stamina', 'Fine motor'],
        'Investigative': ['Science', 'Mathematics', 'Critical and analytical thinking'],
        'Artistic': ['Creativity and innovation', 'Writing and reading'],
        'Social': ['Interpersonal', 'Speaking and listening', 'Customer service'],
        'Enterprising': ['Leadership', 'Project management', 
                         'Problem solving and decision making', 'Adaptability'],
        'Conventional': ['Computers and information technology', 'Detail oriented', 'Adaptability'],
    }

    for theme, columns in riasec_mapping.items():
        df1[theme] = df1[columns].mean(axis=1)

    x = df1[['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']]

    x = x.fillna(x.mean())

    correlation_matrix = x.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()

    y = df1['2023 National Employment Matrix code']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    x_train, x_test = pd.DataFrame(), pd.DataFrame()
    y_train, y_test = [], []

    for category in set(y):
        category_indices = y == category
        x_category = x[category_indices]
        y_category = y[category_indices]

        x_cat_train, x_cat_test, y_cat_train, y_cat_test = train_test_split(
            x_category, y_category, test_size=0.15, random_state=42)

        x_train = pd.concat([x_train, x_cat_train])
        x_test = pd.concat([x_test, x_cat_test])
        y_train.extend(y_cat_train)
        y_test.extend(y_cat_test)

    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")

    print(f"Training set size: {len(y_train)}")
    print(f"Testing set size: {len(y_test)}")

    feature_importances = pd.DataFrame({
        'Feature': x.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    y_pred_proba = model.predict_proba(x_test)
    top_5_predictions_with_confidence = []
    correct_predictions = 0

    for i, _ in enumerate(y_pred_proba):
        top_5_indices = np.argsort(y_pred_proba[i])[-5:][::-1]
        top_5_confidences = y_pred_proba[i][top_5_indices]

        top_5_predictions_with_confidence.append(
            list(zip(top_5_indices, top_5_confidences))
        )

        if y_test.iloc[i] in top_5_indices:
            correct_predictions += 1

    top_5_accuracy = correct_predictions / len(y_test)
    print(f"\nModel Top-5 Accuracy: {top_5_accuracy:.2f}")

    for idx, (preds, true_label) in enumerate(zip(top_5_predictions_with_confidence, y_test)):
        print(f"Sample {idx + 1}:")
        print(f"  True Label: {true_label}")
        print(f"  Top 5 Predictions (Class, Confidence): {preds}")
        if idx == 4:
            break
