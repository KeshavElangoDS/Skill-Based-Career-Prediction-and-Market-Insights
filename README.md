# Skill-Based Career Prediction and Market Insights Using Machine Learning and Hollands Code

## Project Overview

The **Career Path Predictor** is an innovative career planning tool that leverages machine learning techniques to recommend the most suitable career roles based on a user's unique skills, preferences, and aspirations. Using publicly available data from the **U.S. Department of Labor (Bureau of Labor Statistics, 2024)**, this project aims to help users make well-informed career decisions by aligning their profiles with job market trends.

The core methodology uses a **Random Forest model** to predict the top five career roles best suited to a user’s skillset. The model integrates the **Holland Code framework** (Realistic, Investigative, Artistic, Social, Enterprising/Conventional) to quantify the user’s skills and preferences, providing personalized and actionable career recommendations.

Additionally, the system delivers valuable insights on each predicted role, such as:
- Highest-paying job within each category
- Industry trends (growth/decline)
- Required education and qualifications

The insights are presented in an easy-to-read **visual summary** with data visualizations, enabling users to quickly assess and refer to the information for future decision-making.

---

## Key Features

- **Predictive Career Model**: Uses machine learning (Random Forest) to predict the top five career roles based on a user’s skills and preferences.
- **Holland Code Framework Integration**: Categorizes user preferences into five key dimensions (Realistic, Investigative, Artistic, Social, Enterprising/Conventional) to ensure accurate predictions.
- **Comprehensive Insights**: Provides detailed information on predicted careers, including salary data, job market trends, and education requirements.
- **User-Friendly Visual Summaries**: Outputs career predictions and insights in an easy-to-digest visual format (A4-sized report) with graphs and charts for quick reference.

---

## Methodology

- **Data Collection**: Data sourced from the **Bureau of Labor Statistics (2024)**, including salary information, job growth trends, and educational requirements for various career roles.
- **Modeling**: The **Random Forest algorithm** was selected for its interpretability and robustness, making it ideal for predicting career matches based on multiple input features.
- **Framework**: The **Holland Code** was used to categorize user preferences and skills into five dimensions, providing a structured way to match individuals to the most fitting career roles.

---

## Data Preprocessing

1. **Skill Alignment**: User input is analyzed and matched to corresponding categories within the Holland Code framework.
2. **Data Cleaning**: The dataset was cleaned and normalized for consistency, ensuring accurate predictions.
3. **Feature Engineering**: User profiles are transformed into feature vectors that the Random Forest model can process for predictions.

---

## Future Improvements

- Integrate real-time job market data for more accurate and up-to-date predictions.
- Allow users to input more granular preferences (e.g., desired work-life balance, job location).
- Expand the model to predict additional career insights, such as job satisfaction and work environment.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---









