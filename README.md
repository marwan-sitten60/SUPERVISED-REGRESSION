# USA Housing Price Prediction

## Project Overview
This project aims to predict housing prices in the USA using a linear regression model. The analysis is performed using a Jupyter Notebook (`USA-Housing-LinearRegression.ipynb`) that includes data exploration, visualization, and model evaluation.

## Dataset
The dataset (`USA_Housing.csv`) contains the following features:
- **Avg. Area Income**: Average income of residents in the area.
- **Avg. Area House Age**: Average age of houses in the area.
- **Avg. Area Number of Rooms**: Average number of rooms in houses.
- **Avg. Area Number of Bedrooms**: Average number of bedrooms in houses.
- **Area Population**: Population of the area.
- **Price**: House price (target variable).
- **Address**: Address of the house (not used in the model).

## Requirements
To run the notebook, you need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure
- **USA-Housing-LinearRegression.ipynb**: Jupyter Notebook containing the code for data loading, exploratory data analysis (EDA), model training, and evaluation.
- **USA_Housing.csv**: Dataset used for the analysis (not included in this repository; ensure it is available in the correct path).
- **README.md**: This file, providing an overview and instructions.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   ```
2. **Set Up Environment**:
   Ensure all required libraries are installed (see Requirements).
3. **Place Dataset**:
   Place the `USA_Housing.csv` file in the appropriate directory (e.g., `C:\Projects\full project\house prediction\` as specified in the notebook).
4. **Run the Notebook**:
   Open the `USA-Housing-LinearRegression.ipynb` in Jupyter Notebook or JupyterLab and execute the cells sequentially.

## Analysis Steps
1. **Data Loading and Inspection**:
   - Load the dataset using `pandas`.
   - Display the first few rows and check for missing values.
   - Summarize statistical information with `describe()` and `info()`.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize relationships between variables using scatter plots.
   - Plot a histogram of residuals to assess model performance.

3. **Model Training**:
   - Train a linear regression model using features (excluding the `Address` column) to predict the `Price`.
   - Split the data into training and testing sets.

4. **Model Evaluation**:
   - Evaluate the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
   - Example metrics from the notebook:
     - MAE: 79,799.67
     - MSE: 9,888,176,181.46
     - RMSE: 99,439.31

## Visualizations
- **Scatter Plot**: Compares actual prices (`y_test`) vs. predicted prices (`predictions`) to visualize model accuracy.
- **Residual Histogram**: Displays the distribution of residuals (`y_test - predictions`) to assess the model's error distribution.

## Notes
- The notebook assumes the dataset is located at `C:\Projects\full project\house prediction\USA_Housing.csv`. Update the file path in the notebook if necessary.
- The `Address` column is not used in the model as it is non-numeric and not relevant for prediction.
- The model performance metrics (MAE, MSE, RMSE) indicate the model's accuracy, with lower values suggesting better performance.

## Future Improvements
- Experiment with additional features or feature engineering to improve model accuracy.
- Try other regression models (e.g., Random Forest, XGBoost) for comparison.
- Include cross-validation to ensure robust model evaluation.

## License
This project is for educational purposes and does not include a specific license. Ensure you have the right to use the dataset (`USA_Housing.csv`) for your purposes.
