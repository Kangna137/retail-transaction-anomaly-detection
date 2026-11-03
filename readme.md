Retail Transaction Anomaly Detection
An unsupervised machine-learning project that analyzes retail
transaction data and flags unusual transaction patterns for further
investigation.
Important: A flagged transaction is a potential anomaly, not
proof of fraud, error, or financial loss. This project uses an
unsupervised model and does not have verified anomaly labels.

1. Project objective
The goal is to identify retail transaction lines that differ from common
patterns in the dataset. Examples may include unusually high quantities,
prices, revenue, or invoice totals.

The output is intended to help an analyst prioritize transactions for
manual review.
2. Dataset
This project uses the UCI Online Retail dataset.

The original dataset contains retail transaction records with fields
such as:
- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country
The analysis used a cleaned working dataset. Duplicate rows and records
with non-positive quantity or unit price were excluded from the initial
modeling subset. The resulting modeling dataset contained 524,878
transaction rows.

These filtering choices define the scope of this first version; returns
and cancellation-related records should be studied separately in a
future iteration.
3. Workflow
1. Load data --- read the original retail dataset.
2. Clean data --- parse dates, remove duplicate records, and
   prepare a positive-quantity/positive-price modeling subset.
3. Explore data --- inspect missing values, distributions,
   countries, products, revenue, and time patterns.
4. Engineer features --- create transaction-level variables for
   anomaly detection.
5. Prepare model inputs --- select the numeric feature columns.
6. Train the model --- fit an Isolation Forest model.
7. Generate predictions and scores --- label each row as normal or
   potential anomaly and calculate its anomaly score.
8. Investigate flagged records --- inspect unusual transactions and
   compare summary statistics.
9. Validate and export --- check saved output counts and missing
   prediction fields, then export CSV files.
4. Feature engineering
The model was run using these five features:
  Feature                             Meaning
  Revenue                           Quantity × UnitPrice for a
transaction line

  Hour                              Hour of the day from InvoiceDate

  DayOfWeek                         Day of the week from InvoiceDate

  InvoiceTotal                      Total revenue across the invoice,
repeated on its transaction lines
PriceDeviation                    Product-price deviation feature
created during feature engineering
These features provide the model with information about transaction
value, timing, invoice size, and price behavior.
5. Model
Algorithm: Isolation Forest (sklearn.ensemble.IsolationForest)

Configuration used in this project:
- n_estimators=100
- contamination=0.01
- random_state=42
The model returns:
- 1 --- normal transaction
- -1 --- potential anomaly
The contamination=0.01 setting tells the model to estimate
approximately 1% of the training observations as outliers. It is a
modeling setting, not a measured real-world anomaly rate.

The model's decision_function values are stored as AnomalyScore.
More negative values indicate observations that the model considers more
unusual. They are not probabilities.
6. Results
On the 524,878-row modeling dataset, the saved predictions were:

  Model label             Count
  Normal                519,629
Potential anomaly       5,249
Total                 524,878

The flagged share is approximately 1%, consistent with the configured
contamination setting.

A comparison of the two groups showed that flagged transactions had
higher average and median revenue, quantity, and invoice totals in this
dataset. This is useful for investigation, but does not establish that
every flagged record is genuinely problematic.
7. Project files
The working project contains these main files:
  File                                Purpose
  cleaning.ipynb                    Data loading and cleaning

  feature_engineering.ipynb         Feature creation and preparation

  model_training.ipynb              Isolation Forest initialization,
fitting, prediction, and model
saving

  model_evaluation.ipynb            Score and prediction analysis,
anomaly investigation, and result
export

  model_validation.ipynb            Reloading and validating exported
results

  df_model.pkl                      Saved cleaned modeling dataframe

  engineering.pkl                   Saved engineered feature data

  anomaly_detection.pkl             Saved trained Isolation Forest
model

  retail_transaction_results.csv    All transactions with predictions,
scores, and status

  potential_anomalies.csv           Rows flagged as potential anomalies
Online Retail.csv                 Dataset file used in the project
Keep the dataset and generated files in the expected project directory,
or update the notebook paths if you move them.
8. How to run
1. Install Python and Jupyter support.
2. Install the main dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl
3. Place the dataset in the project directory.
4. Open the notebooks in the order shown in the workflow.
5. Run the cells in sequence, ensuring that each notebook saves the
   files required by the next notebook.
9. Limitations and next improvements
- There are no verified anomaly labels, so accuracy, precision,
  recall, and F1-score cannot be meaningfully reported yet.
- The model's flagged records require domain review.
- The contamination value influences how many records are flagged.
- The current modeling subset excludes non-positive quantities and
  prices, so returns and cancellations are outside this version's
  scope.
- Product-price baselines and invoice-level features must be computed
  using only information available at prediction time before this
  workflow is used for prospective monitoring. In particular, a future
  train/test or time-based evaluation should prevent information from
  the evaluation period leaking into feature calculations.
- Future improvements could include time-based validation, expert
  review of sampled flagged records, threshold sensitivity analysis,
  and separate analysis of returns/cancellations.
10. Learning outcomes
This project provides practice with:
- Data loading and cleaning using pandas
- Exploratory data analysis
- Feature engineering
- Unsupervised learning
- Isolation Forest
- Anomaly scores and prediction interpretation
- Saving and loading models with joblib
- Exporting and validating analysis results
Project status: First end-to-end unsupervised anomaly-detection
workflow completed. The results are exploratory and should be reviewed
before any operational use.