
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


st.set_page_config(
    page_title="Retail Anomaly Detection",layout="wide")


BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "anomaly_detection.pkl")
    product_medians = joblib.load(BASE_DIR / "product_medians.pkl")

    # Standardize product codes for matching
    product_medians.index = product_medians.index.astype(str).str.strip()

    return model, product_medians


try:
    model, product_medians = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model or product medians: {e}")


st.title("Retail Transaction Anomaly Detection")

st.markdown("""
<style>
.subtitle {
    font-size: 20px;
    font-weight: 600;
    background: linear-gradient(
        90deg,
        #ff4b4b,
        #ff9f43,
        #feca57,
        #1dd1a1,
        #54a0ff,
        #a55eea
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
</style>

<p class="subtitle">
An ML-powered application to identify unusual retail transactions using Isolation Forest.
</p>
""", unsafe_allow_html=True)

st.divider()


st.header("Upload Retail Transaction Data")

uploaded_file = st.file_uploader(
    "Choose your transaction file",
    type=["csv", "xlsx"]
)

required_columns = [
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice"
]


if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        st.subheader("Dataset Preview")

        st.write("Dataset Shape:", data.shape)

        st.dataframe(
            data.head(10),
            use_container_width=True
        )


        # ==================================
        # 7. DATASET OVERVIEW
        # ==================================

        st.divider()
        st.header("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Transactions",
                f"{len(data):,}"
            )

        with col2:
            st.metric(
                "Total Columns",
                len(data.columns)
            )

        with col3:
            st.metric(
                "Missing Values",
                f"{data.isnull().sum().sum():,}"
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                f"{data.duplicated().sum():,}"
            )


      

        st.divider()
        st.header("🔍 Data Validation")

        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

            st.stop()

        st.success("All required columns are available!")

        if not model_loaded:
            st.stop()


        st.divider()
        st.header("🤖 Anomaly Detection")

        if st.button(
            "Run Anomaly Detection",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Cleaning data and detecting potential anomalies..."
            ):

                cleaned_data = data.copy()

                cleaned_data = cleaned_data.drop_duplicates()

                cleaned_data["Quantity"] = pd.to_numeric(
                    cleaned_data["Quantity"],
                    errors="coerce"
                )

                cleaned_data["UnitPrice"] = pd.to_numeric(
                    cleaned_data["UnitPrice"],
                    errors="coerce"
                )

                cleaned_data["InvoiceDate"] = pd.to_datetime(
                    cleaned_data["InvoiceDate"],
                    errors="coerce"
                )

                cleaned_data = cleaned_data.dropna(
                    subset=[
                        "InvoiceNo",
                        "StockCode",
                        "Quantity",
                        "UnitPrice",
                        "InvoiceDate"
                    ]
                )

                cleaned_data = cleaned_data[
                    (cleaned_data["Quantity"] > 0) &
                    (cleaned_data["UnitPrice"] > 0)
                ].copy()

                if cleaned_data.empty:
                    st.error("No valid transactions found.")
                    st.stop()


                # Revenue
                cleaned_data["Revenue"] = (
                    cleaned_data["Quantity"] *
                    cleaned_data["UnitPrice"]
                )

                # Hour
                cleaned_data["Hour"] = (
                    cleaned_data["InvoiceDate"].dt.hour
                )

                # Day of week
                cleaned_data["DayOfWeek"] = (
                    cleaned_data["InvoiceDate"].dt.dayofweek
                )

                # Invoice total
                cleaned_data["InvoiceTotal"] = (
                    cleaned_data.groupby("InvoiceNo")["Revenue"]
                    .transform("sum")
                )

                # Price deviation using training reference medians
                stock_codes = (
                    cleaned_data["StockCode"]
                    .astype(str)
                    .str.strip()
                )

                reference_prices = stock_codes.map(
                    product_medians
                )

                # Fallback for products not present during training
                global_median = float(product_medians.median())

                reference_prices = reference_prices.fillna(
                    global_median
                )

                cleaned_data["PriceDeviation"] = (
                    (cleaned_data["UnitPrice"] - reference_prices)
                    / reference_prices
                )
                features = [
                    "Revenue",
                    "Hour",
                    "DayOfWeek",
                    "InvoiceTotal",
                    "PriceDeviation"
                ]

                X = cleaned_data[features].copy()

                if not np.isfinite(X.to_numpy()).all():
                    st.error(
                        "Invalid numerical values found in model features."
                    )
                    st.stop()

                predictions = model.predict(X)

                scores = model.decision_function(X)


        
                results = cleaned_data.copy()

                results["Prediction"] = predictions

                results["AnomalyScore"] = scores

                results["Status"] = results["Prediction"].map({
                    1: "Normal",
                    -1: "Potential Anomaly"
                })

                anomalies = results[
                    results["Prediction"] == -1
                ].copy()

                st.session_state["results"] = results
                st.session_state["anomalies"] = anomalies

                st.success(
                    "Anomaly detection completed successfully!"
                )

        if "results" in st.session_state:

            results = st.session_state["results"]
            anomalies = st.session_state["anomalies"]

            st.divider()
            st.header("📈 Anomaly Detection Results")

            normal_count = int(
                (results["Prediction"] == 1).sum()
            )

            anomaly_count = len(anomalies)

            total_count = len(results)

            anomaly_percentage = (
                anomaly_count / total_count * 100
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Transactions",
                    f"{total_count:,}"
                )

            with col2:
                st.metric(
                    "Normal Transactions",
                    f"{normal_count:,}"
                )

            with col3:
                st.metric(
                    "Potential Anomalies",
                    f"{anomaly_count:,}"
                )

            with col4:
                st.metric(
                    "Anomaly Percentage",
                    f"{anomaly_percentage:.2f}%"
                )
            st.divider()
            st.subheader("📊 Prediction Distribution")

            chart_data = pd.DataFrame({
                "Transaction Type": [
                    "Normal",
                    "Potential Anomaly"
                ],
                "Count": [
                    normal_count,
                    anomaly_count
                ]
            })

            st.bar_chart(
                chart_data.set_index("Transaction Type")
            )



            st.divider()
            st.subheader("🚨 Potential Anomalies")

            if not anomalies.empty:

                st.dataframe(
                    anomalies.sort_values(
                        "AnomalyScore",
                        ascending=True
                    ),
                    use_container_width=True
                )

            else:
                st.info("No potential anomalies detected.")


           

            st.divider()
            st.subheader("🔎 Explore Transactions")

            status_filter = st.selectbox(
                "Select transaction status",
                [
                    "All",
                    "Normal",
                    "Potential Anomaly"
                ]
            )

            if status_filter == "All":
                filtered_data = results

            else:
                filtered_data = results[
                    results["Status"] == status_filter
                ]

            st.write(
                "Filtered transactions:",
                len(filtered_data)
            )

            st.dataframe(
                filtered_data,
                use_container_width=True
            )

            st.divider()
            st.header("📥 Download Reports")

            col1, col2 = st.columns(2)

            with col1:

                all_csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download All Results",
                    data=all_csv,
                    file_name="retail_anomaly_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:

                anomaly_csv = anomalies.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Potential Anomalies",
                    data=anomaly_csv,
                    file_name="potential_anomalies.csv",
                    mime="text/csv",
                    use_container_width=True
                )


    except Exception as e:
            st.error(
                f"An error occurred while processing the file: {e}"
            )


else:

    st.info(
        "Please upload a CSV or Excel transaction file to begin."
    )

    st.caption(
        "Supported formats: CSV and XLSX"
    )
