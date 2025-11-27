import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.formula.api as smf
import nltk
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
from io import BytesIO

# Download NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

st.set_page_config(page_title="Survey Evaluation Dashboard", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #4b5fb5;'>Evaluation Study Survey Dashboard</h1>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])


# Helper for safe quoting of column names in regression formulas
def quote_col(col_name: str) -> str:
    # Statsmodels / patsy allow backtick-quoted names
    return f"`{col_name}`"


def build_categorical_summary(df_cat: pd.DataFrame, cat_cols: list) -> dict:
    summaries = {}
    for col in cat_cols:
        vc = df_cat[col].value_counts(dropna=False)
        total = vc.sum()
        freq_df = (
            vc.reset_index()
            .rename(columns={"index": col, col: "Count"})
        )
        freq_df["Percent"] = (freq_df["Count"] / total * 100).round(2)
        summaries[col] = freq_df
    return summaries


def create_report_excel(df_typed: pd.DataFrame, num_cols: list, cat_cols: list) -> BytesIO:
    """
    Excel report with:
    - Numeric descriptive statistics (+ chart of means)
    - Categorical/ordinal frequency tables (+ bar charts)
    - Correlation matrix (if available)
    No raw row-level dataset.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Numeric descriptive statistics
        if num_cols:
            desc = df_typed[num_cols].describe().T
            sheet_name = "Numeric_Describe"
            desc.to_excel(writer, sheet_name=sheet_name)
            numeric_ws = writer.sheets[sheet_name]

            if "mean" in desc.columns:
                chart = workbook.add_chart({"type": "column"})
                n_rows = len(desc.index)

                cat_first_row = 1
                cat_last_row = n_rows
                cat_col_idx = 0  # variable names
                mean_col_idx = list(desc.columns).index("mean") + 1  # +1 for index col

                chart.add_series({
                    "name": "Mean values",
                    "categories": [sheet_name, cat_first_row, cat_col_idx,
                                   cat_last_row, cat_col_idx],
                    "values": [sheet_name, cat_first_row, mean_col_idx,
                               cat_last_row, mean_col_idx],
                })
                chart.set_title({"name": "Mean of numeric variables"})
                chart.set_x_axis({"name": "Variable"})
                chart.set_y_axis({"name": "Mean"})

                numeric_ws.insert_chart(cat_last_row + 3, 0, chart)

        # Categorical/ordinal frequencies with charts
        if cat_cols:
            cat_summaries = build_categorical_summary(df_typed, cat_cols)
            sheet_name = "Categorical_Frequencies"
            cat_ws = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = cat_ws

            start_row = 0
            for col, freq_df in cat_summaries.items():
                cat_ws.write(start_row, 0, col)
                freq_df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=start_row + 1,
                    startcol=0,
                    index=False
                )

                n_rows = len(freq_df)
                chart = workbook.add_chart({"type": "column"})

                cat_first_row = start_row + 2
                cat_last_row = start_row + 1 + n_rows
                cat_col_idx = 0
                count_col_idx = 1

                chart.add_series({
                    "name": col,
                    "categories": [sheet_name, cat_first_row, cat_col_idx,
                                   cat_last_row, cat_col_idx],
                    "values": [sheet_name, cat_first_row, count_col_idx,
                               cat_last_row, count_col_idx],
                })
                chart.set_title({"name": f"Distribution of {col}"})
                chart.set_x_axis({"name": "Response"})
                chart.set_y_axis({"name": "Count"})

                cat_ws.insert_chart(start_row, 4, chart)

                start_row = cat_last_row + 5

        # Correlation matrix
        if len(num_cols) >= 2:
            corr = df_typed[num_cols].corr()
            corr.to_excel(writer, sheet_name="Correlation_Matrix")

    buffer.seek(0)
    return buffer


if uploaded_file is not None:
    # Load data
    filename = uploaded_file.name
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data preview")
    st.dataframe(df.head())

    # Reset type map when a new file is uploaded
    if "filename" not in st.session_state:
        st.session_state["filename"] = None
    if "type_map" not in st.session_state:
        st.session_state["type_map"] = {}

    if st.session_state["filename"] != filename:
        st.session_state["filename"] = filename
        st.session_state["type_map"] = {}

    type_map = st.session_state["type_map"]

    # Optional: show raw pandas dtypes just for your information
    with st.expander("View raw detected dtypes (for info only)"):
        dtypes_df = pd.DataFrame(
            {"Variable": df.columns, "Detected dtype": df.dtypes.astype(str)}
        )
        st.dataframe(dtypes_df)

    st.markdown("## Manually Set Variable Types")

    type_options = ["Numeric", "Categorical", "Text", "Ordinal", "Datetime"]

    # Manual type selection for every column (no auto-detection logic)
    for i, col in enumerate(df.columns):
        current = type_map.get(col, "Text")
        if current not in type_options:
            current = "Text"

        selected = st.selectbox(
            f"Select type for {col}",
            type_options,
            index=type_options.index(current),
            key=f"type_{i}"
        )
        type_map[col] = selected

    # Build typed dataframe
    df_typed = df.copy()
    for col, typ in type_map.items():
        if typ == "Numeric":
            df_typed[col] = pd.to_numeric(df_typed[col], errors="coerce")
        elif typ in ["Categorical", "Ordinal"]:
            df_typed[col] = df_typed[col].astype("category")
        elif typ == "Datetime":
            df_typed[col] = pd.to_datetime(df_typed[col], errors="coerce")
        # Text left as-is

    st.markdown("## Step 2: Explore your data")

    if len(df_typed.columns) == 0:
        st.info("No columns found in the uploaded file.")
    else:
        tab_univariate, tab_desc, tab_freq, tab_corr = st.tabs(
            ["Univariate explorer", "Descriptive statistics", "Frequency tables", "Correlation matrix"]
        )

        # Univariate explorer
        with tab_univariate:
            st.subheader("Univariate explorer")
            selected_column = st.selectbox("Select a column to explore", df_typed.columns)
            col_type = type_map.get(selected_column, "Text")

            if col_type in ["Categorical", "Ordinal"]:
                value_counts = df_typed[selected_column].value_counts(dropna=False).reset_index()
                value_counts.columns = [selected_column, "Count"]
                value_counts["Percent"] = (value_counts["Count"] /
                                           value_counts["Count"].sum() * 100).round(2)
                fig = px.bar(
                    value_counts,
                    x=selected_column,
                    y="Count",
                    title=f"Distribution of {selected_column}",
                    labels={selected_column: selected_column, "Count": "Count"},
                    color=selected_column
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(value_counts)

            elif col_type == "Numeric":
                fig = px.histogram(
                    df_typed,
                    x=selected_column,
                    nbins=20,
                    title=f"Distribution of {selected_column}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write("Summary statistics")
                st.dataframe(df_typed[selected_column].describe())

            elif col_type == "Text":
                st.subheader("Word cloud and text summary")
                text_series = df_typed[selected_column].dropna().astype(str)
                text = " ".join(text_series)

                if text.strip():
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white"
                    ).generate(text)
                    st.image(wordcloud.to_array(), use_container_width=True)
                else:
                    st.info("No text available to generate a word cloud.")

                if st.button("Generate text summary"):
                    if text.strip():
                        words = nltk.word_tokenize(text.lower())
                        stop_words = set(stopwords.words("english"))
                        words = [w for w in words if w.isalpha() and w not in stop_words]
                        most_common = Counter(words).most_common(10)
                        keywords = ", ".join([w for w, _ in most_common])

                        blob = TextBlob(text)
                        sentiment = blob.sentiment.polarity
                        if sentiment > 0:
                            sentiment_label = "Positive"
                        elif sentiment < 0:
                            sentiment_label = "Negative"
                        else:
                            sentiment_label = "Neutral"

                        st.markdown(f"Most mentioned keywords: {keywords}")
                        st.markdown(f"Overall sentiment: {sentiment_label}")

                        summary_sentences = blob.sentences[:3]
                        summary_text = " ".join(str(s) for s in summary_sentences)
                        st.markdown(f"Auto generated summary: {summary_text}")
                    else:
                        st.info("No text available for summary.")

        # Descriptive statistics
        with tab_desc:
            st.subheader("Descriptive statistics for numeric columns")
            num_cols = [col for col, typ in type_map.items() if typ == "Numeric"]
            if num_cols:
                desc = df_typed[num_cols].describe().T
                st.dataframe(desc)
            else:
                st.info("No numeric columns have been defined.")

        # Frequency tables
        with tab_freq:
            st.subheader("Frequency tables for categorical and ordinal columns")
            cat_cols = [col for col, typ in type_map.items() if typ in ["Categorical", "Ordinal"]]
            if cat_cols:
                freq_col = st.selectbox("Select a column", cat_cols, key="freq_col")
                vc = df_typed[freq_col].value_counts(dropna=False)
                total = vc.sum()
                freq_df = (
                    vc.reset_index()
                    .rename(columns={"index": freq_col, freq_col: "Count"})
                )
                freq_df["Percent"] = (freq_df["Count"] / total * 100).round(2)
                st.dataframe(freq_df)
            else:
                st.info("No categorical or ordinal columns have been defined.")

        # Correlation matrix
        with tab_corr:
            st.subheader("Correlation matrix for numeric columns")
            num_cols_corr = [col for col, typ in type_map.items() if typ == "Numeric"]
            if len(num_cols_corr) >= 2:
                corr = df_typed[num_cols_corr].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(corr)
            else:
                st.info("Correlation requires at least two numeric columns.")

    # Only run tests and export if we have a file
    if uploaded_file is not None:
        st.markdown("## Step 3: Statistical tests")

        cat_cols_all = [col for col, typ in type_map.items() if typ in ["Categorical", "Ordinal"]]
        num_cols_all = [col for col, typ in type_map.items() if typ == "Numeric"]

        # Chi square
        if len(cat_cols_all) >= 2:
            st.subheader("Chi square test")
            chi1 = st.selectbox("Categorical variable 1", cat_cols_all, key="chi1")
            chi2 = st.selectbox("Categorical variable 2", cat_cols_all, key="chi2")
            if st.button("Run chi square"):
                chi_table = pd.crosstab(df_typed[chi1], df_typed[chi2])
                chi2_stat, p_val, _, _ = chi2_contingency(chi_table)
                st.write(f"Chi square test p value: {p_val:.4f}")
                st.dataframe(chi_table)
        else:
            st.info("Chi square test needs at least two categorical or ordinal variables.")

        # ANOVA
        if len(cat_cols_all) >= 1 and len(num_cols_all) >= 1:
            st.subheader("ANOVA")
            group_col = st.selectbox("Group (categorical or ordinal)", cat_cols_all, key="anova_group")
            value_col = st.selectbox("Value (numeric)", num_cols_all, key="anova_val")
            if st.button("Run ANOVA"):
                grouped = [
                    group[value_col].dropna()
                    for _, group in df_typed.groupby(group_col)
                ]
                if len(grouped) >= 2:
                    f_stat, p_val = f_oneway(*grouped)
                    st.write(f"ANOVA test p value: {p_val:.4f}")
                else:
                    st.info("ANOVA requires at least two groups.")
        else:
            st.info("ANOVA needs at least one categorical/ordinal and one numeric variable.")

        # Regression
        if len(num_cols_all) >= 2:
            st.subheader("Linear regression")
            y = st.selectbox("Dependent variable (numeric)", num_cols_all, key="reg_y")
            x = st.selectbox("Independent variable", df_typed.columns, key="reg_x")
            if st.button("Run regression"):
                x_type = type_map.get(x, "Numeric")

                y_term = quote_col(y)
                if x_type in ["Categorical", "Text", "Ordinal"]:
                    x_term = f"C({quote_col(x)})"
                else:
                    x_term = quote_col(x)

                formula = f"{y_term} ~ {x_term}"
                try:
                    model = smf.ols(formula=formula, data=df_typed).fit()
                    st.text(model.summary())
                except Exception as e:
                    st.error(f"Regression failed: {e}")
        else:
            st.info("Linear regression needs at least two numeric variables.")

        st.markdown("## Step 4: Download Excel report (tables + charts)")

        if st.button("Generate Excel report"):
            buffer = create_report_excel(df_typed, num_cols_all, cat_cols_all)
            st.download_button(
                label="Download Excel report",
                data=buffer,
                file_name="evaluation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
