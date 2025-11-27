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


def create_summary_excel(df_typed: pd.DataFrame, num_cols: list, cat_cols: list) -> BytesIO:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        # Raw typed data
        df_typed.to_excel(writer, sheet_name="RawData", index=False)

        # Numeric descriptive statistics
        if num_cols:
            desc = df_typed[num_cols].describe().T
            desc.to_excel(writer, sheet_name="Numeric_Describe")

        # Categorical frequencies
        if cat_cols:
            cat_summaries = build_categorical_summary(df_typed, cat_cols)
            start_row = 0
            sheet_name = "Categorical_Frequencies"
            workbook = writer.book
            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet

            for col, freq_df in cat_summaries.items():
                worksheet.write(start_row, 0, col)
                freq_df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=start_row + 1,
                    startcol=0,
                    index=False
                )
                start_row += len(freq_df) + 3

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

    # Initialise or reset column type state when file changes or structure changes
    cols = list(df.columns)

    if "uploaded_filename" not in st.session_state:
        st.session_state["uploaded_filename"] = None
    if "col_types" not in st.session_state:
        st.session_state["col_types"] = []
    if "col_names" not in st.session_state:
        st.session_state["col_names"] = []

    # If new file or different column structure, re-infer types
    if (
        st.session_state["uploaded_filename"] != filename
        or len(st.session_state["col_names"]) != len(cols)
        or list(st.session_state["col_names"]) != cols
    ):
        inferred_types = []
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                inferred_type = "Numeric"
            else:
                unique_vals = df[col].nunique(dropna=True)
                inferred_type = "Text" if unique_vals > 30 else "Categorical"
            inferred_types.append(inferred_type)

        st.session_state["uploaded_filename"] = filename
        st.session_state["col_names"] = cols
        st.session_state["col_types"] = inferred_types

    st.markdown("## Step 1: Review and confirm column types")

    type_options = ["Numeric", "Categorical", "Text", "Ordinal"]

    col_types = st.session_state["col_types"]

    # Column type selection with persistence
    for i, col in enumerate(cols):
        current_type = col_types[i]
        # Ensure current_type is valid
        if current_type not in type_options:
            current_type = "Text"

        selected_type = st.selectbox(
            f"{col} (detected: {current_type})",
            type_options,
            index=type_options.index(current_type),
            key=f"col_type_{i}"
        )

        col_types[i] = selected_type

    # Build a mapping from column name to selected type
    user_column_types = {col: col_types[i] for i, col in enumerate(cols)}

    # Apply chosen types to a working copy of the dataframe
    df_typed = df.copy()

    for col, typ in user_column_types.items():
        if typ == "Numeric":
            df_typed[col] = pd.to_numeric(df_typed[col], errors="coerce")
        elif typ in ["Categorical", "Ordinal"]:
            df_typed[col] = df_typed[col].astype("category")
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
            col_type = user_column_types.get(selected_column, "Text")

            if col_type == "Categorical" or col_type == "Ordinal":
                value_counts = df_typed[selected_column].value_counts(dropna=False).reset_index()
                value_counts.columns = [selected_column, "Count"]
                value_counts["Percent"] = (value_counts["Count"] / value_counts["Count"].sum() * 100).round(2)
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
                    st.image(wordcloud.to_array(), use_column_width=True)
                else:
                    st.info("No text available to generate a word cloud.")

                if st.button("Generate text summary"):
                    if text.strip():
                        words = nltk.word_tokenize(text.lower())
                        stop_words = set(stopwords.words("english"))
                        words = [word for word in words if word.isalpha() and word not in stop_words]
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
            num_cols = [col for col, typ in user_column_types.items() if typ == "Numeric"]
            if num_cols:
                desc = df_typed[num_cols].describe().T
                st.dataframe(desc)
            else:
                st.info("No numeric columns have been defined.")

        # Frequency tables
        with tab_freq:
            st.subheader("Frequency tables for categorical and ordinal columns")
            cat_cols = [col for col, typ in user_column_types.items() if typ in ["Categorical", "Ordinal"]]
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
            num_cols = [col for col, typ in user_column_types.items() if typ == "Numeric"]
            if len(num_cols) >= 2:
                corr = df_typed[num_cols].corr()
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

    st.markdown("## Step 3: Statistical tests")

    cat_cols_all = [col for col, typ in user_column_types.items() if typ in ["Categorical", "Ordinal"]]
    num_cols_all = [col for col, typ in user_column_types.items() if typ == "Numeric"]

    # Chi square
    if len(cat_cols_all) >= 2:
        st.subheader("Chi square test")
        chi1 = st.selectbox("Categorical variable 1", cat_cols_all, key="chi1")
        chi2 = st.selectbox("Categorical variable 2", cat_cols_all, key="chi2")
        if st.button("Run chi square"):
            chi_table = pd.crosstab(df_typed[chi1], df_typed[chi2])
            chi2_stat, p_val, _, _ = chi2_contingency(chi_table)
            st.write(f"Chi square test p value: {p_val:.4f}")
            if p_val < 0.05:
                st.write("There is evidence of an association between the variables at the 5 percent level.")
            else:
                st.write("There is no strong evidence of an association between the variables at the 5 percent level.")
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
                if p_val < 0.05:
                    st.write("There is evidence of a difference in means across groups at the 5 percent level.")
                else:
                    st.write("There is no strong evidence of a difference in means across groups at the 5 percent level.")
            else:
                st.info("ANOVA requires at least two groups.")
    else:
        st.info("ANOVA needs at least one categorical or ordinal variable and one numeric variable.")

    # Regression
    if len(num_cols_all) >= 2:
        st.subheader("Linear regression")
        y = st.selectbox("Dependent variable (numeric)", num_cols_all, key="reg_y")
        x = st.selectbox("Independent variable", df_typed.columns, key="reg_x")
        if st.button("Run regression"):
            x_type = user_column_types.get(x, "Numeric")

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

    st.markdown("## Step 4: Export summaries to Excel")

    if st.button("Generate Excel summary"):
        buffer = create_summary_excel(df_typed, num_cols_all, cat_cols_all)
        st.download_button(
            label="Download Excel summary",
            data=buffer,
            file_name="evaluation_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
