import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from scipy.stats import chi2_contingency
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
    - Correlation matrix (if any numeric columns exist)
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Numeric descriptive statistics
        if num_cols:
            desc = df_typed[num_cols].describe().T
            sheet_name = "Numeric_Describe"
            desc.to_excel(writer, sheet_name=sheet_name)
            ws = writer.sheets[sheet_name]

            if "mean" in desc.columns:
                chart = workbook.add_chart({"type": "column"})
                n_rows = len(desc.index)
                cat_first_row = 1
                cat_last_row = n_rows
                cat_col_idx = 0
                mean_col_idx = list(desc.columns).index("mean") + 1

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
                ws.insert_chart(cat_last_row + 3, 0, chart)

        # Categorical / ordinal frequency tables
        if cat_cols:
            cat_summaries = build_categorical_summary(df_typed, cat_cols)
            sheet_name = "Categorical_Frequencies"
            ws = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = ws

            start_row = 0
            for col, freq_df in cat_summaries.items():
                ws.write(start_row, 0, col)
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
                ws.insert_chart(start_row, 4, chart)

                start_row = cat_last_row + 5

        # Correlation matrix (optional)
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

    # Session state for type mapping
    if "filename" not in st.session_state:
        st.session_state["filename"] = None
    if "type_map" not in st.session_state:
        st.session_state["type_map"] = {}

    if st.session_state["filename"] != filename:
        st.session_state["filename"] = filename
        st.session_state["type_map"] = {}

    type_map = st.session_state["type_map"]

    with st.expander("View raw pandas dtypes (for information only)"):
        dtypes_df = pd.DataFrame(
            {"Variable": df.columns, "Detected dtype": df.dtypes.astype(str)}
        )
        st.dataframe(dtypes_df)

    st.markdown("## Manually Set Variable Types")

    # For evaluation surveys, default to Categorical
    type_options = ["Categorical", "Text", "Datetime", "Numeric", "Ordinal"]

    for i, col in enumerate(df.columns):
        current = type_map.get(col, "Categorical")
        if current not in type_options:
            current = "Categorical"

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
        tab_uni, tab_xy, tab_group, tab_tables, tab_corr = st.tabs(
            ["Univariate", "X–Y Chart Explorer", "Group Comparison", "Summary tables", "Correlation"]
        )

        # Univariate explorer
        with tab_uni:
            st.subheader("Univariate explorer")
            col = st.selectbox("Select a column", df_typed.columns, key="uni_col")
            col_type = type_map.get(col, "Categorical")

            if col_type in ["Categorical", "Ordinal"]:
                vc = df_typed[col].value_counts(dropna=False).reset_index()
                vc.columns = [col, "Count"]
                vc["Percent"] = (vc["Count"] / vc["Count"].sum() * 100).round(2)
                fig = px.bar(
                    vc,
                    x=col,
                    y="Count",
                    title=f"Distribution of {col}",
                    labels={col: col, "Count": "Count"},
                    color=col
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(vc)

            elif col_type == "Numeric":
                fig = px.histogram(
                    df_typed,
                    x=col,
                    nbins=20,
                    title=f"Distribution of {col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write("Summary statistics")
                st.dataframe(df_typed[col].describe())

            elif col_type == "Text":
                st.subheader("Word cloud and text summary")
                text_series = df_typed[col].dropna().astype(str)
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

                if st.button("Generate text summary", key="summary_button"):
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

        # X–Y Chart Explorer (categorical vs categorical)
        with tab_xy:
            st.subheader("X–Y Chart Explorer (categorical vs categorical)")

            cat_cols = [c for c, t in type_map.items() if t in ["Categorical", "Ordinal"]]

            if len(cat_cols) < 2:
                st.info("Select at least two columns as Categorical/Ordinal to use this explorer.")
            else:
                x_var = st.selectbox("X variable (group on this)", cat_cols, key="xy_x")
                y_var = st.selectbox("Y variable (categories inside each X)", cat_cols, key="xy_y")
                chart_type = st.selectbox(
                    "Chart type",
                    ["Side-by-side bar", "Stacked bar (count)", "Stacked bar (percent)", "Heatmap"],
                    key="xy_chart_type"
                )

                if x_var == y_var:
                    st.warning("Pick two different variables for X and Y.")
                else:
                    table = pd.crosstab(df_typed[x_var], df_typed[y_var])
                    st.write("Frequency table")
                    st.dataframe(table)

                    if chart_type in ["Side-by-side bar", "Stacked bar (count)", "Stacked bar (percent)"]:
                        if chart_type == "Stacked bar (percent)":
                            table_plot = table.div(table.sum(axis=1), axis=0) * 100
                            value_label = "Percent"
                        else:
                            table_plot = table
                            value_label = "Count"

                        long_df = (
                            table_plot.reset_index()
                            .melt(id_vars=x_var, var_name=y_var, value_name=value_label)
                        )

                        fig = px.bar(
                            long_df,
                            x=x_var,
                            y=value_label,
                            color=y_var,
                            title=f"{y_var} by {x_var}",
                        )
                        if chart_type.startswith("Stacked"):
                            fig.update_layout(barmode="stack")
                        else:
                            fig.update_layout(barmode="group")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Heatmap":
                        fig = px.imshow(
                            table,
                            labels=dict(x=y_var, y=x_var, color="Count"),
                            aspect="auto",
                            text_auto=True,
                            title=f"Heatmap of {y_var} by {x_var}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # Group Comparison Tool (frequency comparison by group)
        with tab_group:
            st.subheader("Group Comparison Tool")

            cat_cols = [c for c, t in type_map.items() if t in ["Categorical", "Ordinal"]]

            if len(cat_cols) < 2:
                st.info("Select at least two columns as Categorical/Ordinal to use this tool.")
            else:
                group_col = st.selectbox("Group by", cat_cols, key="group_by")
                compare_col = st.selectbox("Question to compare", cat_cols, key="group_compare")
                group_chart_type = st.selectbox(
                    "Group chart type",
                    ["Bar (count)", "Bar (percent)"],
                    key="group_chart_type"
                )

                if group_col == compare_col:
                    st.warning("Group by and Question to compare must be different.")
                else:
                    table = pd.crosstab(df_typed[group_col], df_typed[compare_col])
                    st.write("Group comparison table")
                    st.dataframe(table)

                    if group_chart_type == "Bar (percent)":
                        table_plot = table.div(table.sum(axis=1), axis=0) * 100
                        value_label = "Percent"
                    else:
                        table_plot = table
                        value_label = "Count"

                    long_df = (
                        table_plot.reset_index()
                        .melt(id_vars=group_col, var_name=compare_col, value_name=value_label)
                    )

                    fig = px.bar(
                        long_df,
                        x=group_col,
                        y=value_label,
                        color=compare_col,
                        title=f"{compare_col} distribution within {group_col}",
                    )
                    fig.update_layout(barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

        # Summary tables tab – quick overview of all categorical variables
        with tab_tables:
            st.subheader("Summary frequency tables")
            cat_cols_all = [c for c, t in type_map.items() if t in ["Categorical", "Ordinal"]]
            if not cat_cols_all:
                st.info("No categorical/ordinal columns defined.")
            else:
                sel = st.selectbox("Select a column", cat_cols_all, key="summary_freq_col")
                vc = df_typed[sel].value_counts(dropna=False)
                total = vc.sum()
                freq_df = (
                    vc.reset_index()
                    .rename(columns={"index": sel, sel: "Count"})
                )
                freq_df["Percent"] = (freq_df["Count"] / total * 100).round(2)
                st.dataframe(freq_df)

        # Correlation tab (only if numeric variables exist)
        with tab_corr:
            st.subheader("Correlation (numeric variables, if any)")
            num_cols = [c for c, t in type_map.items() if t == "Numeric"]
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
                st.info("No or insufficient numeric columns for correlation.")

    # Step 3: Chi-square tests for categorical variables
    if uploaded_file is not None:
        st.markdown("## Step 3: Chi-square test (association between two categorical questions)")

        cat_cols_all = [c for c, t in type_map.items() if t in ["Categorical", "Ordinal"]]
        num_cols_all = [c for c, t in type_map.items() if t == "Numeric"]

        if len(cat_cols_all) >= 2:
            chi1 = st.selectbox("Variable 1", cat_cols_all, key="chi1")
            chi2 = st.selectbox("Variable 2", cat_cols_all, key="chi2")
            if st.button("Run chi-square test"):
                if chi1 == chi2:
                    st.warning("Please select two different variables.")
                else:
                    chi_table = pd.crosstab(df_typed[chi1], df_typed[chi2])
                    chi2_stat, p_val, _, _ = chi2_contingency(chi_table)
                    st.write(f"Chi-square p-value: {p_val:.4f}")
                    st.dataframe(chi_table)
        else:
            st.info("Chi-square test needs at least two categorical or ordinal variables.")

        st.markdown("## Step 4: Download Excel report (tables + charts)")

        if st.button("Generate Excel report"):
            buffer = create_report_excel(
                df_typed,
                num_cols_all,
                [c for c in type_map if type_map[c] in ["Categorical", "Ordinal"]]
            )
            st.download_button(
                label="Download Excel report",
                data=buffer,
                file_name="evaluation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
