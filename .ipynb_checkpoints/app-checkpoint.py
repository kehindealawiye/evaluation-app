
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.formula.api as smf
from wordcloud import WordCloud
import io

st.set_page_config(page_title="Complete Survey Analysis Tool", layout="wide")
st.title("Survey Analysis Dashboard – Final Complete Version")

uploaded_file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])

def detect_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().sum() >= len(series) * 0.5:
            return "Datetime"
    except:
        pass
    if pd.api.types.is_string_dtype(series):
        if series.nunique() < 10:
            return "Categorical"
        elif series.nunique() < 30:
            return "Categorical"
        else:
            return "Text"
    return "Unknown"

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    column_types = {col: detect_type(df[col]) for col in df.columns}
    st.markdown("### Detected Variable Types")
    st.dataframe(pd.DataFrame.from_dict(column_types, orient='index', columns=['Type']))

    chart_mode = st.radio("Chart display mode", ["Interactive", "Static"], horizontal=True)

    st.markdown("## Multi-Chart Dashboard")
    for col in df.columns:
        col_type = column_types[col]
        st.markdown(f"### {col} ({col_type})")
        try:
            if col_type == "Categorical":
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "Count"]
                if chart_mode == "Interactive":
                    st.plotly_chart(px.bar(counts, x=col, y="Count", color=col))
                    st.plotly_chart(px.pie(counts, names=col, values="Count"))
                else:
                    fig, ax = plt.subplots()
                    sns.barplot(data=counts, x=col, y="Count", ax=ax)
                    st.pyplot(fig)
            elif col_type == "Numeric":
                if chart_mode == "Interactive":
                    st.plotly_chart(px.histogram(df, x=col))
                    st.plotly_chart(px.box(df, y=col))
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    sns.histplot(df[col], kde=True, ax=axs[0])
                    axs[0].set_title("Histogram")
                    sns.boxplot(y=df[col], ax=axs[1])
                    axs[1].set_title("Boxplot")
                    st.pyplot(fig)
        except Exception as e:
            st.warning(f"Chart error for {col}: {e}")

    st.markdown("## X–Y Chart Explorer")
    x_options = [col for col in df.columns if column_types[col] in ["Numeric", "Categorical", "Datetime"]]
    x_col = st.selectbox("X Variable", x_options)
    x_type = column_types[x_col]
    if x_type == "Numeric":
        y_options = [col for col in df.columns if col != x_col and column_types[col] == "Numeric"]
    elif x_type in ["Categorical", "Datetime"]:
        y_options = [col for col in df.columns if col != x_col and column_types[col] == "Numeric"]
    else:
        y_options = []

    if not y_options:
        st.warning("No compatible Y variables found. Try another X.")
    else:
        y_col = st.selectbox("Y Variable", y_options)
        df_chart = df[[x_col, y_col]].dropna()
        try:
            if chart_mode == "Interactive":
                if x_type == "Numeric":
                    st.plotly_chart(px.scatter(df_chart, x=x_col, y=y_col))
                elif x_type == "Categorical":
                    st.plotly_chart(px.box(df_chart, x=x_col, y=y_col))
                elif x_type == "Datetime":
                    st.plotly_chart(px.line(df_chart.sort_values(x_col), x=x_col, y=y_col))
            else:
                fig, ax = plt.subplots()
                if x_type == "Numeric":
                    sns.scatterplot(data=df_chart, x=x_col, y=y_col, ax=ax)
                elif x_type == "Categorical":
                    sns.boxplot(data=df_chart, x=x_col, y=y_col, ax=ax)
                elif x_type == "Datetime":
                    ax.plot(df_chart[x_col], df_chart[y_col])
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Rendering failed: {e}")

    st.markdown("## Group Comparison Tool")
    cat_cols = [col for col in df.columns if column_types[col] == "Categorical"]
    num_cols = [col for col in df.columns if column_types[col] == "Numeric"]

    if cat_cols and num_cols:
        group_col = st.selectbox("Group by", cat_cols, key="group")
        value_col = st.selectbox("Compare numeric", num_cols, key="value")
        stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])
        st.dataframe(stats)

        if chart_mode == "Interactive":
            st.plotly_chart(px.box(df, x=group_col, y=value_col, title=f"{value_col} by {group_col}"))
        else:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=group_col, y=value_col, ax=ax)
            ax.set_title(f"{value_col} by {group_col}")
            st.pyplot(fig)

    st.markdown("## Correlation Matrix with Interpretation")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    st.markdown("## Word Cloud Generator")
    text_cols = [col for col in df.columns if column_types[col] in ["Text", "Categorical"]]
    if text_cols:
        word_col = st.selectbox("Column for Word Cloud", text_cols)
        text = " ".join(df[word_col].dropna().astype(str))
        if len(text.split()) > 3:
            wc = WordCloud(width=800, height=400).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Not enough text for Word Cloud.")
    else:
        st.warning("No valid text column found.")

    st.markdown("## Statistical Test Builder")
    dep = st.selectbox("Dependent Variable", df.columns)
    dep_type = column_types[dep]
    if dep_type == "Numeric":
        indep_options = [col for col in df.columns if col != dep and column_types[col] in ["Numeric", "Categorical"]]
    elif dep_type == "Categorical":
        indep_options = [col for col in df.columns if col != dep and column_types[col] == "Categorical"]
    else:
        indep_options = []

    if indep_options:
        indep = st.selectbox("Independent Variable", indep_options)
        st.info("Chi-square (cat–cat), ANOVA (num–cat), or Regression (num–num)")
        df_test = df[[dep, indep]].dropna()
        if st.button("Run Test"):
            try:
                if len(df_test) < 10:
                    st.warning("Too few rows.")
                else:
                    if dep_type == "Categorical" and column_types[indep] == "Categorical":
                        table = pd.crosstab(df_test[dep], df_test[indep])
                        chi2, p, _, _ = chi2_contingency(table)
                        st.write(f"Chi-square p = {p:.4f}")
                        st.success("Significant" if p < 0.05 else "Not significant")
                    elif dep_type == "Numeric" and column_types[indep] == "Categorical":
                        groups = [g[dep] for _, g in df_test.groupby(indep)]
                        f_stat, p_val = f_oneway(*groups)
                        st.write(f"ANOVA p = {p_val:.4f}")
                        st.success("Significant" if p_val < 0.05 else "Not significant")
                    elif dep_type == "Numeric" and column_types[indep] == "Numeric":
                        model = smf.ols(f"{dep} ~ {indep}", data=df_test).fit()
                        st.text(model.summary())
                        st.success(f"Significant (R²={model.rsquared:.2f})" if model.pvalues[1] < 0.05 else f"Not significant (R²={model.rsquared:.2f})")
            except Exception as e:
                st.error(f"Test error: {e}")
    else:
        st.warning("No valid independent variable found.")

    st.markdown("## Export Tools")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Dataset", csv, "survey_data.csv", "text/csv")
