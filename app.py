
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
st.title("Survey Analysis Dashboard – A Complete Survey Analysis Tool")

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
                st.markdown('**Chart Data Table:**')
                st.dataframe(df_chart if 'df_chart' in locals() else df)
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
                st.markdown('**Chart Data Table:**')
                st.dataframe(df_chart if 'df_chart' in locals() else df)
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
        chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Box"])
        df_chart = df[[x_col, y_col]].dropna()
        try:
            if chart_mode == "Interactive":
                if chart_type == "Scatter":
                    st.plotly_chart(px.scatter(df_chart, x=x_col, y=y_col))
                elif chart_type == "Line":
                    st.plotly_chart(px.line(df_chart.sort_values(x_col), x=x_col, y=y_col))
                elif chart_type == "Box":
                    st.plotly_chart(px.box(df_chart, x=x_col, y=y_col))
            else:
                fig, ax = plt.subplots()
                if chart_type == "Scatter":
                    sns.scatterplot(data=df_chart, x=x_col, y=y_col, ax=ax)
                elif chart_type == "Line":
                    ax.plot(df_chart[x_col], df_chart[y_col])
                elif chart_type == "Box":
                    sns.boxplot(data=df_chart, x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)
                st.markdown('**Chart Data Table:**')
                st.dataframe(df_chart if 'df_chart' in locals() else df)
        except Exception as e:
            st.warning(f"Rendering failed: {e}")

    st.markdown("## Group Comparison Tool")
    cat_cols = [col for col in df.columns if column_types[col] == "Categorical"]
    num_cols = [col for col in df.columns if column_types[col] == "Numeric"]

    if cat_cols and num_cols:
        group_col = st.selectbox("Group by", cat_cols, key="group")
        value_col = st.selectbox("Compare numeric", num_cols, key="value")
        group_chart = st.selectbox("Group Chart Type", ["Box", "Violin", "Bar"])
        stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])
        st.dataframe(stats)

        if chart_mode == "Interactive":
            if group_chart == "Box":
                st.plotly_chart(px.box(df, x=group_col, y=value_col))
            elif group_chart == "Violin":
                st.plotly_chart(px.violin(df, x=group_col, y=value_col, box=True, points="all"))
            elif group_chart == "Bar":
                bar_df = df.groupby(group_col)[value_col].mean().reset_index()
                st.plotly_chart(px.bar(bar_df, x=group_col, y=value_col))
        else:
            fig, ax = plt.subplots()
            if group_chart == "Box":
                sns.boxplot(data=df, x=group_col, y=value_col, ax=ax)
            elif group_chart == "Violin":
                sns.violinplot(data=df, x=group_col, y=value_col, ax=ax)
            elif group_chart == "Bar":
                bar_df = df.groupby(group_col)[value_col].mean().reset_index()
                sns.barplot(data=bar_df, x=group_col, y=value_col, ax=ax)
            ax.set_title(f"{value_col} by {group_col}")
            st.pyplot(fig)
            st.markdown('**Chart Data Table:**')
            st.dataframe(df_chart if 'df_chart' in locals() else df)
            
    st.markdown("## Regression, ANOVA, and Chi-Square Tests")
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