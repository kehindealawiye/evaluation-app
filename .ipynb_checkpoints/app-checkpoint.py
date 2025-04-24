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
        if series.nunique() < 30:
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
                    fig_bar = px.bar(counts, x=col, y="Count", color=col)
                    fig_bar.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_bar)
                    st.markdown('**Chart Data Table:**')
                    st.dataframe(counts)
                    fig_pie = px.pie(counts, names=col, values="Count")
                    st.plotly_chart(fig_pie)
                else:
                    fig, ax = plt.subplots()
                    sns.barplot(data=counts, x=col, y="Count", ax=ax)
                    ax.tick_params(axis='x', labelrotation=45)
                    st.pyplot(fig)
                    st.markdown('**Chart Data Table:**')
                    st.dataframe(counts)

            elif col_type == "Numeric":
                if chart_mode == "Interactive":
                    fig_hist = px.histogram(df, x=col)
                    fig_hist.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_hist)
                    st.markdown('**Chart Data Table:**')
                    st.dataframe(df[[col]].dropna())
                    fig_box = px.box(df, y=col)
                    st.plotly_chart(fig_box)
                    st.markdown('**Chart Summary Table:**')
                    st.dataframe(df[[col]].describe().T)
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    sns.histplot(df[col], kde=True, ax=axs[0])
                    axs[0].set_title("Histogram")
                    sns.boxplot(y=df[col], ax=axs[1])
                    axs[1].set_title("Boxplot")
                    st.pyplot(fig)
                    st.markdown('**Chart Summary Table:**')
                    st.dataframe(df[[col]].describe().T)

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
                ax.tick_params(axis='x', labelrotation=45)
                st.pyplot(fig)
                st.markdown('**Chart Data Table:**')
                st.dataframe(df_chart)
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
            ax.tick_params(axis='x', labelrotation=45)
            st.pyplot(fig)

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
                        safe_dep = f'Q("{dep}")'
                        safe_indep = f'Q("{indep}")'
                        model = smf.ols(f"{safe_dep} ~ {safe_indep}", data=df_test).fit()
                        st.text(model.summary())
                        st.success(f"Significant (R² = {model.rsquared:.2f})" if model.pvalues[1] < 0.05 else f"Not significant (R² = {model.rsquared:.2f})")
            except Exception as e:
                st.error(f"Test error: {e}")
    else:
        st.warning("No valid independent variable found.")

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

    st.markdown("## Correlation Matrix with Interpretation")
    num_cols = [col for col in df.columns if column_types[col] == "Numeric"]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.tick_params(axis='x', labelrotation=45)
        st.pyplot(fig)
        st.markdown('**Chart Data Table:**')
        st.dataframe(df[num_cols].corr())

        if 'df' in locals():
            st.markdown("## Pivot Summary Explorer")

        # User selects what to pivot
        row_group = st.multiselect("Row Group(s)", df.columns, key="pivot_row")
        col_group = st.multiselect("Column Group(s)", df.columns, key="pivot_col")
        value_cols = st.multiselect("Value Column(s)", df.select_dtypes(include=np.number).columns, key="pivot_values")
        agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max"], key="pivot_agg")


        # Optional filtering
        st.markdown("### Optional Filter")
        filter_col = st.selectbox("Filter Column (Optional)", ["None"] + list(df.columns), key="pivot_filter_col")
        if filter_col != "None":
            filter_vals = df[filter_col].dropna().unique()
            selected_filter = st.multiselect("Filter Value(s)", filter_vals, key="pivot_filter_val")
            if selected_filter:
                df = df[df[filter_col].isin(selected_filter)]

        # Build and show pivot table
    if value_cols:
        pivot_table = pd.pivot_table(
            df,
            index=row_group,
            columns=None if col_group == "None" else col_group,
            values=value_cols,
            aggfunc=agg_func
        )

        # Fix for JSON serialization: reset index and convert columns to string
        pivot_table = pivot_table.reset_index()
        pivot_table.columns = pivot_table.columns.map(str)

        st.dataframe(pivot_table)
    else:
        st.info("Please select at least one numeric column to summarize.")
    
    st.markdown("## Export Tools")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Dataset", csv, "survey_data.csv", "text/csv")

