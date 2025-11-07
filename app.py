
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

st.set_page_config(
    page_title="Personal Loan Propensity - Universal Bank",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Personal Loan Propensity Dashboard")
st.caption("For Marketing Head – Lead Prioritization, Targeting & Conversion Optimization")

@st.cache_data
def load_base_data():
    df = pd.read_csv("UniversalBank.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

def get_feature_target(df):
    target_col = "Personal Loan"
    id_col = "ID" if "ID" in df.columns else "Id"
    X = df.drop(columns=[c for c in [target_col, id_col] if c in df.columns])
    y = df[target_col].astype(int)
    return X, y

def train_models(df, test_size=0.2, random_state=42):
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
            max_depth=5,
            min_samples_leaf=25
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosted Tree": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )
    }

    metrics_rows = []
    roc_data = {}
    cms = {}
    feature_imps = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        y_tr_proba = model.predict_proba(X_train)[:, 1]
        y_te_proba = model.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, y_tr_pred)
        test_acc = accuracy_score(y_test, y_te_pred)
        precision = precision_score(y_test, y_te_pred, zero_division=0)
        recall = recall_score(y_test, y_te_pred, zero_division=0)
        f1 = f1_score(y_test, y_te_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_te_proba)

        metrics_rows.append([
            name,
            round(train_acc, 4),
            round(test_acc, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(auc, 4)
        ])

        fpr, tpr, _ = roc_curve(y_test, y_te_proba)
        roc_data[name] = (fpr, tpr, auc)

        cms[name] = {
            "train": confusion_matrix(y_train, y_tr_pred),
            "test": confusion_matrix(y_test, y_te_pred),
        }

        if hasattr(model, "feature_importances_"):
            feature_imps[name] = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)

        trained_models[name] = model

    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "Algorithm",
            "Training Accuracy",
            "Testing Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC"
        ]
    ).sort_values("AUC", ascending=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name, base_model in models.items():
        aucs = cross_val_score(
            base_model,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        cv_rows.append([name, round(aucs.mean(), 4), round(aucs.std(), 4)])
    cv_df = pd.DataFrame(
        cv_rows,
        columns=["Algorithm", "CV AUC (mean)", "CV AUC (std)"]
    )

    return {
        "metrics_df": metrics_df,
        "cv_df": cv_df,
        "roc_data": roc_data,
        "cms": cms,
        "feature_imps": feature_imps,
        "trained_models": trained_models,
        "X_columns": X.columns
    }

def plot_confusion(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Loan (0)", "Loan (1)"],
        yticklabels=["No Loan (0)", "Loan (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

def plot_feature_importance(series, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=series.values,
        y=series.index,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

if "model_result" not in st.session_state:
    st.session_state.model_result = None

tab1, tab2, tab3 = st.tabs(
    ["📊 Customer Insights", "🤖 Model Performance", "📂 Predict New Data"]
)

with tab1:
    df = load_base_data()
    st.subheader("Customer Profile & Loan Uptake Overview")
    st.dataframe(df.head(), use_container_width=True)

    data = df.copy()
    data["IncomeBand"] = pd.cut(
        data["Income"],
        bins=[0, 40, 60, 80, 120, 1000],
        labels=["<40k", "40-60k", "60-80k", "80-120k", "120k+"]
    )
    pivot = (
        data.groupby(["IncomeBand", "Education"])["Personal Loan"]
        .mean()
        .reset_index()
    )
    st.markdown("#### 1️⃣ Loan Acceptance by Income Band & Education")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=pivot,
        x="IncomeBand",
        y="Personal Loan",
        hue="Education",
        ax=ax1
    )
    ax1.set_ylabel("Loan Acceptance Rate")
    ax1.set_xlabel("Income Band")
    st.pyplot(fig1)

    st.markdown("#### 2️⃣ Heatmap: Card Spend vs Online Banking vs Loan Acceptance")
    data["CCBand"] = pd.cut(
        data["CCAvg"],
        bins=[-0.01, 0.5, 1, 2, 3, 10],
        labels=["≤0.5k", "0.5-1k", "1-2k", "2-3k", "3k+"]
    )
    heat = (
        data.groupby(["CCBand", "Online"])["Personal Loan"]
        .mean()
        .unstack()
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax2
    )
    st.pyplot(fig2)

    st.markdown("#### 3️⃣ Income Distribution by Loan Uptake (Boxplot)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df,
        x="Personal Loan",
        y="Income",
        ax=ax3
    )
    st.pyplot(fig3)

    st.markdown("#### 4️⃣ Family Size & Household Potential")
    fam = (
        df.groupby("Family")["Personal Loan"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=fam,
        x="Family",
        y="mean",
        ax=ax4
    )
    ax4.set_ylabel("Loan Acceptance Rate")
    ax4.set_xlabel("Family Size")
    st.pyplot(fig4)

    st.markdown("#### 5️⃣ High-Value Scatter: Income vs Card Spend")
    fig5, ax5 = plt.subplots(figsize=(7, 4))
    sample = df.sample(min(1500, len(df)), random_state=42)
    scatter = ax5.scatter(
        sample["Income"],
        sample["CCAvg"],
        c=sample["Personal Loan"],
        cmap="coolwarm",
        alpha=0.7
    )
    ax5.set_xlabel("Income ($000)")
    ax5.set_ylabel("CCAvg ($000/month)")
    fig5.colorbar(scatter, ax=ax5, label="Personal Loan (0/1)")
    st.pyplot(fig5)

with tab2:
    st.subheader("Train & Compare Models")
    base_df = load_base_data()

    if st.button("🚀 Run All 3 Models (DT, RF, GBT)"):
        st.session_state.model_result = train_models(base_df)

    if st.session_state.model_result is None:
        st.info("Click **Run All 3 Models** to train and view performance.")
    else:
        res = st.session_state.model_result

        st.markdown("### Overall Performance Metrics (Test Set)")
        st.dataframe(res["metrics_df"], use_container_width=True)

        st.markdown("### 5-Fold Cross-Validation (AUC)")
        st.dataframe(res["cv_df"], use_container_width=True)

        st.markdown("### ROC Curves (All Algorithms)")
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, (fpr, tpr, auc) in res["roc_data"].items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Confusion Matrices (Train vs Test)")
        for name, cms in res["cms"].items():
            col1, col2 = st.columns(2)
            with col1:
                plot_confusion(cms["train"], f"{name} - Train")
            with col2:
                plot_confusion(cms["test"], f"{name} - Test")

        st.markdown("### Feature Importances")
        for name, fi in res["feature_imps"].items():
            plot_feature_importance(fi, f"{name} - Top Predictors")

with tab3:
    st.subheader("Upload New Customer File & Score")
    st.markdown(
        "Upload CSV with same structure as `UniversalBank.csv`. The best-performing model will score Personal Loan propensity."
    )

    uploaded = st.file_uploader("Upload new customer CSV", type=["csv"])

    threshold = st.slider(
        "Prediction Threshold for 'Interested in Personal Loan'",
        0.1,
        0.9,
        0.5,
        0.05
    )

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        new_df.columns = [c.strip() for c in new_df.columns]

        if st.session_state.model_result is None:
            st.session_state.model_result = train_models(load_base_data())

        res = st.session_state.model_result
        metrics_df = res["metrics_df"]
        best_algo = metrics_df.iloc[0]["Algorithm"]
        best_model = res["trained_models"][best_algo]
        feature_cols = list(res["X_columns"])

        st.write(f"Using **{best_algo}** as champion model.")

        missing = [c for c in feature_cols if c not in new_df.columns]
        extra = [c for c in new_df.columns if c not in feature_cols and c not in ['ID', 'Personal Loan']]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X_new = new_df[feature_cols].copy()
            probs = best_model.predict_proba(X_new)[:, 1]
            preds = (probs >= threshold).astype(int)

            scored = new_df.copy()
            scored['Predicted_Personal_Loan_Prob'] = probs
            scored['Predicted_Personal_Loan'] = preds

            st.markdown("### Preview of Scored Data")
            st.dataframe(scored.head(), use_container_width=True)

            csv_bytes = scored.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Scored Dataset (CSV)",
                data=csv_bytes,
                file_name="scored_personal_loan_customers.csv",
                mime="text/csv"
            )

            if extra:
                st.caption(f"Ignored extra columns: {extra}")
    else:
        st.info("Upload a CSV to score new customers.")
