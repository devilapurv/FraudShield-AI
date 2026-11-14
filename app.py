import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="FraudShield AI | Risk Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {color-scheme:dark}
.stApp {background: radial-gradient(ellipse at 80% 0%,#261014 0%,transparent 40%),#0b0b0e;color:#f5f5f7;font-family:'DM Sans',sans-serif}
[data-testid="stHeader"] {background:transparent}
.block-container {max-width:1280px;padding-top:1.3rem;padding-bottom:4rem}
#MainMenu,footer {visibility:hidden}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
h1 {font-size:3.4rem!important}
h3 {font-size:1.3rem!important}
.brand {font-family:'Space Grotesk',sans-serif;font-size:1.35rem;font-weight:800;letter-spacing:.05em}
.brand span {color:#e50914}
.nav {display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #29252a;padding:5px 0 20px;margin-bottom:30px}
.nav-right {color:#b6aeb2;font-size:.85rem;letter-spacing:.08em}
.hero {background:linear-gradient(100deg,rgba(11,11,14,.95) 15%,rgba(11,11,14,.7) 60%,rgba(80,6,13,.5)),radial-gradient(circle at 90% 40%,#85111c,#171014 55%);border:1px solid #3b2228;border-radius:18px;padding:50px 46px;margin-bottom:24px;min-height:270px}
.eyebrow {color:#f25560;letter-spacing:.18em;font-size:.75rem;font-weight:800}
.hero-title {font-family:'Space Grotesk',sans-serif;font-size:clamp(2.3rem,5vw,4.2rem);font-weight:700;line-height:1.07;letter-spacing:-.05em;margin:13px 0}
.hero-copy {max-width:570px;color:#c9c1c5;font-size:1.03rem;line-height:1.65}
.pill {display:inline-block;background:#40151b;border:1px solid #9a2933;color:#ffc8cc;border-radius:30px;padding:6px 13px;font-size:.78rem;margin-top:14px}
.section-label {color:#f4f0f2;font-size:1.35rem;font-weight:800;margin:28px 0 12px}
.stat {background:#19171b;border:1px solid #302a30;border-radius:14px;padding:22px 20px;min-height:130px}
.stat-label {font-size:.8rem;color:#b8afb5;text-transform:uppercase;letter-spacing:.09em}
.stat-num {font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;margin:12px 0 3px}
.stat-sub {font-size:.8rem;color:#bcb4ba}
.stFileUploader {background:#171519;border:1px dashed #7a3039;border-radius:16px;padding:16px}
.stButton>button,.stDownloadButton>button {background:#e50914!important;border:0!important;color:white!important;border-radius:8px!important;font-weight:700!important;padding:.65rem 1.3rem!important}
.stButton>button:hover,.stDownloadButton>button:hover {background:#f52d37!important}
[data-testid="stDataFrame"], [data-testid="stAlert"] {border-radius:12px;overflow:hidden}
hr {border-color:#31272c!important}
.small-note {color:#9f969c;font-size:.84rem;line-height:1.6}
[data-testid='stMetric'] {background:#19171b;border:1px solid #302a30;padding:16px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nav"><div class="brand"><span>◆</span> FRAUDSHIELD <span>AI</span></div><div class="nav-right">RISK INTELLIGENCE / ANALYTICS</div></div>', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">● INTELLIGENT TRANSACTION MONITORING</div><div class="hero-title">Spot the risk.<br><span style="color:#ed1b2f">Stay ahead.</span></div><div class="hero-copy">Upload anonymized card transactions to screen for potential fraud with an XGBoost machine-learning model. Review risk scores, investigate alerts, and export results.</div><div class="pill">✦ XGBoost-powered · Batch fraud screening</div></div>''', unsafe_allow_html=True)

MODEL_FILE = 'fraud_model.pkl'
FEATURE_FILE = 'features.pkl'
THRESHOLD = 0.7  # Experimental threshold; selected using test data, not independently validated.

with st.expander('How it works · Input requirements', expanded=False):
    st.write('Upload a CSV with the original Kaggle creditcard.csv input columns: Time, V1–V28 and Amount. The Class label is optional and ignored for prediction. Ordinary bank statements are not supported by this trained model.')
    st.caption('The anonymized V1–V28 features cannot be reconstructed from ordinary bank statement fields. Model probabilities may be uncalibrated; alerts require human review.')

if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURE_FILE):
    st.error('Model files missing. Add fraud_model.pkl and features.pkl to this folder, then rerun the app.')
    st.code('''import joblib
joblib.dump(model, "fraud_model.pkl")
joblib.dump(X_train.columns.tolist(), "features.pkl")''', language='python')
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURE_FILE)
except Exception as exc:
    st.error(f'Could not load model files: {exc}')
    st.stop()

st.markdown('<div class="section-label">01 / Upload transactions</div>', unsafe_allow_html=True)
upload = st.file_uploader('Select a CSV file', type=['csv'], help='CSV must include all original model input features.')

if upload is None:
    st.markdown('<div class="small-note">Your dashboard will appear here after uploading a compatible CSV. For a quick test, export X_test.head(100).to_csv("sample.csv", index=False) from Colab.</div>', unsafe_allow_html=True)
    st.stop()

try:
    df = pd.read_csv(upload)
except Exception as exc:
    st.error(f'Unable to read CSV: {exc}')
    st.stop()

if df.empty:
    st.warning('The uploaded file contains no rows.')
    st.stop()

missing = [f for f in features if f not in df.columns]
if missing:
    st.error('Missing required columns: ' + ', '.join(missing))
    st.stop()

try:
    X = df.loc[:, features].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X.to_numpy()).all():
        st.error('Input features contain missing or infinite values. Clean the CSV and try again.')
        st.stop()
    probabilities = model.predict_proba(X)[:, 1]
except Exception as exc:
    st.error(f'Prediction failed: {exc}')
    st.stop()

predictions = (probabilities >= THRESHOLD).astype(int)
results = df.copy()
results['Fraud_Probability'] = np.round(probabilities, 5)
results['Risk_Score'] = np.round(probabilities * 100, 1)
results['Prediction'] = predictions
results['Status'] = np.where(predictions == 1, 'Potential Fraud', 'Legitimate')
results['Risk_Level'] = np.select([probabilities >= THRESHOLD, probabilities >= 0.3], ['High', 'Medium'], default='Low')

n = len(results)
flagged = int(predictions.sum())
legit = n - flagged
rate = 100 * flagged / n
st.markdown('<div class="section-label">02 / Screening overview</div>', unsafe_allow_html=True)
cols = st.columns(4)
for col, label, value, sub, color in zip(cols, ['TRANSACTIONS', 'LEGITIMATE', 'FLAGGED', 'ALERT RATE'], [f'{n:,}', f'{legit:,}', f'{flagged:,}', f'{rate:.2f}%'], ['Uploaded records', 'Below alert threshold', 'Requires review', 'Share of uploaded records'], ['#fff', '#7ee4ac', '#ff5762', '#fff']):
    with col:
        st.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-num" style="color:{color}">{value}</div><div class="stat-sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">03 / Explore risk</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.markdown('**Prediction distribution**')
    chart_df = pd.DataFrame({'Status': ['Legitimate', 'Potential Fraud'], 'Transactions': [legit, flagged]})
    chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X('Status:N', sort=['Legitimate', 'Potential Fraud'], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Transactions:Q', title='Transactions'),
        color=alt.Color('Status:N', scale=alt.Scale(domain=['Legitimate', 'Potential Fraud'], range=['#777982', '#e50914']), legend=None),
        tooltip=['Status', 'Transactions']
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)
with right:
    st.markdown('**Risk score distribution**')
    bins = pd.cut(results['Risk_Score'], bins=[-0.01, 10, 30, 50, 70, 90, 100], labels=['0–10', '10–30', '30–50', '50–70', '70–90', '90–100'], include_lowest=True)
    counts = bins.value_counts().reindex(bins.cat.categories, fill_value=0)
    hist = pd.DataFrame({'Risk band': counts.index.astype(str), 'Transactions': counts.values})
    chart = alt.Chart(hist).mark_bar(color='#e50914', cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X('Risk band:N', sort=list(hist['Risk band']), title='Risk score (%)', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Transactions:Q', title='Transactions'),
        tooltip=['Risk band', 'Transactions']
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

st.markdown('<div class="section-label">04 / Transaction explorer</div>', unsafe_allow_html=True)
filter_choice = st.selectbox('Filter transactions', ['All transactions', 'Potential fraud only', 'Legitimate only'])
shown = results
if filter_choice == 'Potential fraud only':
    shown = results[results['Prediction'] == 1]
elif filter_choice == 'Legitimate only':
    shown = results[results['Prediction'] == 0]
front = ['Status', 'Risk_Level', 'Risk_Score', 'Fraud_Probability']
st.dataframe(shown[front + [c for c in shown.columns if c not in front]], use_container_width=True, hide_index=True, height=360)
st.caption(f'Showing {len(shown):,} of {len(results):,} uploaded transactions. Rows retain their original upload order.')

st.markdown('<div class="section-label">05 / Investigate a transaction</div>', unsafe_allow_html=True)
st.write('Select a row number from the uploaded file to inspect its risk score and model explanation.')
row_number = st.number_input('Transaction row (starting at 1)', min_value=1, max_value=n, value=1, step=1)
row_idx = int(row_number) - 1
selected_prob = float(probabilities[row_idx])
col_a, col_b, col_c = st.columns(3)
col_a.metric('Risk score', f'{selected_prob*100:.1f}/100')
col_b.metric('Risk level', str(results.iloc[row_idx]['Risk_Level']))
col_c.metric('Screening result', 'Flagged' if predictions[row_idx] else 'Legitimate')
st.progress(min(max(selected_prob, 0.0), 1.0), text='Model score (not a calibrated probability)')
st.caption('High: score ≥ 70 · Medium: 30–69.9 · Low: below 30. These are demonstration thresholds, not validated risk categories.')

with st.expander('Explain this prediction with SHAP', expanded=False):
    st.caption('SHAP shows which anonymized features push the model score up or down. V1–V28 do not have human-readable banking meanings.')
    if st.button('Generate SHAP explanation', type='primary'):
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            explanation = explainer(X.iloc[[row_idx]])
            shap_values = np.asarray(explanation.values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]
            impacts = pd.DataFrame({'Feature': features, 'SHAP impact': shap_values[0]})
            impacts['Absolute impact'] = impacts['SHAP impact'].abs()
            impacts = impacts.nlargest(12, 'Absolute impact').sort_values('SHAP impact')
            impacts['Direction'] = np.where(impacts['SHAP impact'] >= 0, 'Raises fraud score', 'Lowers fraud score')
            shap_chart = alt.Chart(impacts).mark_bar().encode(
                x=alt.X('SHAP impact:Q', title='SHAP contribution (model output scale)'),
                y=alt.Y('Feature:N', sort=alt.SortField(field='SHAP impact', order='ascending'), title=None),
                color=alt.Color('Direction:N', scale=alt.Scale(domain=['Raises fraud score', 'Lowers fraud score'], range=['#e50914', '#777982'])),
                tooltip=['Feature', alt.Tooltip('SHAP impact:Q', format='.3f'), 'Direction']
            ).properties(height=350)
            st.altair_chart(shap_chart, use_container_width=True)
            st.caption('SHAP values may be in raw model-margin units, not percentage points. They explain model behavior, not causality or confirmed fraud.')
        except ImportError:
            st.error('SHAP is not installed. Run: python -m pip install shap')
        except Exception as exc:
            st.error(f'Could not generate SHAP explanation: {exc}')

st.markdown('<div class="section-label">06 / Export screening results</div>', unsafe_allow_html=True)
st.download_button('↓ Download all screened transactions', data=results.to_csv(index=False).encode('utf-8'), file_name='fraudshield_predictions.csv', mime='text/csv')
st.download_button('↓ Download filtered transactions', data=shown.to_csv(index=False).encode('utf-8'), file_name='fraudshield_filtered.csv', mime='text/csv')
st.markdown('<div class="small-note">Research demonstration only. Model scores are not necessarily calibrated probabilities. The 0.70 alert threshold was explored on the test set and requires independent validation. Do not use this app to make automated financial decisions.</div>', unsafe_allow_html=True)
