import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

LUNG_CANCER_VOC_RISK = {
    'toluene':      {'weight': 1.00, 'label': 'Toluene',       'risk': 'Very High'},
    'hexane':       {'weight': 0.90, 'label': 'Hexane',        'risk': 'High'},
    'ethyl acetate':{'weight': 0.80, 'label': 'Ethyl Acetate', 'risk': 'High'},
    'acetone':      {'weight': 0.70, 'label': 'Acetone',       'risk': 'Moderate'},
    'isopropanol':  {'weight': 0.55, 'label': 'Isopropanol',   'risk': 'Moderate'},
    'ethanol':      {'weight': 0.40, 'label': 'Ethanol',       'risk': 'Lower'},
}

FEATURES = [
    'gas_enc', 'ppb', 'sensor_mean', 'sensor_max',
    'sensor_min', 'sensor_std', 'sensor_range',
    'voc_risk_weight', 'concentration_normalized'
]


class DIBAOncologyEngine:
    def __init__(self, dataset_path="Dataset/gsalc.csv"):
        self.dataset_path = dataset_path
        self.models = {}
        self.metrics = {}
        self.best_model_name = None
        self.is_trained = False
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.features = FEATURES

    def _load_and_process(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at '{self.dataset_path}'. "
                "Please place gsalc.csv in the Dataset/ folder."
            )

        raw_df = pd.read_csv(self.dataset_path, header=None)
        rows = []

        gas_names = raw_df.iloc[:, 0].str.strip().str.lower().values
        self.label_encoder.fit(gas_names)

        for idx, row in raw_df.iterrows():
            gas_str = str(row.iloc[0]).strip().lower()
            ppb_str = str(row.iloc[1]).strip().lower().replace('ppb', '').strip()

            try:
                ppb_val = float(ppb_str)
            except ValueError:
                ppb_val = 100.0

            signals = pd.to_numeric(row.iloc[2:], errors='coerce').dropna().values
            if len(signals) == 0:
                continue

            gas_enc = self.label_encoder.transform([gas_str])[0]
            voc_weight = LUNG_CANCER_VOC_RISK.get(gas_str, {}).get('weight', 0.5)

            rows.append({
                'gas_label':             gas_str,
                'gas_enc':               float(gas_enc),
                'ppb':                   ppb_val,
                'sensor_mean':           float(np.mean(signals)),
                'sensor_max':            float(np.max(signals)),
                'sensor_min':            float(np.min(signals)),
                'sensor_std':            float(np.std(signals)),
                'sensor_range':          float(np.max(signals) - np.min(signals)),
                'voc_risk_weight':       voc_weight,
                'concentration_normalized': ppb_val / 200.0,
            })

        df = pd.DataFrame(rows)

        risk_score = (
            df['voc_risk_weight'] * 0.40 +
            df['concentration_normalized'] * 0.30 +
            df['sensor_range'] / df['sensor_range'].max() * 0.30
        )
        median_thresh = risk_score.median()
        df['target'] = (risk_score >= median_thresh).astype(int)

        if df['target'].nunique() < 2:
            half = len(df) // 2
            df.iloc[:half, df.columns.get_loc('target')] = 0
            df.iloc[half:, df.columns.get_loc('target')] = 1

        return df

    def train(self):
        df = self._load_and_process()
        X_raw = df[self.features]
        y = df['target']

        X_scaled = self.scaler.fit_transform(X_raw)
        X = pd.DataFrame(X_scaled, columns=self.features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        model_definitions = {
            'Random Forest':    RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
            'XGBoost':          XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                              eval_metric='logloss', random_state=42, verbosity=0),
            'SVM':              SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'Decision Tree':    DecisionTreeClassifier(max_depth=6, random_state=42),
        }

        best_f1 = -1
        self.best_model_name = 'Random Forest'

        for name, model in model_definitions.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            acc   = float(accuracy_score(y_test, preds))
            prec  = float(precision_score(y_test, preds, zero_division=0))
            rec   = float(recall_score(y_test, preds, zero_division=0))
            f1    = float(f1_score(y_test, preds, zero_division=0))
            auc   = float(roc_auc_score(y_test, probs))
            cm    = confusion_matrix(y_test, preds).tolist()
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1').tolist()

            self.metrics[name] = {
                'Accuracy':  acc,
                'Precision': prec,
                'Recall':    rec,
                'F1 Score':  f1,
                'AUC-ROC':   auc,
                'Confusion Matrix': cm,
                'CV F1 Scores': cv_scores,
                'CV F1 Mean': float(np.mean(cv_scores)),
            }

            self.models[name] = model

            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name

        self.is_trained = True
        return self.metrics, self.best_model_name

    def predict(self, raw_payload):
        if not self.is_trained:
            self.train()

        gas_str = str(raw_payload.get('gas_label', 'ethanol')).strip().lower()
        ppb_val = float(raw_payload.get('ppb', 100.0))
        s_mean  = float(raw_payload.get('sensor_mean', 0.35))
        s_max   = float(raw_payload.get('sensor_max', 3.90))
        s_min   = float(raw_payload.get('sensor_min', 0.33))
        s_std   = float(raw_payload.get('sensor_std', 0.97))
        s_range = float(raw_payload.get('sensor_range', 3.57))
        voc_w   = LUNG_CANCER_VOC_RISK.get(gas_str, {}).get('weight', 0.5)

        input_dict = {
            'gas_enc':               float(self.label_encoder.transform([gas_str])[0]),
            'ppb':                   ppb_val,
            'sensor_mean':           s_mean,
            'sensor_max':            s_max,
            'sensor_min':            s_min,
            'sensor_std':            s_std,
            'sensor_range':          s_range,
            'voc_risk_weight':       voc_w,
            'concentration_normalized': ppb_val / 200.0,
        }

        df_input = pd.DataFrame([input_dict])[self.features]
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_input), columns=self.features
        )

        best_model = self.models[self.best_model_name]
        prob = best_model.predict_proba(df_scaled)[0][1]
        score = int(prob * 100)

        if score <= 25:
            tier  = "Safe Tier"
            color = "#10B981"
        elif score <= 50:
            tier  = "Moderate Concern"
            color = "#F59E0B"
        elif score <= 75:
            tier  = "High Risk Tier"
            color = "#EF4444"
        else:
            tier  = "Critical Alert"
            color = "#7C3AED"

        voc_info = LUNG_CANCER_VOC_RISK.get(gas_str, {'risk': 'Unknown'})

        return {
            'score':     score,
            'tier':      tier,
            'color':     color,
            'prob':      round(prob, 4),
            'model_used': self.best_model_name,
            'voc_risk':  voc_info.get('risk', 'Unknown'),
            'gas_label': gas_str,
            'ppb':       ppb_val,
        }

    def get_feature_importance(self):
        if not self.is_trained:
            return {}
        model = self.models.get(self.best_model_name)
        if hasattr(model, 'feature_importances_'):
            fi = dict(zip(self.features, model.feature_importances_))
            return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        return {}

    def get_all_model_proba(self, raw_payload):
        if not self.is_trained:
            self.train()

        gas_str = str(raw_payload.get('gas_label', 'ethanol')).strip().lower()
        ppb_val = float(raw_payload.get('ppb', 100.0))
        s_mean  = float(raw_payload.get('sensor_mean', 0.35))
        s_max   = float(raw_payload.get('sensor_max', 3.90))
        s_min   = float(raw_payload.get('sensor_min', 0.33))
        s_std   = float(raw_payload.get('sensor_std', 0.97))
        s_range = float(raw_payload.get('sensor_range', 3.57))
        voc_w   = LUNG_CANCER_VOC_RISK.get(gas_str, {}).get('weight', 0.5)

        input_dict = {
            'gas_enc':               float(self.label_encoder.transform([gas_str])[0]),
            'ppb':                   ppb_val,
            'sensor_mean':           s_mean,
            'sensor_max':            s_max,
            'sensor_min':            s_min,
            'sensor_std':            s_std,
            'sensor_range':          s_range,
            'voc_risk_weight':       voc_w,
            'concentration_normalized': ppb_val / 200.0,
        }

        df_input = pd.DataFrame([input_dict])[self.features]
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_input), columns=self.features
        )

        results = {}
        for name, model in self.models.items():
            prob = model.predict_proba(df_scaled)[0][1]
            results[name] = round(float(prob) * 100, 1)

        return results
