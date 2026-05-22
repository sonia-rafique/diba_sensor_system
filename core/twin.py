import numpy as np

LUNG_CANCER_VOC_RISK = {
    'toluene':       {'weight': 1.00, 'risk': 'Very High'},
    'hexane':        {'weight': 0.90, 'risk': 'High'},
    'ethyl acetate': {'weight': 0.80, 'risk': 'High'},
    'acetone':       {'weight': 0.70, 'risk': 'Moderate'},
    'isopropanol':   {'weight': 0.55, 'risk': 'Moderate'},
    'ethanol':       {'weight': 0.40, 'risk': 'Lower'},
}


class VirtualPatientProfileTwin:
    def __init__(self, patient_id: str, baseline_payload: dict):
        self.patient_id = patient_id
        self.baseline   = baseline_payload

    def compute_progression_trajectory(self, current_risk_score: int) -> tuple[int, str, list]:
        ppb_level   = float(self.baseline.get('ppb', 100.0))
        s_max       = float(self.baseline.get('sensor_max', 3.9))
        gas_str     = str(self.baseline.get('gas_label', 'ethanol')).lower()
        voc_weight  = LUNG_CANCER_VOC_RISK.get(gas_str, {}).get('weight', 0.5)

        drift = 0
        if ppb_level >= 200.0:
            drift += 15
        elif ppb_level >= 100.0:
            drift += 5

        if s_max > 3.90:
            drift += 10
        elif s_max > 3.50:
            drift += 4

        drift = int(drift * voc_weight)
        projected = int(np.clip(current_risk_score + drift, 0, 100))

        if drift > 10:
            direction = "WORSENING"
            summary = (
                f"Digital twin simulation predicts significant risk escalation. "
                f"Score is projected to rise from {current_risk_score} → {projected} "
                f"within 60 days due to elevated {gas_str.title()} saturation at {ppb_level} PPB."
            )
        elif drift > 0:
            direction = "MILD INCREASE"
            summary = (
                f"Slight upward trend detected. Score may increase from "
                f"{current_risk_score} → {projected} over 60 days. "
                f"Continued monitoring recommended."
            )
        else:
            direction = "STABLE"
            summary = (
                f"Digital twin profile is STABLE. No significant VOC drift pattern "
                f"detected over the next 60-day simulation window. "
                f"Score remains at {current_risk_score}."
            )

        monthly_curve = []
        for month in range(1, 7):
            monthly_score = int(np.clip(
                current_risk_score + (drift * month / 6), 0, 100
            ))
            monthly_curve.append({'month': f"Month {month}", 'score': monthly_score})

        return projected, summary, direction, monthly_curve
