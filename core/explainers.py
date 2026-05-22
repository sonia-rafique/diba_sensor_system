import numpy as np

class DIBAExplainableAI:
    FEATURE_DISPLAY_NAMES = {
        'gas_enc':                 'VOC Gas Type Encoding',
        'ppb':                     'Gas Concentration (PPB)',
        'sensor_mean':             'Mean Absorption Signal',
        'sensor_max':              'Peak Signal Amplitude',
        'sensor_min':              'Baseline Signal Floor',
        'sensor_std':              'Signal Variability (Std)',
        'sensor_range':            'Signal Dynamic Range',
        'voc_risk_weight':         'VOC Oncological Risk Weight',
        'concentration_normalized':'Normalized Concentration Load',
    }

    @staticmethod
    def calculate_local_attribution(payload: dict) -> dict:
        ppb_val    = float(payload.get('ppb', 100.0))
        s_max      = float(payload.get('sensor_max', 3.9))
        s_mean     = float(payload.get('sensor_mean', 0.35))
        s_std      = float(payload.get('sensor_std', 0.97))
        s_range    = float(payload.get('sensor_range', 3.57))
        voc_weight = float(payload.get('voc_risk_weight', 0.5))

        raw = {
            'VOC Oncological Risk Weight':  voc_weight * 40.0,
            'Gas Concentration (PPB)':      ppb_val * 0.30,
            'Signal Dynamic Range':         s_range * 8.0,
            'Peak Signal Amplitude':        s_max * 5.0,
            'Mean Absorption Signal':       s_mean * 5.0,
            'Signal Variability (Std)':     s_std * 3.0,
        }

        total = sum(raw.values()) or 1.0
        normalized = {k: round(v / total, 4) for k, v in raw.items()}
        return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def generate_explanation_text(result: dict, attribution: dict) -> str:
        score     = result.get('score', 0)
        tier      = result.get('tier', 'Unknown')
        gas       = result.get('gas_label', 'unknown gas').title()
        ppb       = result.get('ppb', 100)
        voc_risk  = result.get('voc_risk', 'Unknown')
        top_feat  = list(attribution.keys())[0] if attribution else 'N/A'

        if score <= 25:
            outlook = (
                f"The exhaled breath sample containing {gas} at {ppb} PPB "
                f"shows NO significant malignant VOC signature. "
                f"Oncological risk classification: {tier}."
            )
        elif score <= 50:
            outlook = (
                f"Moderate VOC biomarker elevation detected for {gas} at {ppb} PPB. "
                f"Classification: {tier}. The '{top_feat}' parameter is the "
                f"primary driver of this risk score. Follow-up monitoring advised."
            )
        elif score <= 75:
            outlook = (
                f"HIGH-RISK VOC pattern identified: {gas} at {ppb} PPB with "
                f"inherent oncological risk level '{voc_risk}'. "
                f"Classification: {tier}. Immediate specialist review required."
            )
        else:
            outlook = (
                f"CRITICAL: Malignant-pattern VOC signature detected — {gas} at {ppb} PPB. "
                f"Risk category: {voc_risk}. Classification: {tier}. "
                f"Emergency multidisciplinary oncology pathway activation required."
            )

        return outlook
