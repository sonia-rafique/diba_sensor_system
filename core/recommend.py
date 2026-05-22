class DIBAClinicalRecommendationEngine:
    @staticmethod
    def evaluate_clinical_action(risk_score: int, tier: str, gas_label: str = '') -> tuple[list, list]:
        notifications = []
        gas_name = gas_label.title() if gas_label else 'VOC'

        if risk_score <= 25:
            actions = [
                f"Routine non-invasive breath telemetry re-screening in 12 months.",
                f"Log {gas_name} baseline to immutable patient record.",
                "No urgent laboratory or imaging diagnostic pathway required.",
                "Maintain standard healthy lifestyle advisories for preventive care.",
            ]

        elif risk_score <= 50:
            actions = [
                f"Schedule E-Nose breath matrix re-test within 15 days.",
                "Consult respiratory specialist for deep cellular breath analysis.",
                "Cross-reference prior breath biomarker history if available.",
                "Consider spirometry and chest X-ray as supplementary baseline checks.",
            ]
            notifications.append(
                f"SYSTEM ALERT: Moderate {gas_name} biomarker drift isolated on active tracking channel."
            )

        elif risk_score <= 75:
            actions = [
                "URGENT: Schedule Low-Dose CT (LDCT) thoracic scan within 72 hours.",
                "Fast-track diagnostic package delivery to oncology department.",
                "Initiate multi-biomarker correlation panel review with pulmonologist.",
                "Flag patient for weekly VOC monitoring across next 4 weeks.",
            ]
            notifications.append(
                f"URGENT ALARM: High-risk lung cancer VOC pattern caught — {gas_name} signature confirmed."
            )

        else:
            actions = [
                "CRITICAL: Immediate patient pathway isolation for biopsy confirmation.",
                "Deploy emergency fast-track multidisciplinary oncology care protocol.",
                "Contact thoracic surgery team within 24 hours for MDT case review.",
                "Activate hospital cancer registry notification and family counseling pathway.",
            ]
            notifications.append(
                f"CRITICAL BROADCAST: Direct intervention required — malignant {gas_name} signature matched."
            )

        return actions, notifications
