"""Statistical analysis and root cause detection"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import re


class AnalyticsEngine:
    """Statistical analysis engine"""

    def __init__(self, data_loader):
        self.data = data_loader

    def detect_anomalies(self, metric: str, threshold: float = 2.0) -> pd.DataFrame:
        """Detect statistical anomalies using Z-score"""
        if metric not in self.data.df.columns:
            return pd.DataFrame()

        values = self.data.df[metric].dropna()
        if len(values) < 3:
            return pd.DataFrame()

        z_scores = np.abs(stats.zscore(values))
        anomalies = self.data.df.iloc[z_scores > threshold].copy()
        anomalies["z_score"] = z_scores[z_scores > threshold]

        return anomalies

    def root_cause_analysis(
        self, kpi_name: str, doctor: str = None, bu: str = None
    ) -> Dict:
        """Perform root cause analysis"""
        df = self.data.df.copy()

        if doctor:
            df = df[df["Doctor Name"] == doctor]
        if bu:
            df = df[df["BU"] == bu]

        if df.empty:
            return {"error": "No data available"}

        # Get latest vs previous
        latest_date = df["Date"].max()
        previous_date = latest_date - pd.Timedelta(days=30)

        current = df[df["Date"] == latest_date]
        previous = df[df["Date"] == previous_date]

        if current.empty or previous.empty:
            # Use first vs last if monthly comparison not available
            current = df.sort_values("Date").tail(1)
            previous = df.sort_values("Date").head(1)

        current_value = (
            current[kpi_name].iloc[0] if kpi_name in current.columns else None
        )
        previous_value = (
            previous[kpi_name].iloc[0] if kpi_name in previous.columns else None
        )

        if current_value is None:
            return {"error": f"KPI {kpi_name} not found"}

        if previous_value and previous_value != 0:
            variance_pct = (current_value - previous_value) / previous_value * 100
        else:
            variance_pct = 0

        # Get drivers from knowledge base
        drivers = self._get_drivers(kpi_name, current, previous)

        # Determine severity
        if abs(variance_pct) > 20:
            severity = "critical"
        elif abs(variance_pct) > 10:
            severity = "high"
        elif abs(variance_pct) > 5:
            severity = "medium"
        else:
            severity = "low"

        return {
            "kpi": kpi_name,
            "current_value": current_value,
            "previous_value": previous_value if previous_value else "N/A",
            "variance_pct": variance_pct,
            "trend": "declining"
            if variance_pct < -5
            else "improving"
            if variance_pct > 5
            else "stable",
            "severity": severity,
            "primary_drivers": drivers[:3],
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def _get_drivers(
        self, kpi_name: str, current: pd.DataFrame, previous: pd.DataFrame
    ) -> List[Dict]:
        """Get driver analysis"""
        # Common driver relationships
        driver_map = {
            "Total Revenue": [
                "No. Cases",
                "Charge per case",
                "Total Leakage Revenue Losses",
            ],
            "No. Cases": [
                "No. Booking",
                "Patient Retention %",
                "Patient Acquisition %",
            ],
            "No-Show %": ["Reminder Compliance", "Booking Quality"],
            "Service Leakage %": ["No. Missed Opportunity", "Workflow Compliance"],
        }

        drivers = driver_map.get(kpi_name, [])
        result = []

        for driver in drivers:
            if driver in current.columns:
                curr_val = current[driver].iloc[0]
                prev_val = (
                    previous[driver].iloc[0] if driver in previous.columns else curr_val
                )

                if prev_val and prev_val != 0:
                    change = (curr_val - prev_val) / prev_val * 100
                else:
                    change = 0

                result.append(
                    {
                        "driver": driver,
                        "current": curr_val,
                        "previous": prev_val,
                        "change_pct": change,
                    }
                )

        return result

    def doctor_ranking(
        self, metric: str, bu: str = None, top_n: int = 10
    ) -> pd.DataFrame:
        """Rank doctors by performance"""
        df = self.data.df.copy()
        if bu:
            df = df[df["BU"] == bu]

        if "Revenue" in metric:
            ranking = df.groupby("Doctor Name")[metric].sum().reset_index()
        else:
            ranking = df.groupby("Doctor Name")[metric].mean().reset_index()

        ranking = ranking.sort_values(metric, ascending=False).head(top_n)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking
