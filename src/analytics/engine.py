"""Statistical analysis and root cause detection."""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


class AnalyticsEngine:
    """Statistical analysis engine driven by knowledge-base metadata."""

    def __init__(self, data_loader):
        self.data = data_loader

    def detect_anomalies(self, metric: str, threshold: float = 2.0) -> pd.DataFrame:
        """Detect statistical anomalies using Z-score."""
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
        self,
        kpi_name: str,
        doctor: str = None,
        bu: str = None,
    ) -> Dict:
        """Perform root cause analysis using the KPI relationship map."""
        if kpi_name not in self.data.df.columns:
            return {"error": f"KPI {kpi_name} not found"}

        df = self.data.df.copy()
        if doctor:
            df = df[df["Doctor Name"] == doctor]
        if bu:
            df = df[df["BU"] == bu]
        if df.empty:
            return {"error": "No data available"}

        current, previous = self._current_previous_periods(df)
        current_value = self._aggregate_metric(current, kpi_name)
        previous_value = self._aggregate_metric(previous, kpi_name)
        variance_pct = (
            (current_value - previous_value) / previous_value * 100
            if previous_value not in (0, None)
            else 0
        )

        drivers = self._get_drivers(kpi_name, current, previous)
        playbook = self.data.get_playbook(kpi_name)
        metadata = self.data.get_kpi_metadata(kpi_name)

        return {
            "kpi": kpi_name,
            "current_value": current_value,
            "previous_value": previous_value,
            "variance_pct": variance_pct,
            "trend": "declining"
            if variance_pct < -5
            else "improving"
            if variance_pct > 5
            else "stable",
            "severity": self._severity_from_variance(variance_pct),
            "primary_drivers": drivers[:5],
            "recommended_investigations": self._column_values(
                playbook, "Recommended_Investigation"
            ),
            "recommended_actions": self._column_values(playbook, "Recommended_Action"),
            "metadata": metadata,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

    def _current_previous_periods(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        latest_date = df["Date"].max()
        date_values = sorted(df["Date"].dropna().unique())

        current = df[df["Date"] == latest_date]
        if len(date_values) >= 2:
            previous = df[df["Date"] == date_values[-2]]
        else:
            previous = df[df["Date"] == latest_date]

        return current, previous

    def _aggregate_metric(self, df: pd.DataFrame, metric: str) -> float:
        if df.empty or metric not in df.columns:
            return 0

        metric_lower = metric.lower()
        if "%" in metric or "cr%" in metric_lower or "achievement" in metric_lower:
            return float(df[metric].mean())

        if any(
            word in metric_lower
            for word in [
                "revenue",
                "losses",
                "cases",
                "booking",
                "services",
                "opportunity",
                "clinics",
            ]
        ):
            return float(df[metric].sum())

        return float(df[metric].mean())

    def _get_drivers(
        self,
        kpi_name: str,
        current: pd.DataFrame,
        previous: pd.DataFrame,
    ) -> List[Dict]:
        """Get driver analysis from the knowledge-base relationship map."""
        relationships = self.data.get_kpi_relationships(kpi_name)
        if relationships.empty:
            return []

        if "Investigation_Order" in relationships.columns:
            relationships = relationships.sort_values("Investigation_Order")

        result = []
        for _, relationship in relationships.iterrows():
            driver_name = str(relationship["Child_KPI"])
            driver_column = self.data.resolve_kpi(driver_name)

            item = {
                "driver": driver_name,
                "relationship": relationship.get("Relationship_Type", ""),
                "weight": relationship.get("Weight", ""),
                "available_in_dataset": driver_column in self.data.df.columns
                if driver_column
                else False,
            }

            if driver_column in self.data.df.columns:
                current_value = self._aggregate_metric(current, driver_column)
                previous_value = self._aggregate_metric(previous, driver_column)
                change_pct = (
                    (current_value - previous_value) / previous_value * 100
                    if previous_value not in (0, None)
                    else 0
                )
                item.update(
                    {
                        "dataset_column": driver_column,
                        "current": current_value,
                        "previous": previous_value,
                        "change_pct": change_pct,
                    }
                )
            else:
                item.update(
                    {
                        "dataset_column": None,
                        "current": None,
                        "previous": None,
                        "change_pct": None,
                    }
                )

            result.append(item)

        return result

    def doctor_ranking(
        self,
        metric: str,
        bu: str = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Rank doctors by performance."""
        df = self.data.df.copy()
        if bu:
            df = df[df["BU"] == bu]
        if metric not in df.columns or df.empty:
            return pd.DataFrame()

        if any(
            word in metric.lower()
            for word in [
                "revenue",
                "losses",
                "cases",
                "booking",
                "services",
                "opportunity",
                "clinics",
            ]
        ):
            ranking = df.groupby("Doctor Name")[metric].sum().reset_index()
        else:
            ranking = df.groupby("Doctor Name")[metric].mean().reset_index()

        ranking = ranking.sort_values(metric, ascending=False).head(top_n)
        ranking["rank"] = range(1, len(ranking) + 1)
        return ranking

    @staticmethod
    def _severity_from_variance(variance_pct: float) -> str:
        if abs(variance_pct) > 20:
            return "critical"
        if abs(variance_pct) > 10:
            return "high"
        if abs(variance_pct) > 5:
            return "medium"
        return "low"

    @staticmethod
    def _column_values(df: pd.DataFrame, column: str) -> list[str]:
        if df.empty or column not in df.columns:
            return []
        return [str(value) for value in df[column].dropna().tolist()]
