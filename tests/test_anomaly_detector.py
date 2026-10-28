"""
Unit tests for anomaly_detector.py
Validates Isolation Forest scoring, Explainable AI (SHAP-style attribution),
WMO-No. 8 flagging, physical consistency checks, and self-healing imputation.
"""
import unittest
from anomaly_detector import AnomalyDetector, generate_historical_data


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.detector = AnomalyDetector(seed=42)

    def test_historical_data_generation(self):
        """Verify synthetic historical training data has correct shape and covariance."""
        X = generate_historical_data(n=500, seed=42, t_mean=28.0, p_mean=1008.0, h_mean=62.0)
        self.assertEqual(X.shape, (500, 3))
        # Temperatures bounded
        self.assertTrue((X[:, 0] >= 16.0).all() and (X[:, 0] <= 40.0).all())
        # Humidities bounded
        self.assertTrue((X[:, 2] >= 10.0).all() and (X[:, 2] <= 98.0).all())

    def test_nominal_reading_classification(self):
        """Verify standard normal reading is classified as Nominal with WMO Flag 1."""
        reading = (28.2, 1007.8, 61.5)
        is_anom, conf, sc = self.detector.score(reading)
        self.assertFalse(is_anom)
        self.assertLess(conf, 50.0)

        info = self.detector.classify_anomaly(reading)
        self.assertEqual(info["type"], "Nominal")
        self.assertEqual(info["severity"], "NONE")
        self.assertIn("FLAG 1", info["wmo_status"])
        self.assertTrue(info["is_genuine_weather"])

    def test_temperature_spike_classification(self):
        """Verify thermal spike is classified as Critical with WMO Flag 2."""
        spike_reading = (56.0, 1008.0, 60.0)
        is_anom, conf, sc = self.detector.score(spike_reading)
        self.assertTrue(is_anom)
        self.assertGreaterEqual(conf, 80.0)

        info = self.detector.classify_anomaly(spike_reading)
        self.assertEqual(info["type"], "Thermal Sensor Spike")
        self.assertEqual(info["severity"], "CRITICAL")
        self.assertIn("FLAG 2", info["wmo_status"])
        self.assertFalse(info["is_genuine_weather"])

        # XAI should attribute majority (> 70%) to temperature
        xai = info["xai"]
        self.assertGreater(xai["temp"], 70.0)

    def test_frozen_sensor_classification(self):
        """Verify repeated invariant reading with frozen_counter >= 3 is flagged with WMO Flag 3."""
        reading = (28.0, 1008.0, 60.0)
        info = self.detector.classify_anomaly(reading, frozen_counter=4)
        self.assertEqual(info["type"], "Frozen Sensor (ADC Stuck)")
        self.assertEqual(info["severity"], "HIGH")
        self.assertIn("FLAG 3", info["wmo_status"])
        self.assertFalse(info["is_genuine_weather"])

    def test_communication_error_classification(self):
        """Verify dropped observation (None) is classified as Critical with WMO Flag 4."""
        info = self.detector.classify_anomaly(None)
        self.assertEqual(info["type"], "Communication Failure")
        self.assertEqual(info["severity"], "CRITICAL")
        self.assertIn("FLAG 4", info["wmo_status"])
        self.assertFalse(info["is_genuine_weather"])

    def test_humidity_fault_classification(self):
        """Verify capacitive breakdown (near-zero RH) is flagged."""
        reading = (28.0, 1008.0, 0.2)
        info = self.detector.classify_anomaly(reading)
        self.assertEqual(info["type"], "Hygrometer Degradation")
        self.assertEqual(info["severity"], "MEDIUM")
        self.assertGreater(info["xai"]["hum"], 70.0)

    def test_pressure_fault_classification(self):
        """Verify piezoresistive sensor failure is flagged."""
        reading = (28.0, 840.0, 60.0)
        info = self.detector.classify_anomaly(reading)
        self.assertEqual(info["type"], "Barometric Transducer Failure")
        self.assertEqual(info["severity"], "CRITICAL")
        self.assertGreater(info["xai"]["press"], 70.0)

    def test_genuine_weather_event_distinction(self):
        """Verify physical storm (multivariate coherence) is classified as severe weather, not sensor fault."""
        # Temp drops by >5°C, Pressure drops by >5 hPa, Humidity surges >85%
        storm_reading = (21.5, 999.0, 94.0)
        info = self.detector.classify_anomaly(storm_reading)
        self.assertEqual(info["type"], "Genuine Meteorological Event")
        self.assertEqual(info["severity"], "NOMINAL")
        self.assertTrue(info["is_genuine_weather"])
        self.assertIn("FLAG 1", info["wmo_status"])

    def test_xai_attribution_sums_to_100(self):
        """Verify SHAP-style attribution percentages sum approximately to 100%."""
        reading = (45.0, 1002.0, 20.0)
        xai = self.detector.compute_xai_attribution(reading)
        total = xai["temp"] + xai["press"] + xai["hum"]
        self.assertAlmostEqual(total, 100.0, delta=1.0)

    def test_self_healing_imputation(self):
        """Verify auto-imputation fills missing values using historical normals."""
        # When history is empty, falls back to station mean
        self.detector.recent_normals = []
        c1 = self.detector.suggest_correction((55.0, 1008.0, 60.0))
        self.assertEqual(c1, (28.0, 1008.0, 62.0))

        # Add recent observations
        self.detector.update_normal_history((27.5, 1009.0, 60.0))
        self.detector.update_normal_history((28.0, 1008.5, 62.0))
        self.detector.update_normal_history((28.5, 1008.0, 64.0))

        c2 = self.detector.suggest_correction((55.0, 1008.0, 60.0))
        self.assertAlmostEqual(c2[0], 28.0, delta=0.5)

    def test_station_retraining(self):
        """Verify retrain_station resets baseline and re-fits Isolation Forest."""
        self.detector.retrain_station(t_mean=9.5, p_mean=712.0, h_mean=32.0)
        self.assertAlmostEqual(self.detector.t_mean, 9.5)
        self.assertAlmostEqual(self.detector.p_mean, 712.0)

        # 9.5°C should now score as nominal for Leh
        is_anom, conf, _ = self.detector.score((9.5, 712.0, 32.0))
        self.assertFalse(is_anom)

    def test_partial_none_robustness(self):
        """Verify partial telemetry drops (None in tuple) do not crash score or classify."""
        partial_reading = (25.0, None, 60.0)
        is_anom, conf, sc = self.detector.score(partial_reading)
        self.assertTrue(is_anom)
        self.assertGreaterEqual(conf, 90.0)

        info = self.detector.classify_anomaly(partial_reading)
        self.assertEqual(info["type"], "Channel Hardware Dropout")
        self.assertEqual(info["severity"], "CRITICAL")
        self.assertIn("Pressure (Barometer)", info["root_cause"])
        self.assertEqual(info["xai"]["press"], 100.0)

    def test_model_cache_speed(self):
        """Verify model caching stores and retrieves fitted models instantly without refitting."""
        # Initial fit
        self.detector.retrain_station(28.0, 1008.0, 62.0)
        model1 = self.detector.model

        # Switch to Leh
        self.detector.retrain_station(9.5, 712.0, 32.0)
        model2 = self.detector.model
        self.assertIsNot(model1, model2)

        # Switch back to Delhi — should return cached model1 directly
        self.detector.retrain_station(28.0, 1008.0, 62.0)
        self.assertIs(self.detector.model, model1)


if __name__ == "__main__":
    unittest.main()

