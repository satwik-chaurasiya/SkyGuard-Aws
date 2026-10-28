"""
Unit tests for aws_simulator.py
Validates station profile configurations, physical weather simulation,
and all fault injection modes.
"""
import unittest
from aws_simulator import AWSSimulator, STATION_PROFILES


class TestAWSSimulator(unittest.TestCase):

    def setUp(self):
        self.sim = AWSSimulator(seed=42)

    def test_station_profiles_loaded(self):
        """Verify standard Indian climate profiles exist with valid meteorological values."""
        self.assertIn("AWS-101 [New Delhi / Safdarjung]", STATION_PROFILES)
        self.assertIn("AWS-204 [Leh / High-Altitude]", STATION_PROFILES)
        self.assertIn("AWS-308 [Mumbai / Colaba Marine]", STATION_PROFILES)
        self.assertIn("AWS-415 [Jaisalmer / Thar Desert]", STATION_PROFILES)
        self.assertIn("AWS-512 [Kolkata / Alipore Marine]", STATION_PROFILES)
        self.assertIn("AWS-620 [Bengaluru / Hebbal Deccan]", STATION_PROFILES)
        self.assertIn("AWS-930 [Srinagar / Kashmir Valley]", STATION_PROFILES)
        self.assertIn("AWS-128 [Port Blair / Andaman Islands]", STATION_PROFILES)

        for name, profile in STATION_PROFILES.items():
            self.assertIn("t_mean", profile)
            self.assertIn("p_mean", profile)
            self.assertIn("h_mean", profile)
            self.assertGreater(profile["p_mean"], 600.0, f"Unrealistic baseline pressure for {name}")
            self.assertLess(profile["p_mean"], 1050.0, f"Unrealistic baseline pressure for {name}")
            self.assertGreaterEqual(profile["h_mean"], 10.0)
            self.assertLessEqual(profile["h_mean"], 100.0)

    def test_set_station_switching(self):
        """Verify station switching re-calibrates the mean baselines."""
        self.sim.set_station("AWS-204 [Leh / High-Altitude]")
        self.assertEqual(self.sim.station, "AWS-204 [Leh / High-Altitude]")
        self.assertAlmostEqual(self.sim.t_mean, 9.5)
        self.assertAlmostEqual(self.sim.p_mean, 712.0)

        # Invalid station falls back gracefully
        self.sim.set_station("INVALID_STATION_NAME")
        self.assertIn(self.sim.station, STATION_PROFILES)

    def test_generate_normal_readings(self):
        """Verify normal readings produce valid 3-tuples within climatological bounds."""
        for _ in range(50):
            reading = self.sim.generate_normal()
            self.assertIsInstance(reading, tuple)
            self.assertEqual(len(reading), 3)
            t, p, h = reading

            # Temperature within physical limits
            self.assertGreater(t, self.sim.t_mean - 15.0)
            self.assertLess(t, self.sim.t_mean + 15.0)

            # Barometric pressure within physical limits
            self.assertGreater(p, self.sim.p_mean - 30.0)
            self.assertLess(p, self.sim.p_mean + 30.0)

            # Relative Humidity strictly between 5% and 99%
            self.assertGreaterEqual(h, 5.0)
            self.assertLessEqual(h, 99.0)

    def test_inject_temperature_spike(self):
        """Verify temperature spike injects a severe anomalous jump (> 50°C for Delhi)."""
        normal = self.sim.generate_normal()
        spike = self.sim.inject_temperature_spike()
        self.assertIsInstance(spike, tuple)
        self.assertGreater(spike[0], self.sim.t_mean + 20.0)
        self.assertEqual(spike[1], normal[1])
        self.assertEqual(spike[2], normal[2])

    def test_inject_frozen_sensor(self):
        """Verify frozen sensor returns completely static readings across ticks."""
        self.sim.generate_normal()
        f1 = self.sim.inject_frozen_sensor()
        f2 = self.sim.inject_frozen_sensor()
        f3 = self.sim.inject_frozen_sensor()

        self.assertEqual(f1[0], f2[0])
        self.assertEqual(f2[0], f3[0])
        self.assertEqual(self.sim.frozen_count, 3)

    def test_inject_humidity_fault(self):
        """Verify humidity fault drops relative humidity to near zero."""
        self.sim.generate_normal()
        fault = self.sim.inject_humidity_fault()
        self.assertAlmostEqual(fault[2], 0.2)

    def test_inject_pressure_fault(self):
        """Verify pressure fault simulates severe barometric leak/drop."""
        self.sim.generate_normal()
        fault = self.sim.inject_pressure_fault()
        self.assertLess(fault[1], self.sim.p_mean - 100.0)

    def test_inject_communication_error(self):
        """Verify communication error returns None (dropped observation)."""
        comm_error = self.sim.inject_communication_error()
        self.assertIsNone(comm_error)

    def test_inject_drift(self):
        """Verify progressive sensor calibration drift increases offset over time."""
        d1 = self.sim.inject_drift()
        self.assertTrue(self.sim.drift_active)
        d2 = self.sim.generate_normal()
        # Normal reading with active drift should be shifted upward
        self.assertGreater(self.sim.drift_offset, 3.5)

    def test_inject_genuine_weather_event(self):
        """Verify genuine convective storm displays coherent physical covariance."""
        storm = self.sim.inject_genuine_weather_event()
        t, p, h = storm
        # Temperature drops, Pressure drops, Humidity surges
        self.assertLess(t, self.sim.t_mean)
        self.assertLess(p, self.sim.p_mean)
        self.assertGreater(h, 80.0)

    def test_reset_fault(self):
        """Verify resetting fault clears frozen counter and drift states."""
        self.sim.inject_frozen_sensor()
        self.sim.inject_drift()
        self.assertTrue(self.sim.drift_active)

        restored = self.sim.reset_fault()
        self.assertFalse(self.sim.drift_active)
        self.assertEqual(self.sim.drift_offset, 0.0)
        self.assertEqual(self.sim.frozen_count, 0)
        self.assertIsNone(self.sim.frozen_temp)
        self.assertIsInstance(restored, tuple)


if __name__ == "__main__":
    unittest.main()
