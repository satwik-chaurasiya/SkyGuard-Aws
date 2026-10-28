"""
aws_simulator.py — Advanced Automatic Weather Station (AWS) Simulator
Supports multi-station profiles, sensor fault injection, calibration drift,
optical fibre backhaul telemetry (OTDR/DAS), solar-battery power telemetry,
remaining useful life (RUL) estimation, and natural disaster simulation for SIH 2026.
"""
import numpy as np


STATION_PROFILES = {
    "AWS-101 [New Delhi / Safdarjung]": {
        "region": "Subtropical Plains (MoES Station 42182)",
        "t_mean": 28.0, "t_std": 3.8,
        "p_mean": 1008.0, "p_std": 5.5,
        "h_mean": 62.0, "h_std": 10.0,
        "elevation": "216m ASL",
        "fibre_km": 14.5
    },
    "AWS-204 [Leh / High-Altitude]": {
        "region": "High-Altitude Cold Desert (MoES Station 42027)",
        "t_mean": 9.5, "t_std": 3.2,
        "p_mean": 712.0, "p_std": 4.2,
        "h_mean": 32.0, "h_std": 8.0,
        "elevation": "3,524m ASL",
        "fibre_km": 42.0
    },
    "AWS-308 [Mumbai / Colaba Marine]": {
        "region": "Coastal Maritime Station (MoES Station 43057)",
        "t_mean": 31.5, "t_std": 2.2,
        "p_mean": 1012.5, "p_std": 3.5,
        "h_mean": 84.0, "h_std": 7.5,
        "elevation": "11m ASL",
        "fibre_km": 8.2
    },
    "AWS-415 [Jaisalmer / Thar Desert]": {
        "region": "Arid Desert Zone (MoES Station 42328)",
        "t_mean": 39.5, "t_std": 4.5,
        "p_mean": 997.0, "p_std": 5.0,
        "h_mean": 22.0, "h_std": 6.5,
        "elevation": "225m ASL",
        "fibre_km": 35.0
    },
    "AWS-512 [Kolkata / Alipore Marine]": {
        "region": "Gangetic Delta & Coastal (MoES Station 42809)",
        "t_mean": 30.2, "t_std": 2.8,
        "p_mean": 1009.5, "p_std": 4.8,
        "h_mean": 78.0, "h_std": 8.5,
        "elevation": "6m ASL",
        "fibre_km": 12.0
    },
    "AWS-620 [Bengaluru / Hebbal Deccan]": {
        "region": "Deccan Plateau Upland (MoES Station 43295)",
        "t_mean": 24.5, "t_std": 2.5,
        "p_mean": 915.0, "p_std": 4.0,
        "h_mean": 65.0, "h_std": 9.0,
        "elevation": "920m ASL",
        "fibre_km": 18.5
    },
    "AWS-714 [Chennai / Nungambakkam Coastal]": {
        "region": "Tropical East Coast (MoES Station 43279)",
        "t_mean": 32.0, "t_std": 2.0,
        "p_mean": 1011.0, "p_std": 3.8,
        "h_mean": 80.0, "h_std": 7.0,
        "elevation": "7m ASL",
        "fibre_km": 9.5
    },
    "AWS-822 [Guwahati / Brahmaputra Valley]": {
        "region": "North-East River Basin (MoES Station 42309)",
        "t_mean": 26.5, "t_std": 3.2,
        "p_mean": 1004.0, "p_std": 5.0,
        "h_mean": 82.0, "h_std": 8.0,
        "elevation": "55m ASL",
        "fibre_km": 22.0
    },
    "AWS-930 [Srinagar / Kashmir Valley]": {
        "region": "Himalayan Alpine Valley (MoES Station 42059)",
        "t_mean": 14.0, "t_std": 4.0,
        "p_mean": 855.0, "p_std": 4.5,
        "h_mean": 68.0, "h_std": 11.0,
        "elevation": "1,585m ASL",
        "fibre_km": 28.0
    },
    "AWS-104 [Bhopal / Central Plateau]": {
        "region": "Central Vindhya Plateau (MoES Station 42452)",
        "t_mean": 29.0, "t_std": 3.5,
        "p_mean": 968.0, "p_std": 4.8,
        "h_mean": 58.0, "h_std": 10.5,
        "elevation": "523m ASL",
        "fibre_km": 16.0
    },
    "AWS-116 [Shimla / Western Himalayas]": {
        "region": "Mountain Ridge Observatory (MoES Station 42071)",
        "t_mean": 16.5, "t_std": 3.8,
        "p_mean": 790.0, "p_std": 4.2,
        "h_mean": 60.0, "h_std": 9.5,
        "elevation": "2,276m ASL",
        "fibre_km": 31.0
    },
    "AWS-128 [Port Blair / Andaman Islands]": {
        "region": "Island Maritime Tropical (MoES Station 43333)",
        "t_mean": 29.8, "t_std": 1.8,
        "p_mean": 1010.0, "p_std": 3.2,
        "h_mean": 85.0, "h_std": 6.0,
        "elevation": "16m ASL",
        "fibre_km": 11.0
    }
}


class FibreSimulator:
    """Simulates Optical Fibre Backhaul Telemetry (OTDR & Distributed Acoustic Sensing - DAS)."""
    def __init__(self, seed=42, span_km=15.0):
        self.rng = np.random.default_rng(seed)
        self.span_km = span_km
        self.base_attenuation = 0.22
        self.attenuation = self.base_attenuation
        self.strain_micro = 12.5
        self.pmd = 0.08
        self.fault_status = "Nominal"
        self.warning_pre_detect = False

    def step(self, disaster_mode=None):
        if disaster_mode == "earthquake":
            self.strain_micro = float(self.rng.uniform(450.0, 920.0))
            self.attenuation = float(self.rng.uniform(0.85, 1.45))
            self.fault_status = "Seismic Strain / Micro-Bend Event"
            self.warning_pre_detect = True
        elif disaster_mode == "landslide" or disaster_mode == "cyclone":
            self.strain_micro = float(self.rng.uniform(300.0, 650.0))
            self.attenuation = float(self.rng.uniform(0.65, 1.10))
            self.fault_status = "Cable Span Mechanical Stress"
            self.warning_pre_detect = True
        elif disaster_mode == "fibre_degradation":
            self.attenuation = float(self.rng.uniform(0.48, 0.75))
            self.strain_micro = float(self.rng.uniform(35.0, 65.0))
            self.fault_status = "⚠️ PRE-FAULT: Attenuation Buildup (Micro-bend/Rodent)"
            self.warning_pre_detect = True
        elif disaster_mode == "fibre_cut":
            self.attenuation = 25.0
            self.strain_micro = 0.0
            self.fault_status = "🔴 CRITICAL: Optical Fibre Cut / Link Down"
            self.warning_pre_detect = True
        else:
            self.attenuation = float(round(self.rng.normal(self.base_attenuation, 0.01), 3))
            self.strain_micro = float(round(self.rng.normal(12.0, 1.5), 1))
            self.fault_status = "Nominal (OTDR Verified)"
            self.warning_pre_detect = False

        return {
            "attenuation_db_km": self.attenuation,
            "strain_micro": self.strain_micro,
            "pmd_ps": self.pmd,
            "status": self.fault_status,
            "warning": self.warning_pre_detect
        }


class PowerMicrogridSimulator:
    """Simulates Solar Panel Charging, Battery Voltage, and Enclosure Power Health."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.battery_voltage = 13.6  # Volts (Fully charged LiFePO4)
        self.solar_current = 2.4     # Amps
        self.enclosure_temp = 32.5   # °C
        self.power_status = "Nominal (Solar Charging Active)"

    def step(self, storm_mode=False):
        if storm_mode:
            self.solar_current = float(self.rng.uniform(0.1, 0.4))  # Low solar during cloudburst/cyclone
            self.battery_voltage = max(11.2, self.battery_voltage - 0.05)
            self.power_status = "⚠️ Low Solar Yield (Battery Discharging)"
        else:
            self.solar_current = float(round(self.rng.normal(2.5, 0.3), 2))
            self.battery_voltage = float(round(min(14.2, max(12.4, self.battery_voltage + 0.002)), 2))
            self.power_status = "Nominal (Solar Charging Active)"

        return {
            "battery_v": self.battery_voltage,
            "solar_a": self.solar_current,
            "enclosure_temp": self.enclosure_temp,
            "status": self.power_status
        }


class AWSSimulator:
    """Advanced AWS Simulator with multi-station, fibre OTDR/DAS, solar power, RUL, and natural disasters."""

    def __init__(self, seed=42, station="AWS-101 [New Delhi / Safdarjung]"):
        self.rng = np.random.default_rng(seed)
        self.station = station
        self.set_station(station)
        self.frozen_temp = None
        self.frozen_count = 0
        self.drift_offset = 0.0
        self.drift_active = False
        self.step_counter = 0
        self.fibre = FibreSimulator(seed=seed, span_km=15.0)
        self.power = PowerMicrogridSimulator(seed=seed)
        self.active_disaster = None

    def set_station(self, station_name):
        if station_name in STATION_PROFILES:
            self.station = station_name
        else:
            self.station = list(STATION_PROFILES.keys())[0]
        p = STATION_PROFILES[self.station]
        self.t_mean = p["t_mean"]
        self.p_mean = p["p_mean"]
        self.h_mean = p["h_mean"]
        self.t_std = p["t_std"]
        self.p_std = p["p_std"]
        self.h_std = p["h_std"]
        self.last_normal = (self.t_mean, self.p_mean, self.h_mean)
        self.fibre = FibreSimulator(span_km=float(p["fibre_km"]))

    def generate_normal(self):
        self.step_counter += 1
        self.active_disaster = None
        self.fibre.step(None)
        self.power.step(False)

        diurnal_t = 1.2 * np.sin(self.step_counter / 15.0)
        diurnal_p = -0.6 * np.sin(self.step_counter / 15.0)

        temp = float(np.clip(self.rng.normal(self.t_mean + diurnal_t, self.t_std * 0.4),
                             self.t_mean - 12.0, self.t_mean + 12.0))
        pressure_shift = (temp - self.t_mean) * -0.25 + diurnal_p
        pressure = float(np.clip(self.rng.normal(self.p_mean + pressure_shift, self.p_std * 0.35),
                                 self.p_mean - 25.0, self.p_mean + 25.0))
        hum_shift = (temp - self.t_mean) * -0.8
        humidity = float(np.clip(self.rng.normal(self.h_mean + hum_shift, self.h_std * 0.4),
                                 5.0, 99.0))

        if self.drift_active:
            self.drift_offset += 0.4
            temp += self.drift_offset

        reading = (round(temp, 1), round(pressure, 1), round(humidity, 1))
        self.last_normal = reading
        self.frozen_count = 0
        return reading

    def inject_temperature_spike(self):
        spike_val = self.t_mean + 27.5 + float(self.rng.uniform(-1.5, 2.0))
        self.fibre.step(None)
        self.power.step(False)
        return (round(spike_val, 1), self.last_normal[1], self.last_normal[2])

    def inject_frozen_sensor(self):
        if self.frozen_temp is None:
            self.frozen_temp = self.last_normal[0]
        self.frozen_count += 1
        self.fibre.step(None)
        self.power.step(False)
        return (self.frozen_temp, self.last_normal[1], self.last_normal[2])

    def inject_humidity_fault(self):
        self.fibre.step(None)
        self.power.step(False)
        return (self.last_normal[0], self.last_normal[1], 0.2)

    def inject_pressure_fault(self):
        bad_pressure = max(600.0, self.p_mean - 160.0)
        self.fibre.step(None)
        self.power.step(False)
        return (self.last_normal[0], round(bad_pressure, 1), self.last_normal[2])

    def inject_communication_error(self):
        self.fibre.step("fibre_cut")
        self.power.step(True)
        return None

    def inject_drift(self):
        self.drift_active = True
        self.drift_offset = 3.5
        self.fibre.step("fibre_degradation")
        self.power.step(False)
        temp = self.last_normal[0] + self.drift_offset
        return (round(temp, 1), self.last_normal[1], self.last_normal[2])

    def inject_genuine_weather_event(self):
        self.active_disaster = "cloudburst"
        self.fibre.step(None)
        self.power.step(True)
        storm_temp = round(self.t_mean - 7.5, 1)
        storm_pressure = round(self.p_mean - 12.0, 1)
        return (storm_temp, storm_pressure, 98.5)

    def inject_earthquake_event(self):
        self.active_disaster = "earthquake"
        self.fibre.step("earthquake")
        self.power.step(True)
        return (self.last_normal[0], round(self.p_mean - 18.5, 1), self.last_normal[2])

    def inject_cyclone_event(self):
        self.active_disaster = "cyclone"
        self.fibre.step("landslide")
        self.power.step(True)
        return (round(self.t_mean - 5.0, 1), round(self.p_mean - 35.0, 1), 99.0)

    def reset_fault(self):
        self.frozen_temp = None
        self.frozen_count = 0
        self.drift_active = False
        self.drift_offset = 0.0
        self.active_disaster = None
        self.fibre.step(None)
        self.power.step(False)
        return self.generate_normal()

    def get_rul_estimation(self):
        """Calculates Remaining Useful Life (RUL) in days based on calibration drift & fibre attenuation."""
        if self.drift_active:
            return {"sensor_rul_days": 12, "fibre_rul_days": 45, "status": "Action Required (Recalibrate Sensor)"}
        elif self.fibre.warning_pre_detect:
            return {"sensor_rul_days": 180, "fibre_rul_days": 24, "status": "Action Required (Fibre Splicing Needed)"}
        else:
            return {"sensor_rul_days": 320, "fibre_rul_days": 540, "status": "Optimal (Healthy)"}
