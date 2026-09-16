"""STS3215 position control using FEETECH's STS SDK, without HLS torque semantics."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .control import HandState
from .model import JOINT_NAMES, MODEL_ID, ModelSpec, vector


@dataclass(frozen=True)
class JointCalibration:
    motor_id: int
    angles: np.ndarray
    ticks: np.ndarray
    model_number: int

    @classmethod
    def parse(cls, name, data):
        if data.get("verified") is not True:
            raise ValueError(f"{name}: mapping/calibration has not been verified")
        angles = np.deg2rad(np.asarray(data["model_angles_deg"], dtype=float))
        ticks = np.asarray(data["ticks"], dtype=float)
        if angles.shape != (3,) or ticks.shape != (3,) or not np.isfinite([angles, ticks]).all():
            raise ValueError(f"{name}: need three finite model angles and encoder readings")
        if not np.all(np.diff(angles) > 0):
            raise ValueError(f"{name}: angles must be ordered lower, neutral, upper")
        if not (np.all(np.diff(ticks) > 0) or np.all(np.diff(ticks) < 0)):
            raise ValueError(f"{name}: ticks must be strictly monotonic; check direction and wrapping")
        if np.any(ticks < 0) or np.any(ticks > 4095) or np.ptp(ticks) > 3500:
            raise ValueError(f"{name}: invalid single-turn span; reposition the spool and recalibrate")
        if np.min(np.abs(np.diff(ticks))) < 5:
            raise ValueError(f"{name}: calibration anchors are too close")
        motor_id = int(data["motor_id"])
        if not 1 <= motor_id <= 253:
            raise ValueError("Motor ID must be 1..253")
        return cls(motor_id, angles, ticks, int(data["model_number"]))

    def to_ticks(self, q):
        if not self.angles[0] - 1e-8 <= q <= self.angles[-1] + 1e-8:
            raise ValueError(f"Motor {self.motor_id}: target outside measured range")
        return int(round(np.interp(q, self.angles, self.ticks)))

    def to_angle(self, tick):
        lo, hi = sorted((self.ticks[0], self.ticks[-1]))
        if not lo - 3 <= tick <= hi + 3:
            raise RuntimeError(f"Motor {self.motor_id}: feedback outside calibrated encoder span")
        if self.ticks[0] < self.ticks[-1]:
            return float(np.interp(tick, self.ticks, self.angles))
        return float(np.interp(tick, self.ticks[::-1], self.angles[::-1]))


def load_device(path):
    data = yaml.safe_load(Path(path).read_text())
    if data.get("model_id") != MODEL_ID or data.get("calibrated") is not True:
        raise ValueError("Device needs completed calibration for orca-v1-right")
    if not data.get("port"):
        raise ValueError("Set the actual serial port in the device YAML")
    if set(data["joints"]) != set(JOINT_NAMES):
        raise ValueError("Device must contain exactly the 17 canonical joints")
    cals = [JointCalibration.parse(n, data["joints"][n]) for n in JOINT_NAMES]
    if len({c.motor_id for c in cals}) != 17:
        raise ValueError("Duplicate motor IDs in mapping")
    if not 1 <= data.get("speed", 60) <= 300 or not 1 <= data.get("acceleration", 10) <= 50:
        raise ValueError("Bring-up limits: speed 1..300 and acceleration 1..50; zero speed means maximum")
    if not 20 <= data.get("max_temperature_c", 55) <= 60:
        raise ValueError("Temperature threshold must be within 20..60 C")
    spec = ModelSpec.load()
    for i, cal in enumerate(cals):
        if abs(cal.angles[1] - spec.neutral[i]) > np.deg2rad(0.1):
            raise ValueError(f"{JOINT_NAMES[i]}: middle anchor must be the model neutral angle")
    lower = np.maximum(spec.lower, [c.angles[0] for c in cals])
    upper = np.minimum(spec.upper, [c.angles[-1] for c in cals])
    if np.any(lower >= upper):
        raise ValueError("Measured range does not overlap model range")
    return data, cals, lower, upper


class STSBus:
    def __init__(self, port, baudrate=1_000_000):
        import scservo_sdk as sdk
        self.sdk = sdk
        self.port = sdk.PortHandler(port)
        self.packet = sdk.sms_sts(self.port)
        self.lock = threading.RLock()
        self.armed = set()
        if not self.port.setBaudRate(baudrate):
            self.port.closePort()
            raise OSError(f"Cannot open {port} at {baudrate}")
        # Bound writes as well as reads; unplugging must not block indefinitely.
        self.port.ser.write_timeout = 0.1

    def _check(self, result, error, operation):
        if result != self.sdk.COMM_SUCCESS or error:
            raise OSError(f"{operation}: communication={result}, servo_error={error}")

    def ping(self, motor_id):
        with self.lock:
            model, result, error = self.packet.ping(motor_id)
            self._check(result, error, f"ping {motor_id}")
            return model

    def read_register(self, motor_id, address, length=1):
        with self.lock:
            method = self.packet.read1ByteTxRx if length == 1 else self.packet.read2ByteTxRx
            value, result, error = method(motor_id, address)
            self._check(result, error, f"read {motor_id}:{address}")
            return value

    def read(self, motor_ids):
        with self.lock:
            group = self.sdk.GroupSyncRead(self.packet, 56, 15)
            for mid in motor_ids:
                if not group.addParam(mid):
                    raise ValueError(f"Duplicate or invalid read ID {mid}")
            result = group.txRxPacket()
            self._check(result, 0, "STS sync read")
            rows = []
            for mid in motor_ids:
                available, error = group.isAvailable(mid, 56, 15)
                if not available or error:
                    raise OSError(f"Missing/faulty feedback for motor {mid}; cached values are rejected")
                position = self.packet.scs_tohost(group.getData(mid, 56, 2), 15)
                if not 0 <= position <= 4095:
                    raise OSError(f"Motor {mid} returned non-single-turn position {position}")
                rows.append({"id": mid, "ticks": position,
                             "speed_raw": self.packet.scs_tohost(group.getData(mid, 58, 2), 15),
                             "temperature_c": group.getData(mid, 63, 1),
                             "voltage_raw": group.getData(mid, 62, 1),
                             "current_raw": self.packet.scs_tohost(group.getData(mid, 69, 2), 15)})
            return rows

    def torque(self, motor_ids, enabled):
        failures = []
        with self.lock:
            for mid in motor_ids:
                # A lost acknowledgement does not prove the enable command failed.
                # Track every attempted enable so close() also unloads that servo.
                if enabled:
                    self.armed.add(mid)
                result, error = self.packet.write1ByteTxRx(mid, 40, int(enabled))
                if result != self.sdk.COMM_SUCCESS or error:
                    failures.append(mid)
                elif not enabled:
                    self.armed.discard(mid)
        if failures:
            raise OSError(f"Torque {'enable' if enabled else 'disable'} unacknowledged for {failures}")

    def write(self, motor_ids, ticks, speed=60, acceleration=10):
        if len(motor_ids) != len(ticks) or len(set(motor_ids)) != len(motor_ids):
            raise ValueError("Invalid write IDs/positions")
        if not 1 <= speed <= 300 or not 1 <= acceleration <= 50:
            raise ValueError("Invalid speed/acceleration; zero speed is unsafe for bring-up")
        with self.lock:
            group = self.packet.groupSyncWrite
            group.clearParam()
            try:
                for mid, tick in zip(motor_ids, ticks, strict=True):
                    if not np.isfinite(tick) or int(tick) != tick or not 0 <= tick <= 4095:
                        raise ValueError("STS target must be an integer in 0..4095")
                    # Vendor STS API has FOUR arguments; bytes 44–45 are zero, not torque.
                    if not self.packet.SyncWritePosEx(mid, int(tick), speed, acceleration):
                        raise OSError(f"Could not add STS command for {mid}")
                self._check(group.txPacket(), 0, "STS sync write")
            finally:
                group.clearParam()

    def close(self):
        try:
            if self.armed:
                self.torque(sorted(self.armed), False)
        finally:
            self.port.closePort()


class HardwareBackend:
    """Position backend with an independent input watchdog and calibrated coordinates."""
    def __init__(self, config_path, bus_factory=STSBus):
        self.config, self.calibration, self.lower, self.upper = load_device(config_path)
        self.motor_ids = [c.motor_id for c in self.calibration]
        self.bus = bus_factory(self.config["port"], self.config.get("baudrate", 1_000_000))
        self.last_input = time.monotonic()
        self.fault = None
        self.stop_event = threading.Event()
        self.thread = None
        self._state_lock = threading.RLock()
        try:
            for cal in self.calibration:
                if self.bus.ping(cal.motor_id) != cal.model_number:
                    raise RuntimeError(f"Motor {cal.motor_id} model differs from calibration")
                if self.bus.read_register(cal.motor_id, 33) != 0:
                    raise RuntimeError(f"Motor {cal.motor_id} is not in STS position mode (0)")
            initial = self.read()
            if np.any(initial.positions < self.lower) or np.any(initial.positions > self.upper):
                raise RuntimeError("Initial position is outside the intersection of measured and model limits")
            # Make stored goals equal current feedback before enabling torque.
            self.bus.write(self.motor_ids, [c.to_ticks(q) for c, q in zip(self.calibration, initial.positions)],
                           self.config.get("speed", 60), self.config.get("acceleration", 10))
            self.bus.torque(self.motor_ids, True)
            self.last_input = time.monotonic()
            self.thread = threading.Thread(target=self._watchdog, name="orca-input-watchdog", daemon=True)
            self.thread.start()
        except BaseException:
            self.bus.close()
            raise

    def _watchdog(self):
        while not self.stop_event.wait(0.05):
            with self._state_lock:
                if time.monotonic() - self.last_input > 2:
                    self.fault = "Input timeout: torque disabled; restart required"
                    try:
                        self.bus.torque(self.motor_ids, False)
                    except OSError as exc:
                        self.fault += f"; disable NOT acknowledged: {exc}"
                    self.stop_event.set()

    def read(self):
        if self.fault:
            raise RuntimeError(self.fault)
        try:
            rows = self.bus.read(self.motor_ids)
            if any(row["temperature_c"] >= self.config.get("max_temperature_c", 55) for row in rows):
                raise RuntimeError("Motor temperature threshold reached")
            q = np.array([c.to_angle(row["ticks"]) for c, row in zip(self.calibration, rows, strict=True)])
            return HandState(q, estimated=True, telemetry={"motors": rows})
        except BaseException as exc:
            self.fault = f"Feedback fault: {exc}"
            # A broken link cannot guarantee receipt; preserve the original read error.
            try:
                self.bus.torque(self.motor_ids, False)
            except OSError:
                pass
            raise

    def write(self, q, source_timestamp=None):
        q = vector(q)
        now = time.monotonic()
        with self._state_lock:
            if self.fault:
                raise RuntimeError(self.fault)
            if source_timestamp is None or not 0 <= now - source_timestamp <= 0.25:
                return  # Stale data never renews the independent watchdog.
            if source_timestamp < self.last_input:
                # The first target may have been created just before arming.
                if now - self.last_input > 0.25:
                    raise ValueError("Out-of-order hardware input")
            if np.any(q < self.lower) or np.any(q > self.upper):
                raise ValueError("Hardware target exceeds measured soft limits")
            try:
                self.bus.write(self.motor_ids, [c.to_ticks(x) for c, x in zip(self.calibration, q, strict=True)],
                               self.config.get("speed", 60), self.config.get("acceleration", 10))
            except OSError as exc:
                self.fault = f"Command fault: {exc}"
                try:
                    self.bus.torque(self.motor_ids, False)
                except OSError as disable_error:
                    self.fault += f"; disable NOT acknowledged: {disable_error}"
                raise
            self.last_input = source_timestamp

    def render(self):
        return None

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.bus.close()


def scan(port, baudrate=1_000_000, ids="1-17"):
    if "-" in ids:
        start, end = map(int, ids.split("-"))
        numbers = list(range(start, end + 1))
    else:
        numbers = [int(x) for x in ids.split(",")]
    if not numbers or any(not 1 <= mid <= 253 for mid in numbers):
        raise ValueError("IDs must be within 1..253")
    bus = STSBus(port, baudrate)
    result = []
    try:
        for mid in numbers:
            try:
                result.append({"id": mid, "model_number": bus.ping(mid), "found": True,
                               "mode": bus.read_register(mid, 33)})
            except OSError:
                result.append({"id": mid, "found": False})
    finally:
        bus.close()
    return result
