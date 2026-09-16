import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from orca_feetech.hardware import HardwareBackend, JointCalibration, STSBus, load_device
from orca_feetech.model import JOINT_NAMES, ModelSpec


@pytest.fixture
def device(tmp_path):
    spec = ModelSpec.load()
    data = {"model_id": "orca-v1-right", "calibrated": True, "port": "FAKE", "joints": {}}
    for i, name in enumerate(JOINT_NAMES):
        data["joints"][name] = {
            "motor_id": i + 1, "verified": True, "model_number": 777,
            "model_angles_deg": np.rad2deg([spec.lower[i], spec.neutral[i], spec.upper[i]]).tolist(),
            "ticks": [1000, 2000, 3000] if i % 2 == 0 else [3000, 2000, 1000],
        }
    path = tmp_path / "hand.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_calibration_roundtrip_and_reversed_directions(device):
    _, cals, lower, upper = load_device(device)
    for cal, lo, hi in zip(cals, lower, upper):
        for angle in np.linspace(lo, hi, 31):
            assert abs(cal.to_angle(cal.to_ticks(angle)) - angle) < np.deg2rad(0.1)


@pytest.mark.parametrize("ticks", [[1000, 900, 1200], [4000, 20, 60], [-1, 1000, 2000]])
def test_invalid_or_wrapping_calibration_rejected(device, ticks):
    data = yaml.safe_load(device.read_text())["joints"]["index_mcp"]
    data["ticks"] = ticks
    with pytest.raises(ValueError):
        JointCalibration.parse("index_mcp", data)


def test_missing_calibration_never_opens_bus(device):
    data = yaml.safe_load(device.read_text())
    data["joints"]["wrist"]["verified"] = False
    device.write_text(yaml.safe_dump(data))
    calls = []
    with pytest.raises(ValueError):
        HardwareBackend(device, bus_factory=lambda *a: calls.append(a))
    assert not calls


def test_duplicate_ids_rejected(device):
    data = yaml.safe_load(device.read_text())
    data["joints"]["wrist"]["motor_id"] = 1
    device.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Duplicate"):
        load_device(device)


def test_vendor_sdk_packet_has_zero_time_bytes_not_hls_torque():
    sdk = pytest.importorskip("scservo_sdk")
    packet = sdk.sms_sts(SimpleNamespace())
    assert packet.SyncWritePosEx(1, 2048, 60, 10)
    assert packet.groupSyncWrite.data_dict[1] == [10, 0, 8, 0, 0, 60, 0]


def test_failed_sync_write_clears_packet_and_raises():
    bus = object.__new__(STSBus)
    bus.lock = threading.RLock()
    bus.sdk = SimpleNamespace(COMM_SUCCESS=0)
    cleared = []
    group = SimpleNamespace(clearParam=lambda: cleared.append(True), txPacket=lambda: -1001)
    bus.packet = SimpleNamespace(groupSyncWrite=group, SyncWritePosEx=lambda *a: True)
    with pytest.raises(OSError):
        bus.write([1], [2048])
    assert len(cleared) == 2
    with pytest.raises(ValueError):
        bus.write([1], [2048], speed=0)


def test_enable_with_lost_acknowledgement_is_still_disabled_on_close():
    bus = object.__new__(STSBus)
    bus.lock = threading.RLock()
    bus.sdk = SimpleNamespace(COMM_SUCCESS=0)
    bus.armed = set()
    calls = []
    def write(mid, address, enabled):
        calls.append((mid, enabled))
        return (-1001, 0) if enabled else (0, 0)
    bus.packet = SimpleNamespace(write1ByteTxRx=write)
    bus.port = SimpleNamespace(closePort=lambda: calls.append(("port", "closed")))
    with pytest.raises(OSError, match="unacknowledged"):
        bus.torque([1], True)
    bus.close()
    assert calls == [(1, 1), (1, 0), ("port", "closed")]
    assert not bus.armed


class FakeBus:
    def __init__(self, *args):
        self.events, self.armed = [], set()
        self.ticks = {i: 2000 for i in range(1, 18)}
        self.fail_read = False

    def ping(self, mid):
        return 777

    def read_register(self, mid, address):
        return 0

    def read(self, ids):
        if self.fail_read:
            raise OSError("unplugged")
        return [{"ticks": self.ticks[i], "temperature_c": 25} for i in ids]

    def write(self, ids, ticks, *args):
        self.events.append(("write", copy.copy(ticks)))
        self.ticks.update(zip(ids, ticks))

    def torque(self, ids, enabled):
        self.events.append(("torque", enabled))
        self.armed = set(ids) if enabled else set()

    def close(self):
        self.torque(sorted(self.armed), False)
        self.events.append(("close", None))


def test_arming_holds_current_position_and_watchdog_unloads(device):
    robot = HardwareBackend(device, bus_factory=FakeBus)
    try:
        assert robot.bus.events[:2] == [("write", [2000] * 17), ("torque", True)]
        with robot._state_lock:
            robot.last_input = time.monotonic() - 3
        assert robot.stop_event.wait(0.5)
        assert not robot.bus.armed
        with pytest.raises(RuntimeError, match="timeout"):
            robot.write(ModelSpec.load().neutral, time.monotonic())
    finally:
        robot.close()


def test_stale_input_does_not_feed_watchdog(device):
    robot = HardwareBackend(device, bus_factory=FakeBus)
    try:
        previous = robot.last_input
        robot.write(ModelSpec.load().neutral, time.monotonic() - 1)
        assert robot.last_input == previous
    finally:
        robot.close()


def test_read_failure_disables_torque(device):
    robot = HardwareBackend(device, bus_factory=FakeBus)
    try:
        robot.bus.fail_read = True
        with pytest.raises(OSError, match="unplugged"):
            robot.read()
        assert not robot.bus.armed
    finally:
        robot.close()
