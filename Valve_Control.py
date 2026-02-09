import time
import numpy as np
import serial

from picosdk.ps2000a import ps2000a as ps
from picosdk.functions import adc2mV, assert_pico_ok
from ctypes import byref, c_int16, c_int32, c_uint32

# -----------------------------
# 用户参数（按你的情况预设：低误触发优先）
# -----------------------------
COM_PORT = "COM3"
BAUD = 115200
ARDUINO_CMD = b"C"

FS_APPROX = 24400.0  # 你当前大约 24.4 kS/s（用于换算点数；实际采样率以timebase返回为准）

WINDOW_MS = 200.0     # 每次采集窗口
SMOOTH_MS = 15.0      # 平滑窗口
MIN_HOLD_MS = 50.0    # 超阈值最小持续时间（强力抑制误触发）
COOLDOWN_MS = 400.0   # 触发后冷却时间

TH_ON_MV = 1.8        # 进入阈值（mV）
TH_OFF_MV = 0.9       # 退出阈值（mV，滞回）

# 量程：±1V（如果你的信号可能超过1V，改成2V）
INPUT_RANGE = ps.PS2000A_RANGE["PS2000A_1V"]


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """简单移动平均，n>=1。"""
    if n <= 1:
        return x
    # 使用卷积实现，边界用same会引入边缘效应，但对阈值检测影响通常可接受
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def detect_event_block(v_mv: np.ndarray, fs_hz: float,
                       th_on_mv: float = TH_ON_MV,
                       th_off_mv: float = TH_OFF_MV,
                       smooth_ms: float = SMOOTH_MS,
                       min_hold_ms: float = MIN_HOLD_MS) -> bool:
    """
    对单个 block 的电压序列（mV）做低误触发检测：
    1) 15ms 平滑
    2) 中位数作为基线
    3) |residual| > TH_ON 连续 >= 50ms 才触发
    """
    n_smooth = max(1, int(round((smooth_ms / 1000.0) * fs_hz)))
    n_hold = max(1, int(round((min_hold_ms / 1000.0) * fs_hz)))

    v_s = moving_average(v_mv, n_smooth)
    baseline = float(np.median(v_s))
    r = v_s - baseline

    above = np.abs(r) > th_on_mv

    # 计算最长连续 True 段长度（run length）
    if not np.any(above):
        return False

    # run length 计算（向量化）
    # 找到 True 段的起止
    idx = np.flatnonzero(above)
    # 连续段：相邻差为1
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.insert(idx[breaks + 1], 0, idx[0])
    ends = np.append(idx[breaks], idx[-1])

    max_run = int(np.max(ends - starts + 1))
    return max_run >= n_hold


class Pico2204A:
    def __init__(self):
        self.handle = c_int16(0)
        self.status = {}

    def open(self):
        self.status["openUnit"] = ps.ps2000aOpenUnit(byref(self.handle), None)
        assert_pico_ok(self.status["openUnit"])

    def close(self):
        if self.handle.value != 0:
            ps.ps2000aCloseUnit(self.handle)
            self.handle = c_int16(0)

    def set_channel_a(self, enabled=True, dc=True, vrange=INPUT_RANGE):
        coupling = ps.PS2000A_COUPLING["PS2000A_DC"] if dc else ps.PS2000A_COUPLING["PS2000A_AC"]
        self.status["setChA"] = ps.ps2000aSetChannel(
            self.handle,
            ps.PS2000A_CHANNEL["PS2000A_CHANNEL_A"],
            1 if enabled else 0,
            coupling,
            vrange,
            0.0  # analogueOffset
        )
        assert_pico_ok(self.status["setChA"])

    def _choose_timebase_for_fs(self, target_fs_hz: float, max_samples: int):
        """
        通过 ps2000aGetTimebase2 迭代寻找一个接近 target_fs 的 timebase。
        返回：(timebase, actual_interval_ns)
        """
        best_tb = None
        best_err = float("inf")
        best_interval_ns = None

        time_interval_ns = c_float = None  # placeholder to emphasize type; actual defined below

        for tb in range(1, 5000):
            interval_ns = c_float32()
            max_samp = c_int32()
            status = ps.ps2000aGetTimebase2(
                self.handle,
                tb,
                max_samples,
                byref(interval_ns),
                0,
                byref(max_samp),
                0
            )
            if status != 0:  # picoOK
                interval = float(interval_ns.value)
                fs = 1e9 / interval if interval > 0 else 0.0
                err = abs(fs - target_fs_hz)
                if err < best_err:
                    best_err = err
                    best_tb = tb
                    best_interval_ns = interval
                    # 早停：如果已经足够接近（<2%），可以退出
                    if err / target_fs_hz < 0.02:
                        break

        if best_tb is None:
            raise RuntimeError("Failed to find a valid timebase for target sampling rate.")
        return best_tb, best_interval_ns

    def capture_block_mv(self, window_ms: float, target_fs_hz: float = FS_APPROX):
        """
        采集一个 block，返回 (v_mv, fs_actual_hz)
        """
        # 预计样点数（取整）
        n_samples = int(round((window_ms / 1000.0) * target_fs_hz))
        n_samples = max(1000, n_samples)  # 下限保护

        # 选择 timebase，使实际采样率接近 target_fs
        # 注：ps2000aGetTimebase2 需要 c_float32 / c_int32
        from ctypes import c_float as c_float32  # ctypes里 c_float 是32-bit float
        tb_best = None
        best_err = float("inf")
        best_interval_ns = None

        for tb in range(1, 5000):
            interval_ns = c_float32()
            max_samp = c_int32()
            st = ps.ps2000aGetTimebase2(
                self.handle,
                tb,
                n_samples,
                byref(interval_ns),
                0,
                byref(max_samp),
                0
            )
            if st == 0:  # picoOK
                interval = float(interval_ns.value)
                fs = 1e9 / interval if interval > 0 else 0.0
                err = abs(fs - target_fs_hz)
                if err < best_err:
                    best_err = err
                    tb_best = tb
                    best_interval_ns = interval
                    if err / target_fs_hz < 0.02:
                        break

        if tb_best is None:
            raise RuntimeError("Could not determine a valid timebase.")

        fs_actual = 1e9 / best_interval_ns

        pre = 0
        post = n_samples

        # 准备缓冲区（ADC counts）
        buffer_a = (c_int16 * n_samples)()
        self.status["setDataBufferA"] = ps.ps2000aSetDataBuffer(
            self.handle,
            ps.PS2000A_CHANNEL["PS2000A_CHANNEL_A"],
            byref(buffer_a),
            n_samples,
            0,
            ps.PS2000A_RATIO_MODE["PS2000A_RATIO_MODE_NONE"]
        )
        assert_pico_ok(self.status["setDataBufferA"])

        # 运行 block
        time_indisposed_ms = c_int32()
        self.status["runBlock"] = ps.ps2000aRunBlock(
            self.handle,
            pre,
            post,
            tb_best,
            byref(time_indisposed_ms),
            0,
            None,
            None
        )
        assert_pico_ok(self.status["runBlock"])

        # 等待采集完成
        ready = c_int16(0)
        while ready.value == 0:
            ps.ps2000aIsReady(self.handle, byref(ready))
            time.sleep(0.001)

        # 读取数据
        n = c_int32(n_samples)
        overflow = c_int16()
        self.status["getValues"] = ps.ps2000aGetValues(
            self.handle,
            0,
            byref(n),
            1,
            ps.PS2000A_RATIO_MODE["PS2000A_RATIO_MODE_NONE"],
            0,
            byref(overflow)
        )
        assert_pico_ok(self.status["getValues"])

        # 获取最大 ADC 值用于转换到 mV
        max_adc = c_int16()
        self.status["maxValue"] = ps.ps2000aMaximumValue(self.handle, byref(max_adc))
        assert_pico_ok(self.status["maxValue"])

        # 转成 mV
        v_mv = np.array(adc2mV(buffer_a, INPUT_RANGE, max_adc), dtype=np.float32)

        # 停止
        ps.ps2000aStop(self.handle)

        return v_mv, fs_actual


def main():
    scope = Pico2204A()
    scope.open()
    scope.set_channel_a(enabled=True, dc=True, vrange=INPUT_RANGE)

    ser = serial.Serial(COM_PORT, BAUD, timeout=0.1)

    last_fire_t = 0.0

    try:
        print("Starting block polling... Ctrl+C to stop.")
        while True:
            v_mv, fs_actual = scope.capture_block_mv(WINDOW_MS, target_fs_hz=FS_APPROX)

            now = time.time()
            cooldown_ok = (now - last_fire_t) * 1000.0 >= COOLDOWN_MS

            if cooldown_ok:
                hit = detect_event_block(v_mv, fs_actual)
                if hit:
                    ser.write(ARDUINO_CMD)
                    ser.flush()
                    last_fire_t = now
                    print(f"[TRIGGER] sent {ARDUINO_CMD!r}  fs={fs_actual/1000:.2f} kS/s")

            # 轮询间隔：为了低误触发，宁可稍慢，避免CPU压力引入抖动
            # 这里给一个小空档
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        scope.close()


if __name__ == "__main__":
    main()
