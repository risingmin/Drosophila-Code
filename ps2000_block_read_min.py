import time
import numpy as np
from picosdk.ps2000 import ps2000 as ps

RANGE_INDEX = 5   # 先随便用一个；先把采集跑通，量程后面再精确设
CHANNEL_A = 0     # ps2000 里通常 0=A

def call_first_that_works(func, candidates):
    """
    依次尝试不同的参数列表，直到某个调用不抛 TypeError。
    candidates: list[tuple(args)]
    """
    last_err = None
    for args in candidates:
        try:
            return func(*args)
        except TypeError as e:
            last_err = e
    raise last_err

def main():
    handle = ps.ps2000_open_unit()
    if int(handle) <= 0:
        raise RuntimeError(f"ps2000_open_unit failed, handle={handle}")
    print("Opened, handle =", handle)

    # set channel: (handle, channel, enabled, dc, range)
    ps.ps2000_set_channel(handle, CHANNEL_A, 1, 1, RANGE_INDEX)

    n_samples = 5000
    timebase = 8
    oversample = 0

    # ---- run block (try common signatures) ----
    # 不同 wrapper 可能是：
    #   run_block(handle, nSamples, timebase, oversample, timeIndisposedMsPtr)
    #   run_block(handle, nSamples, timebase, oversample, segmentIndex, lpReady)
    #   run_block(handle, nSamples, timebase, oversample, lpReady)
    print("Calling ps2000_run_block ...")
    call_first_that_works(
        ps.ps2000_run_block,
        candidates=[
            (handle, n_samples, timebase, oversample, 0),      # 5 args: last could be time_indisposed_ms or segment
            (handle, n_samples, timebase, oversample, None),   # 5 args
            (handle, n_samples, timebase, oversample, 0, ),    # same as first, explicit
        ],
    )

    # ---- wait ready (try common signatures) ----
    # ready may be:
    #   ps2000_ready(handle) -> 0/1
    #   ps2000_ready() -> 0/1   (rare)
    print("Waiting for ready ...")
    while True:
        try:
            r = ps.ps2000_ready(handle)
        except TypeError:
            r = ps.ps2000_ready()
        if int(r) != 0:
            break
        time.sleep(0.001)

    # ---- get values (try common signatures) ----
    # 不同 wrapper 可能要求传 buffer 指针（int）或 numpy array 直接传
    buff_a = np.empty(n_samples, dtype=np.int16)

    print("Calling ps2000_get_values ...")
    # 方案 1：传 numpy 数组的 ctypes 地址（最通用）
    ptr_a = buff_a.ctypes.data

    # 常见签名可能是：
    #   get_values(handle, bufferA_ptr, bufferB_ptr, bufferC_ptr, bufferD_ptr, nSamples)
    #   get_values(handle, bufferA_ptr, bufferB_ptr, bufferC_ptr, bufferD_ptr, overflow_ptr, nSamples)
    #   get_values(handle, bufferA_ptr, nSamples)  (更简化)
    call_first_that_works(
        ps.ps2000_get_values,
        candidates=[
            (handle, ptr_a, None, None, None, n_samples),
            (handle, ptr_a, 0, 0, 0, n_samples),
            (handle, ptr_a, None, None, None, None, n_samples),
            (handle, ptr_a, n_samples),
            (handle, buff_a, n_samples),  # 有的 wrapper 允许直接传 numpy
        ],
    )

    # stop/close
    try:
        ps.ps2000_stop(handle)
    except Exception:
        pass
    ps.ps2000_close_unit(handle)

    print("raw min/max =", int(buff_a.min()), int(buff_a.max()))
    print("Done.")

if __name__ == "__main__":
    main()
