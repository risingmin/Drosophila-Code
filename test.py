# from ctypes import byref, c_int16
# from picosdk.ps2000a import ps2000a as ps
# from picosdk.functions import assert_pico_ok

# h = c_int16()
# status = ps.ps2000aOpenUnit(byref(h), None)
# print("status =", status)
# assert_pico_ok(status)
# print("handle =", h.value)
# ps.ps2000aCloseUnit(h)


from ctypes import byref, c_int16
from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import assert_pico_ok

handle = c_int16()
status = ps.ps2000_open_unit(byref(handle))
print("ps2000 status =", status, "handle =", handle.value)
assert_pico_ok(status)

ps.ps2000_close_unit(handle)
print("ps2000 open OK")