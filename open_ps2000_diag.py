import sys
import picosdk

from picosdk.ps2000 import ps2000 as ps

print("python  =", sys.executable)
print("picosdk =", picosdk.__file__)

# 注意：你当前的 wrapper 里 ps2000_open_unit() 不接受任何参数
handle = ps.ps2000_open_unit()
print("ps2000_open_unit handle =", handle)

# 一般约定：handle <= 0 表示失败（未找到/未打开）
if handle and int(handle) > 0:
    ps.ps2000_close_unit(handle)
    print("ps2000 close OK")
else:
    print("ps2000 open FAILED (handle <= 0)")
