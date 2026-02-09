# Arduino 延迟触发功能 - 调试指南

## 问题诊断

你的 Arduino 延迟触发功能无法正常工作，主要原因有：

### 1. **队列数据结构不匹配** ❌
**原始问题：**
```python
# 在 schedule_arduino_signal() 中放入：
self._arduino_send_q.put_nowait((due_time, payload))  # 2 个元素

# 但在 _arduino_sender_loop() 中尝试解包：
due_time, seq_num, payload = self._arduino_send_q.get(timeout=0.2)  # 需要 3 个元素
```
→ **结果：** 解包失败，消息被丢弃，没有错误提示

### 2. **缺少调试日志** ❌
- 无法看到消息是否进入队列
- 无法看到线程是否成功启动
- 无法追踪序列号和时间戳
- 消息丢失时无提示

### 3. **触发条件检查不完整** ❌
```python
if self.ser and self.ser.is_open:  # 可能在某个时刻失效
    self.schedule_arduino_signal(...)
```
- 没有检查 `SERIAL_AVAILABLE` 标志
- 异常情况下无诊断信息

### 4. **线程启动日志不足** ❌
- 无法确认线程是否真的启动
- 无法获得线程 ID 用于调试

---

## ✅ 已应用的修复

### 修复 1：修正数据结构（添加序列号追踪）

**schedule_arduino_signal():**
```python
def schedule_arduino_signal(self, payload: bytes, detect_elapsed_ms: float):
    seq_num = self.frame_count  # ← 添加序列号
    # ...
    self._arduino_send_q.put_nowait((due_time, seq_num, payload))  # ← 3个元素
    print(f"[ARDUINO] Scheduled (seq={seq_num}): detect={detect_elapsed_ms:.2f}ms, "
          f"target={self.arduino_target_delay_ms:.0f}ms, wait={wait_ms:.2f}ms")
```

**_arduino_sender_loop():**
```python
def _arduino_sender_loop(self):
    print(f"[ARDUINO] Sender thread started")
    sent_count = 0
    while not self._arduino_thread_stop.is_set() and not self.stop_event.is_set():
        try:
            due_time, seq_num, payload = self._arduino_send_q.get(timeout=0.2)  # ← 解包 3 个元素
        except queue.Empty:
            continue
        # ... 等待和发送逻辑 ...
        if self.ser and getattr(self.ser, "is_open", False):
            try:
                self.ser.write(payload)
                sent_count += 1
                print(f"[ARDUINO] SENT (seq={seq_num}, total={sent_count}) payload={payload}")
            except Exception as e:
                print(f"[ARDUINO] Send failed (seq={seq_num}): {e}")
```

### 修复 2：改进触发条件检查

```python
if trigger_allowed:
    detect_elapsed_ms = (time.time() - frame_start_time) * 1000.0
    # 检查所有必要条件
    if SERIAL_AVAILABLE and self.ser and getattr(self.ser, 'is_open', False):
        self.schedule_arduino_signal(CORRECT_EMBRYO_SIGNAL, detect_elapsed_ms)
        did_trigger = True
        print(f"--> Embryo detected. detect={detect_elapsed_ms:.2f}ms, "
              f"target={self.arduino_target_delay_ms}ms, scheduled.")
    else:
        # 诊断信息
        if not SERIAL_AVAILABLE:
            print(f"[DEBUG] Serial not available (SERIAL_AVAILABLE={SERIAL_AVAILABLE})")
        elif not self.ser:
            print(f"[DEBUG] Serial object is None")
        else:
            print(f"[DEBUG] Serial is_open={getattr(self.ser, 'is_open', False)}")
```

### 修复 3：改进线程启动日志

```python
if self.ser and self.ser.is_open:
    print(f"[ARDUINO] Starting sender thread (ser={self.ser}, is_open={self.ser.is_open})")
    self._arduino_thread_stop.clear()
    self._arduino_sender_thread = threading.Thread(
        target=self._arduino_sender_loop,
        daemon=False  # ← 改为 False，便于显式停止
    )
    self._arduino_sender_thread.start()
    print(f"[ARDUINO] Sender thread started (thread_id={self._arduino_sender_thread.ident})")
else:
    print(f"[ARDUINO] Cannot start sender thread: ser={self.ser}, "
          f"is_open={self.ser.is_open if self.ser else 'N/A'}")
```

---

## 📋 测试清单 - 验证修复

### 1. 启动程序并检查日志
```
[ARDUINO] Starting sender thread (ser=<amcam.Amcam object...>, is_open=True)
[ARDUINO] Sender thread started (thread_id=12345)
```
✅ 如果看到这些，说明线程启动成功

### 2. 检测到胚胎时的日志
```
[ARDUINO] Scheduled (seq=42): detect=15.32ms, target=1000ms, wait=984.68ms, due_time=1703000000.1234
```
✅ 消息成功进入队列

### 3. 预期的发送日志（延迟后）
```
[ARDUINO] SENT (seq=42, total=1) payload=b'C'
```
✅ 消息在正确的时间发送

### 4. 如果检测失败，诊断信息
```
[DEBUG] Serial is_open=False
```
或
```
[DEBUG] Serial not available (SERIAL_AVAILABLE=False)
```
✅ 帮助识别问题所在

---

## 🔧 故障排除

### 问题：看不到任何 [ARDUINO] 日志

**可能原因：**
1. 串口连接失败 → 检查 `SERIAL_PORT` 和 USB 连接
2. `SERIAL_AVAILABLE = False` → 检查 `pyserial` 是否安装：
   ```bash
   pip install pyserial
   ```
3. 没有检测到胚胎 → 检查 `trigger_allowed` 条件

**调试步骤：**
```python
# 在程序启动时添加测试代码
print(f"SERIAL_AVAILABLE: {SERIAL_AVAILABLE}")
print(f"SERIAL_PORT: {SERIAL_PORT}")
import serial
try:
    test_ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Serial port open: {test_ser.is_open}")
    test_ser.close()
except Exception as e:
    print(f"Serial port error: {e}")
```

### 问题：消息进入队列但未发送

**检查：**
- 线程是否仍在运行？搜索 `Sender thread stopped` 日志
- 序列号是否在增加？
- 延迟时间是否合理？`wait=984.68ms` 表示需要等待约 985ms

**手动测试：**
```python
# 在 _arduino_sender_loop() 中临时添加
print(f"[DEBUG] Waiting for {remaining:.3f}s (due_time={due_time:.4f}, now={time.time():.4f})")
```

### 问题：线程启动失败

**检查日志：**
```
[ARDUINO] Cannot start sender thread: ser=None, is_open=N/A
```

**原因：** Arduino 串口连接失败

**解决：**
1. 检查 USB 连接
2. 检查端口号 (`SERIAL_PORT = 'COM3'` 可能需要改为 `'COM4'` 等)
3. 检查驱动程序

---

## 📊 性能考量

- **队列大小：** 默认 50 条消息，足以处理多个胚胎
- **延迟精度：** ±5ms（由 `time.sleep(min(remaining, 0.005))` 决定）
- **线程模式：** `daemon=False` 确保程序关闭前所有信号都被发送

---

## 🎯 常见参数调整

### 调整目标延迟时间
```python
self.arduino_target_delay_ms = 500  # 改为 500ms（原为 1000ms）
```

### 调整触发冷却时间
```python
self.trigger_cooldown_ms = 200  # 改为 200ms（原为 500ms）
```

### 改变串口波特率（如果 Arduino 配置不同）
```python
BAUD_RATE = 115200  # 改为 115200（原为 9600）
```

---

## 📝 总结

主要修复：
1. ✅ 队列数据结构（2 → 3 个元素）
2. ✅ 序列号追踪（便于调试）
3. ✅ 详细日志（每个关键步骤都有输出）
4. ✅ 触发条件检查（防止无声失败）
5. ✅ 线程启动验证（确认线程 ID）

现在运行程序时，如果有问题，日志会清楚地告诉你在哪里失败了。
