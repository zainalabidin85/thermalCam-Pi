#!/usr/bin/env python3
import time
import statistics
from collections import deque
from threading import Lock, Thread, Event
import os
import smbus2

class RobustMLXReader:
    """
    MLX90614 reader with robust error handling and auto-recovery
    """
    
    def __init__(self, window_size=5, poll_interval=0.5, i2c_bus=1, max_retries=3):
        self.window_size = max(1, int(window_size))
        self.poll_interval = max(0.1, float(poll_interval))
        self.i2c_bus = int(i2c_bus)
        self.max_retries = max_retries
        
        # MLX90614 constants
        self.addr = 0x5A
        self.reg_ta = 0x06
        self.reg_tobj1 = 0x07
        
        # Data storage
        self.ambient_history = deque(maxlen=self.window_size)
        self.object_history = deque(maxlen=self.window_size)
        
        # Hardware state
        self.bus = None
        self.ok = False
        
        # Threading
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = None
        
        # Statistics
        self.read_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        
        # Initialize
        self._init_bus()
    
    def _init_bus(self):
        """Initialize or reinitialize I2C bus"""
        try:
            if self.bus:
                try:
                    self.bus.close()
                except:
                    pass
            self.bus = smbus2.SMBus(self.i2c_bus)
            print(f"[MLX] I2C bus {self.i2c_bus} initialized")
            return True
        except Exception as e:
            print(f"[MLX] Failed to initialize I2C bus: {e}")
            return False
    
    def _read_temperatures(self):
        """Read temperatures with correct byte order handling"""
        for attempt in range(self.max_retries):
            try:
                # Read ambient temperature (2 bytes)
                raw_ambient = self.bus.read_word_data(self.addr, self.reg_ta)
                
                # MLX90614 returns data in little-endian format
                # We need to swap bytes to get the correct value
                raw_ambient = ((raw_ambient & 0xFF) << 8) | ((raw_ambient >> 8) & 0xFF)
                
                # Convert to Celsius: each LSB is 0.02°C, and 0K = -273.15°C
                ambient = raw_ambient * 0.02 - 273.15
                
                # Read object temperature
                raw_object = self.bus.read_word_data(self.addr, self.reg_tobj1)
                
                # Swap bytes for object temperature too
                raw_object = ((raw_object & 0xFF) << 8) | ((raw_object >> 8) & 0xFF)
                
                # Object temperature might have an extra bit for indicating if it's above 0°C
                # But for most cases, we just convert directly
                object_temp = raw_object * 0.02 - 273.15
                
                # Validate readings (room temperature should be around 20-30°C)
                if 10 <= ambient <= 40 and 10 <= object_temp <= 40:
                    print(f"[MLX] Good reading - Ambient: {ambient:.2f}°C, Object: {object_temp:.2f}°C")
                    return ambient, object_temp, True
                else:
                    # Try alternative conversion (some MLX90614 modules need different handling)
                    # Maybe the bytes are already in correct order?
                    raw_ambient_alt = self.bus.read_word_data(self.addr, self.reg_ta)
                    ambient_alt = raw_ambient_alt * 0.02 - 273.15
                    
                    raw_object_alt = self.bus.read_word_data(self.addr, self.reg_tobj1)
                    object_temp_alt = raw_object_alt * 0.02 - 273.15
                    
                    if 10 <= ambient_alt <= 40 and 10 <= object_temp_alt <= 40:
                        print(f"[MLX] Good reading (alternative conversion) - Ambient: {ambient_alt:.2f}°C, Object: {object_temp_alt:.2f}°C")
                        return ambient_alt, object_temp_alt, True
                    else:
                        print(f"[MLX] Warning: Out of range values - Ambient: {ambient:.1f}°C, Object: {object_temp:.1f}°C (raw: {raw_ambient:04x}, {raw_object:04x})")
                        
            except OSError as e:
                if attempt < self.max_retries - 1:
                    print(f"[MLX] I2C error (attempt {attempt + 1}): {e}")
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise
            except Exception as e:
                print(f"[MLX] Unexpected error: {e}")
                raise
        
        return None, None, False
    
    def read_once(self):
        """Perform a single temperature reading"""
        try:
            ambient, object_temp, success = self._read_temperatures()
            
            with self.lock:
                if success:
                    self.ambient_history.append(ambient)
                    self.object_history.append(object_temp)
                    self.ok = True
                    self.read_count += 1
                    self.consecutive_errors = 0
                else:
                    self.ok = False
                    self.error_count += 1
                    self.consecutive_errors += 1
                    
                    # If too many consecutive errors, try to reinit
                    if self.consecutive_errors >= 5:
                        print("[MLX] Too many consecutive errors, reinitializing bus...")
                        self._init_bus()
                        self.consecutive_errors = 0
                    
            return ambient, object_temp, success
            
        except Exception as e:
            with self.lock:
                self.error_count += 1
                self.consecutive_errors += 1
                self.ok = False
            raise e
    
    def get_latest(self):
        """Get the latest filtered temperature readings"""
        with self.lock:
            if len(self.object_history) == 0:
                return None, None, False
            
            # Simple moving average (no outlier removal for simplicity)
            ambient_avg = statistics.mean(list(self.ambient_history))
            object_avg = statistics.mean(list(self.object_history))
            
            return ambient_avg, object_avg, self.ok
    
    def _polling_loop(self):
        """Background polling loop"""
        # Warm-up readings
        for _ in range(min(3, self.window_size)):
            try:
                self.read_once()
            except:
                pass
            time.sleep(self.poll_interval)
        
        while not self.stop_event.is_set():
            try:
                self.read_once()
            except Exception as e:
                print(f"[MLX] Polling error: {e}")
            
            # Dynamic polling based on error rate
            sleep_time = self.poll_interval
            with self.lock:
                if self.consecutive_errors > 3:
                    sleep_time = self.poll_interval * 2  # Slow down on errors
            
            time.sleep(sleep_time)
    
    def start(self):
        """Start background polling"""
        if self.thread and self.thread.is_alive():
            return
        
        self.stop_event.clear()
        self.thread = Thread(target=self._polling_loop, daemon=True)
        self.thread.start()
        print(f"[MLX] Started polling every {self.poll_interval}s")
    
    def stop(self):
        """Stop polling and cleanup"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.bus:
            self.bus.close()
        print("[MLX] Reader stopped")

# Factory function
def create_robust_mlx_reader(**kwargs):
    reader = RobustMLXReader(**kwargs)
    reader.start()
    return reader


if __name__ == "__main__":
    # Test the robust reader
    print("Testing Robust MLX90614 Reader...")
    
    reader = create_robust_mlx_reader(
        window_size=3,
        poll_interval=0.5,
        i2c_bus=1,
        max_retries=3
    )
    
    try:
        time.sleep(2)  # Warm-up
        
        for i in range(10):
            ambient, obj, ok = reader.get_latest()
            print(f"\nReading {i+1}:")
            print(f"  Ambient: {ambient:.2f}°C" if ambient else "  Ambient: No data")
            print(f"  Object: {obj:.2f}°C" if obj else "  Object: No data")
            print(f"  Status: {'OK' if ok else 'ERROR'}")
            print(f"  Errors: {reader.error_count}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        reader.stop()
