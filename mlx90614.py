#!/usr/bin/env python3
import time
from smbus2 import SMBus

MLX90614_ADDR  = 0x5A
MLX90614_TA    = 0x06  # ambient
MLX90614_TOBJ1 = 0x07  # object 1

def read_temp_c(bus, reg):
    raw = bus.read_word_data(MLX90614_ADDR, reg)   # already little-endian
    # NO byte swap here
    return raw * 0.02 - 273.15

def main():
    with SMBus(1) as bus:
        while True:
            try:
                ta   = read_temp_c(bus, MLX90614_TA)
                tobj = read_temp_c(bus, MLX90614_TOBJ1)
                print(f"Ambient: {ta:.2f} °C, Object: {tobj:.2f} °C")
                time.sleep(1)
            except OSError as e:
                print("I2C communication error:", e)
                break

if __name__ == "__main__":
    main()
