import os
import time

ITERATIONS = 10000       # Run for 10000 cycles 

print("Overnight Training Loop zzzzz")

for i in range(ITERATIONS):
    print(f"\n=== CYCLE {i+1}/{ITERATIONS} ===")
    
    print(f"Playing games...")
    exit_code = os.system(f"python3 -m agents.Group41.self_play")
    
    if exit_code != 0:
        print("ERROR: Self play crashed stopping loop")
        break
        
    print("training training training...")
    exit_code = os.system(f"python3 -m agents.Group41.train")
    
    if exit_code != 0:
        print("ERROR: Training crashed stopping loop")
        break

    print("compiling new weights for c++...")
    exit_code = os.system(f"python3 trace.py")

    if exit_code != 0:
        print("ERROR: Tracing failed")
        break

print("Good morning h1 ap and tim :D")