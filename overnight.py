import os
import time

ITERATIONS = 100         # Run for 100 cycles (~12 hours at 10 games in self_play.py per iteration averaging ~45 secs per game)
GAMES_PER_ITER = 10      # Play 10 games

print("Overnight Training Loop zzzzz")

for i in range(ITERATIONS):
    print(f"\n=== CYCLE {i+1}/{ITERATIONS} ===")
    
    print(f"Playing {GAMES_PER_ITER} games...")
    exit_code = os.system(f"python3 -m agents.Group41.self_play")
    
    if exit_code != 0:
        print("ERROR: Self play crashed stopping loop")
        break
        
    print("training training training...")
    exit_code = os.system(f"python3 -m agents.Group41.train")
    
    if exit_code != 0:
        print("ERROR: Training crashed stopping loop.")
        break

print("Good morning h1 ap and tim :D")