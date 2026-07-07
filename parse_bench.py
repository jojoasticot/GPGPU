#!/usr/bin/env python3
 
import re
import sys
 
 
def parse_benchmarks(filepath):
    avg_fps_values = []
    exec_times = []
    last_avg = None
 
    avg_re = re.compile(r"average:\s+([0-9]+\.[0-9]+)")
    time_re = re.compile(r"Execution ended after \d+:\d+:([0-9]+\.[0-9]+)")
 
    with open(filepath) as f:
        for line in f:
            m = avg_re.search(line)
            if m:
                last_avg = float(m.group(1))
 
            m = time_re.search(line)
            if m:
                exec_times.append(float(m.group(1)))
 
            if "Freeing pipeline" in line and last_avg is not None:
                avg_fps_values.append(last_avg)
                last_avg = None
 
    return avg_fps_values, exec_times
 
 
def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"
 
    avg_fps_values, exec_times = parse_benchmarks(filepath)
 
    if avg_fps_values:
        mean_fps = sum(avg_fps_values) / len(avg_fps_values)
        print(f"Number of benchmarks:       {len(avg_fps_values)}")
        print(f"Average FPS per benchmark:  {', '.join(f'{v:.2f}' for v in avg_fps_values)}")
        print(f"Average of average FPS:     {mean_fps:.2f}")
    else:
        print("No FPS data found.")
 
    if exec_times:
        mean_time = sum(exec_times) / len(exec_times)
        print(f"Execution times (s):        {', '.join(f'{t:.3f}' for t in exec_times)}")
        print(f"Average execution time (s): {mean_time:.3f}")
    else:
        print("No execution time data found.")
 
 
if __name__ == "__main__":
    main()

