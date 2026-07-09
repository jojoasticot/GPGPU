import re
import sys
from collections import defaultdict


def parse_benchmarks(filepath):
    results = defaultdict(lambda: {"fps": [], "times": []})

    current_file = None
    last_avg = None

    file_re = re.compile(r"FILE:\s+(.+)")
    avg_re = re.compile(r"average:\s+([0-9]+\.[0-9]+)")
    time_re = re.compile(r"Execution ended after \d+:\d+:([0-9]+\.[0-9]+)")

    with open(filepath) as f:
        for line in f:
            m = file_re.search(line)
            if m:
                current_file = m.group(1).strip()
                last_avg = None
                continue

            if current_file is None:
                continue

            m = avg_re.search(line)
            if m:
                last_avg = float(m.group(1))

            m = time_re.search(line)
            if m:
                results[current_file]["times"].append(float(m.group(1)))

            if "Freeing pipeline" in line and last_avg is not None:
                results[current_file]["fps"].append(last_avg)
                last_avg = None

    return results


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"

    results = parse_benchmarks(filepath)

    if not results:
        print("No benchmark data found.")
        return

    for filename, data in results.items():
        print("=" * 60)
        print(f"File: {filename}")

        fps_values = data["fps"]
        exec_times = data["times"]

        if fps_values:
            mean_fps = sum(fps_values) / len(fps_values)
            print(f"Number of benchmarks:       {len(fps_values)}")
            print(f"Average FPS per benchmark:  {', '.join(f'{v:.2f}' for v in fps_values)}")
            print(f"Average of average FPS:     {mean_fps:.2f}")
        else:
            print("No FPS data found.")

        if exec_times:
            mean_time = sum(exec_times) / len(exec_times)
            print(f"Execution times (s):        {', '.join(f'{t:.3f}' for t in exec_times)}")
            print(f"Average execution time (s): {mean_time:.3f}")
        else:
            print("No execution time data found.")

    # Résumé global
    all_fps = []
    all_times = []

    for data in results.values():
        all_fps.extend(data["fps"])
        all_times.extend(data["times"])

    print("=" * 60)
    print("GLOBAL SUMMARY")

    if all_fps:
        mean_fps = sum(all_fps) / len(all_fps)
        print(f"Total number of benchmarks: {len(all_fps)}")
        print(f"Overall average FPS:         {mean_fps:.2f}")
        print(f"All average FPS:            {', '.join(f'{v:.2f}' for v in all_fps)}")
    else:
        print("No FPS data found.")

    if all_times:
        mean_time = sum(all_times) / len(all_times)
        print(f"Overall execution time (s): {mean_time:.3f}")
        print(f"All execution times (s):    {', '.join(f'{t:.3f}' for t in all_times)}")
    else:
        print("No execution time data found.")


if __name__ == "__main__":
    main()
