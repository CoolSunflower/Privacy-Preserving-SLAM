import re
import sys

def compute_average_exhaustive_time(log_path):
    # Match the entire timing line
    # Example line: "  - Timing: Distance 12.3ms, Edges 45.6ms, Total 78.9ms"
    pattern = re.compile(
        r"Timing:\s*Distance\s+[0-9.]+ms,\s*Edges\s+[0-9.]+ms,\s*Total\s+([0-9.]+)ms"
    )

    total_times = []

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                total_times.append(float(match.group(1)))

    if not total_times:
        print("No exhaustive timing entries found in the log.")
        return

    avg_time = sum(total_times) / len(total_times)
    # print(f"Found {len(total_times)} entries.")
    # print(f"Average Exhaustive Total Time: {avg_time:.2f} ms")
    return avg_time

def compute_edges_added(log_path):
    # Regex pattern for edges added line
    # Example: "  - Edges Added: 20"
    pattern = re.compile(
        r"Loop edges added:\s+([0-9]+)"
    )

    edges_added = []

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                edges_added.append(int(match.group(1)))

    if not edges_added:
        print("No edges added entries found in the log.")
        return

    total_edges = sum(edges_added)
    # print(f"Total Edges Added: {total_edges}")
    return total_edges

if __name__ == "__main__":
    log_files = [
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_day_high/vio_city_day_high_base/vio_city_day_high_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_day_low/vio_city_day_low_base/city_day_low_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_day_mid/vio_city_day_base/city_day_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_night_high/vio_city_night_high_base/vio_city_night_high_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_night_low/vio_city_night_low_base/vio_city_night_low_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_night_mid/vio_city_night_mid_base/vio_city_night_mid_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_parking_lot_high/vio_parking_lot_high_base/vio_parking_lot_high_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_parking_lot_low/vio_parking_lot_low_base/vio_parking_lot_low_base.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_parking_lot_mid/vio_parking_lot_base/vio_base_full.txt"
    ]

    for log_file in log_files:
        time = compute_average_exhaustive_time(log_file)
        edges = compute_edges_added(log_file)
        print(f"Log file: {log_file}")
        print(f"\tAverage Exhaustive Total Time: {time:.2f} ms")
        print(f"\tTotal Edges Added: {edges}")