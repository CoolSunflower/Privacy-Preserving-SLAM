import re
import sys

def compute_average_faiss_time(log_path):
    # Regex pattern for the full FAISS timing line
    # Example: "  - Timing: FAISS 1.3ms, Verify 5.9ms, Total 16.2ms"
    pattern = re.compile(
        r"Timing:\s*FAISS\s+[0-9.]+ms,\s*Verify\s+[0-9.]+ms,\s*Total\s+([0-9.]+)ms"
    )

    total_times = []

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                total_times.append(float(match.group(1)))

    if not total_times:
        print("No FAISS timing entries found in the log.")
        return

    avg_time = sum(total_times) / len(total_times)
    # print(f"Found {len(total_times)} FAISS timing entries.")
    # print(f"Average FAISS Total Time: {avg_time:.2f} ms")
    return avg_time

def compute_loop_edges_added(log_path):
    # Regex pattern for loop edges added line
    # Example: "  - Loop Edges Added: 15"
    """
      - Candidates evaluated: 86
  - Loop edges added: 0
  - Total edges: 86

    """
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
        print("No loop edges added entries found in the log.")
        return

    total_edges = sum(edges_added)
    # print(f"Total Loop Edges Added: {total_edges}")
    return total_edges

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python compute_avg_faiss_time.py <log_file>")
    #     sys.exit(1)

    # log_file = sys.argv[1]
    log_files = [
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_day_high/vio_city_day_high_proposed/vio_city_day_high_proposed.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_day_low/vio_city_day_low_proposed/city_day_low_proposed.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_day_mid/vio_city_day_proposed/city_day_mid_mamba.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_night_high/vio_city_night_high_proposed/vio_city_night_high_proposed.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_night_low/vio_city_night_low_proposed/vio_city_night_low_proposed.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_city_night_mid/vio_city_night_mid_proposed/vio_city_night_mid_proposed.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_parking_lot_high/vio_parking_lot_high_proposed/vio_parking_lot_high_proposed.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_parking_lot_low/vio_parking_lot_low_proposed/vio_parking_lot_low_proposed.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP3/WildGS-SLAM/output/COMPLETED/vio_parking_lot_mid/vio_parking_lot_proposed/vio_parking_lot_fe_depthCorr_mamba.txt",
    ]
    for log_file in log_files:
        avg_time = compute_average_faiss_time(log_file)
        edges_added = compute_loop_edges_added(log_file)
        print(f"Log file: {log_file}")
        print(f"\tAverage FAISS Total Time: {avg_time:.2f} ms")
        print(f"\tTotal Loop Edges Added: {edges_added}")
