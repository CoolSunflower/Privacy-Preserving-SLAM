def calculate_average_psnr(file_path: str) -> float:
    total_psnr = 0.0
    count = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            if "PSNR" in line:
                parts = line.strip().split(',')
                for part in parts:
                    if "PSNR" in part:
                        psnr_value = float(part.split(':')[1].strip())
                        total_psnr += psnr_value
                        count += 1
    
    average_psnr = total_psnr / count if count > 0 else 0.0
    return average_psnr

if __name__ == "__main__":
    file_paths = [
        # "../output/s3e_playground_1_base_final/playground_1/online_plots/psnr_log.txt",
        # "../output/s3e_playground_1_fe_imu_noMamba_full_decay_1/playground_1/online_plots/psnr_log.txt",
        # "../output/s3e_playground_1_fe_imu_noMamba_full_decay_07/playground_1/online_plots/psnr_log.txt",
        # "../output/vio_parking_lot_base_full/parking_lot/online_plots/psnr_log.txt",
        # "../output/vio_parking_lot_fe_imu_full_1decay/parking_lot/online_plots/psnr_log.txt",
        # "../output/vio_parking_lot_fe_imu_full_05decay/parking_lot/online_plots/psnr_log.txt",
        # "../output/vio_parking_lot_fe_imu_mamba_full/parking_lot/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_parking_lot_low/vio_parking_lot_low_base/parking_lot_low/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_parking_lot_low/vio_parking_lot_low_proposed/parking_lot_low/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_parking_lot_mid/vio_parking_lot_base/parking_lot/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_parking_lot_mid/vio_parking_lot_proposed/parking_lot/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_parking_lot_high/vio_parking_lot_high_base/parking_lot_high/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_parking_lot_high/vio_parking_lot_high_proposed/parking_lot_high/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_day_low/vio_city_day_low_base/city_day_low/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_day_low/vio_city_day_low_proposed/city_day_low/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_day_mid/vio_city_day_base/parking_lot/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_day_mid/vio_city_day_proposed/city_day_mid/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_day_high/vio_city_day_high_base/city_day_high/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_day_high/vio_city_day_high_proposed/city_day_high/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_night_low/vio_city_night_low_base/city_night_low/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_night_low/vio_city_night_low_proposed/city_night_low/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_night_mid/vio_city_night_mid_base/city_night_mid/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_night_mid/vio_city_night_mid_proposed/city_night_mid/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_night_high/vio_city_night_high_base/city_night_high/online_plots/psnr_log.txt",
        # "../output/COMPLETED/vio_city_night_high/vio_city_night_high_proposed/city_night_high/online_plots/psnr_log.txt"
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP-Phase2/MainSLAM/output/WildGS/advio/advio-13/advio-13/online_plots/psnr_log.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP-Phase2/MainSLAM/output/WildGS/advio/advio-14/advio-14/online_plots/psnr_log.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP-Phase2/MainSLAM/output/WildGS/advio/advio-15/advio-15/online_plots/psnr_log.txt",
        "/home/gpuuser5/Ashok/Adarsh_220101003/BTP-Phase2/MainSLAM/output/WildGS/advio/advio-16/advio-16/online_plots/psnr_log.txt",
    ]
    for f in file_paths:
        avg_psnr = calculate_average_psnr(f)
        print(f"{f}: {avg_psnr:.4f}")