import cv2
import numpy as np
from collections import Counter
import csv
from datetime import datetime
import argparse
import os


# ==========================================
# Config & Arg
# ==========================================
parser = argparse.ArgumentParser(description="Video Wall Sync Analysis Tool")
parser.add_argument('-v', '--video', type=str, required=True,
                    help="Path to the input video file")
parser.add_argument('-hv', action='store_true', dest='hide',
                    help="Hide video")
parser.add_argument('-s', action='store_true', dest='save',
                    help="save all data")
parser.add_argument('-s:f', action='store_true', dest='save_f',
                    help="save frame state data")
parser.add_argument('-s:s', action='store_true', dest='save_s',
                    help="save sync result data")
args = parser.parse_args()

SHOW_VIDEO = not args.hide
SAVE_SYNC = args.save or args.save_s   # -s or -s:s
SAVE_FRAME = args.save or args.save_f  # -s or -s:f

base_dir = ".."

# ==========================================
# Source & Global Variable
# ==========================================
if os.path.isabs(args.video):
    video_path = args.video
else:
    video_path = os.path.join(base_dir, args.video)

if not os.path.exists(video_path):
    print(f"No file\n({video_path})")
    exit()
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

devices = ["dev1", "dev2", "dev3", "dev4"]
device_points = {dev: [] for dev in devices}
current_dev_idx = 0


# ==========================================
# Func: ROI Selection
# ==========================================
def select_points(event, x, y, flags, param):
    global current_dev_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_dev_idx < 4:
            dev = devices[current_dev_idx]
            if len(device_points[dev]) < 8:
                device_points[dev].append((x, y))
                print(f"[{dev}] color box: {len(device_points[dev])}/8")

                # 8개를 모두 찍으면 다음 디바이스로 자동 전환
                if len(device_points[dev]) == 8:
                    current_dev_idx += 1
                    if current_dev_idx < 4:
                        print(
                            f" > Next Device [{devices[current_dev_idx]}]. Click the color boxes.")
                    else:
                        print("-> Setting completed. Press Enter Key.")


# ==========================================
# Func: Color Analysis
# ==========================================
def get_color_state_forced(hsv_img):
    if hsv_img.size == 0:
        return -1
    median_h = np.median(hsv_img[:, :, 0])
    if median_h < 30 or median_h >= 150:
        return 0  # Red
    elif 30 <= median_h < 90:
        return 1             # Green
    elif 90 <= median_h < 150:
        return 2            # Blue
    return 0


# ==========================================
# PHASE 1: Setup ROI
# ==========================================
ret, first_frame = cap.read()
if not ret:
    exit()

cv2.namedWindow("Setup ROI")
cv2.setMouseCallback("Setup ROI", select_points)

print("========= ROI =========")
print(f"-> [{devices[0]}] Click color boxes")

while True:
    temp_frame = first_frame.copy()

    if current_dev_idx < 4:
        msg = f"Setup [{devices[current_dev_idx]}]: {len(device_points[devices[current_dev_idx]])}/8 Clicks"
        color = (0, 255, 255)
    else:
        msg = "All 4 devices ready! Press 'ENTER' to start."
        color = (0, 255, 0)

    cv2.putText(temp_frame, msg, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    for dev, points in device_points.items():
        for pt in points:
            cv2.circle(temp_frame, pt, 5, (255, 0, 255), -1)

    cv2.imshow("Setup ROI", temp_frame)
    if cv2.waitKey(1) & 0xFF == 13:  # Enter 키
        if current_dev_idx >= 4:
            break

cv2.destroyWindow("Setup ROI")


# ==========================================
# PHASE 2: Video Analysis (Sync Detection)
# ==========================================
if SHOW_VIDEO:
    cv2.namedWindow("Sync Analysis")

frame_count = 0
is_changing = False        # 상태 변화 중인지 여부
change_start_frame = 0     # 변화가 시작된 프레임 번호
change_start_ms = 0.0      # 변화가 시작된 절대 시간(ms)

log_data = []         # 측정 결과 저장용 리스트
frame_log_data = []   # 로우 데이터 저장 리스트 (frame_states)
prev_states = {dev: 0 for dev in devices}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    display_frame = frame.copy()
    device_states = {}

    # 각 기기의 최종 색상 상태 추출 (다수결)
    for dev, points in device_points.items():
        box_states = []
        for (cx, cy) in points:
            roi = frame[max(0, cy-10):cy+10, max(0, cx-10):cx+10]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            box_states.append(get_color_state_forced(hsv))

        if box_states:
            vote_result = Counter(box_states)
            top_vote = vote_result.most_common(1)[0]

            if top_vote[1] >= 5:
                most_common_state = top_vote[0]
                prev_states[dev] = most_common_state
            else:
                most_common_state = prev_states[dev]
                # print(f"\n[Hold] Frame {frame_count} = {dev}")
                # print(f" └─ box state: {box_states}")
                # print(
                #     f" └─ 과반수 미달: {dict(vote_result)}, 이전 상태({most_common_state}) 유지\n")

            device_states[dev] = most_common_state

            # 각 기기 화면에 현재 상태 표시
            first_pt = points[0]
            cv2.putText(display_frame, f"{dev}:{most_common_state}",
                        (first_pt[0] - 20, first_pt[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 동기화 변화 감지 알고리즘 (Sync vs Change)
    #  - unique_states의 길이가 1이면 4대 모두 같은 색(동기화 됨)
    #  - unique_states의 길이가 2 이상이면 서로 다른 색(변화 진행 중)
    unique_states = set(device_states.values())

    status_text = ""
    status_color = (0, 0, 0)

    if len(device_states) == 4:
        if len(unique_states) == 1:
            status_text = "SYNC"
            status_color = (0, 255, 0)  # 초록색

            if is_changing:
                is_changing = False
                frames_taken = frame_count - change_start_frame
                time_taken_ms = current_ms - change_start_ms

                # 측정 결과 기록 및 출력
                trial_num = len(log_data) + 1
                log_data.append([trial_num, change_start_frame,
                                frame_count, frames_taken, round(time_taken_ms, 1)])
                # print(
                #     f"[Detected: {trial_num}]:\tFrame Delay: {frames_taken} frames, Transition Time: {time_taken_ms:.1f} ms")

        else:
            status_text = "CHANGING"
            status_color = (0, 0, 255)  # 빨간색

            if not is_changing:
                is_changing = True
                change_start_frame = frame_count
                change_start_ms = current_ms
                # print(f"[Start Detecting]:\t Start Frame: {frame_count}")

        if SAVE_FRAME:
            frame_log_data.append([
                frame_count,
                device_states["dev1"],
                device_states["dev2"],
                device_states["dev3"],
                device_states["dev4"],
                status_text
            ])

        state_str = " | ".join(
            [f"{dev}:{state}" for dev, state in device_states.items()])
        print(f"[Frame {frame_count}] {state_str} => {status_text}")

    if SHOW_VIDEO:
        cv2.imshow("Sync Analysis", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('q'):
            break
    else:
        cv2.waitKey(1)

cap.release()
if SHOW_VIDEO:
    cv2.destroyAllWindows()


# ==========================================
# PHASE 3: Save Results & Reporting
# ==========================================
print("\n\n\n================== [Result] ==================")
now = datetime.now()
time_str = now.strftime('%Y%m%d_%H%M%S')

# display the result
for data in log_data:
    print(f"[{data[0]}] Frame Delay: {data[3]}, Time Delay: {data[4]} ms")

# display the summary
if log_data:
    frame_delays = [data[3] for data in log_data]
    transition_times = [data[4] for data in log_data]

    avg_frame = np.mean(frame_delays)
    max_frame = np.max(frame_delays)
    min_frame = np.min(frame_delays)

    avg_time = np.mean(transition_times)
    max_time = np.max(transition_times)
    min_time = np.min(transition_times)

    print("\n--- [Transition Detection Summary] ---")
    print(f"Count   : {len(log_data):>4}")
    print(f"Average : {avg_frame:>4.1f} frames, {avg_time:>6.1f} ms")
    print(f"Max     : {max_frame:>4} frames, {max_time:>6.1f} ms")
    print(f"Min     : {min_frame:>4} frames, {min_time:>6.1f} ms")
    print("--------------------------------------")
else:
    print("\nNo transition.")

# about delay: -s:s
if SAVE_SYNC and log_data:
    csv_filename = os.path.join(base_dir, f"{time_str}_sync.csv")
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Number", "Start Frame", "End Frame",
                        "Frame Delay", "Time Delay (ms)"])
        writer.writerows(log_data)

        # writer.writerow([])
        # writer.writerow(["---", "Summary", "---", "---", "---"])
        # writer.writerow(["Total Trials", len(log_data), "", "", ""])
        # writer.writerow(["Average", "", "", round(
        #     avg_frame, 1), round(avg_time, 1)])
        # writer.writerow(["Maximum", "", "", max_frame, round(max_time, 1)])
        # writer.writerow(["Minimum", "", "", min_frame, round(min_time, 1)])
    print(f"\nSaved sync delay result in '{csv_filename}'.")

# about frame state: -s:f
if SAVE_FRAME:
    frames_csv_filename = os.path.join(base_dir, f"{time_str}_frame-stat.csv")
    with open(frames_csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "dev1", "dev2", "dev3", "dev4", "Status"])
        writer.writerows(frame_log_data)
    print(f"\nSaved all frame states in '{frames_csv_filename}'.")
