"""
人体动作分析系统 - 完整版
使用 MediaPipe 和 OpenCV 进行高精度姿态检测和动作识别

依赖安装:
pip install opencv-python mediapipe numpy pandas matplotlib scipy

使用方法:
python motion_analyzer.py --source 0  # 使用摄像头
python motion_analyzer.py --source video.mp4  # 分析视频文件
python motion_analyzer.py --source 0 --record output.avi  # 录制分析结果
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime
from collections import deque
import time
import math
import os
import urllib.request


class MotionAnalyzer:
    """动作分析器主类"""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, skip_model=False):
        """
        初始化动作分析器

        Args:
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            skip_model: 跳过模型下载/加载（无检测模式）
        """
        self.skip_model = skip_model
        # 初始化MediaPipe（兼容 solutions 与 tasks 两种API）
        self.use_tasks = not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose')
        self.pose = None
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None

        if not self.use_tasks:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=1
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self.mp_tasks_vision = vision
            # 下载模型（若不存在）
            model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker_heavy.task')
            self.landmarker = None
            if not self.skip_model:
                try:
                    if not os.path.exists(model_path):
                        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float32/1/pose_landmarker_heavy.task'
                        print('正在下载姿态模型...')
                        urllib.request.urlretrieve(url, model_path)
                        print('模型下载完成：', model_path)
                    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
                    opts = self.mp_tasks_vision.PoseLandmarkerOptions(
                        base_options=base_opts,
                        output_segmentation_masks=False
                    )
                    self.landmarker = self.mp_tasks_vision.PoseLandmarker.create_from_options(opts)
                except Exception as e:
                    print('警告：姿态模型不可用，将以无检测模式运行。', str(e))
            # MediaPipe Pose关节索引映射（33点）
            self.pose_idx = {
                'NOSE': 0,
                'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
                'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
                'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
                'LEFT_HIP': 23, 'RIGHT_HIP': 24,
                'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
                'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28
            }

        # 动作历史记录（使用队列存储最近N帧的数据）
        self.action_history = deque(maxlen=30)  # 保存最近30帧
        self.angle_history = deque(maxlen=10)  # 保存最近10帧的角度

        # 统计数据
        self.action_stats = {}
        self.action_count = {}
        self.analysis_results = []

        # 动作计数器
        self.squat_counter = 0
        self.pushup_counter = 0
        self.jump_counter = 0
        self.is_down = False  # 用于计数状态跟踪

    def calculate_angle(self, point1, point2, point3):
        """
        计算三个点之间的角度

        Args:
            point1, point2, point3: 三个关键点坐标

        Returns:
            角度值（度）
        """
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def detect_action(self, landmarks):
        """
        检测当前动作类型

        Args:
            landmarks: MediaPipe检测到的关键点

        Returns:
            动作类型和详细信息
        """
        if not landmarks:
            return "UNKNOWN", {}

        # 获取关键点
        left_shoulder = self._get_lm(landmarks, 'LEFT_SHOULDER')
        right_shoulder = self._get_lm(landmarks, 'RIGHT_SHOULDER')
        left_elbow = self._get_lm(landmarks, 'LEFT_ELBOW')
        right_elbow = self._get_lm(landmarks, 'RIGHT_ELBOW')
        left_wrist = self._get_lm(landmarks, 'LEFT_WRIST')
        right_wrist = self._get_lm(landmarks, 'RIGHT_WRIST')
        left_hip = self._get_lm(landmarks, 'LEFT_HIP')
        right_hip = self._get_lm(landmarks, 'RIGHT_HIP')
        left_knee = self._get_lm(landmarks, 'LEFT_KNEE')
        right_knee = self._get_lm(landmarks, 'RIGHT_KNEE')
        left_ankle = self._get_lm(landmarks, 'LEFT_ANKLE')
        right_ankle = self._get_lm(landmarks, 'RIGHT_ANKLE')
        nose = self._get_lm(landmarks, 'NOSE')

        # 计算关键角度
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        left_shoulder_angle = self.calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = self.calculate_angle(right_elbow, right_shoulder, right_hip)

        # 保存角度数据
        angle_data = {
            'left_elbow': left_elbow_angle,
            'right_elbow': right_elbow_angle,
            'left_knee': left_knee_angle,
            'right_knee': right_knee_angle,
            'left_hip': left_hip_angle,
            'right_hip': right_hip_angle,
            'left_shoulder': left_shoulder_angle,
            'right_shoulder': right_shoulder_angle
        }
        self.angle_history.append(angle_data)

        # 计算身体倾斜度
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2

        # 动作识别逻辑
        action = "STANDING"
        details = {}

        # 1. 深蹲检测
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        if avg_knee_angle < 100 and hip_mid_y > shoulder_mid_y:
            action = "SQUATTING"
            details['knee_angle'] = avg_knee_angle
            # 深蹲计数
            if avg_knee_angle < 90 and not self.is_down:
                self.squat_counter += 1
                self.is_down = True
            elif avg_knee_angle > 160:
                self.is_down = False

        # 2. 俯卧撑检测
        elif nose.y > hip_mid_y and (left_elbow_angle < 90 or right_elbow_angle < 90):
            action = "PUSH_UP"
            details['elbow_angle'] = (left_elbow_angle + right_elbow_angle) / 2

        # 3. 举手检测
        elif (left_wrist.y < left_shoulder.y - 0.1 or right_wrist.y < right_shoulder.y - 0.1):
            action = "RAISING_HANDS"
            if left_wrist.y < nose.y and right_wrist.y < nose.y:
                action = "HANDS_UP"

        # 4. 坐姿检测
        elif left_knee_angle < 120 and right_knee_angle < 120 and left_hip_angle < 120:
            action = "SITTING"
            details['hip_angle'] = (left_hip_angle + right_hip_angle) / 2

        # 5. 跳跃检测（脚离地）
        elif (left_ankle.y < left_knee.y - 0.1 and right_ankle.y < right_knee.y - 0.1):
            action = "JUMPING"

        # 6. 行走检测（腿部交替运动）
        elif len(self.angle_history) >= 5:
            # 检测膝盖角度变化
            knee_variation = np.std([h['left_knee'] for h in self.angle_history])
            if knee_variation > 10:
                action = "WALKING"
                details['movement_intensity'] = knee_variation

        # 7. 向前弯腰
        elif left_hip_angle < 90 or right_hip_angle < 90:
            action = "BENDING_FORWARD"
            details['bend_angle'] = (left_hip_angle + right_hip_angle) / 2

        # 更新统计
        self.action_history.append(action)
        if action not in self.action_count:
            self.action_count[action] = 0
        self.action_count[action] += 1

        # 添加计数信息
        details['squat_count'] = self.squat_counter
        details['pushup_count'] = self.pushup_counter
        details['jump_count'] = self.jump_counter

        return action, details

    def calculate_body_metrics(self, landmarks):
        """
        计算身体指标

        Args:
            landmarks: MediaPipe检测到的关键点

        Returns:
            身体指标字典
        """
        if not landmarks:
            return {}

        left_shoulder = self._get_lm(landmarks, 'LEFT_SHOULDER')
        right_shoulder = self._get_lm(landmarks, 'RIGHT_SHOULDER')
        left_hip = self._get_lm(landmarks, 'LEFT_HIP')
        right_hip = self._get_lm(landmarks, 'RIGHT_HIP')
        nose = self._get_lm(landmarks, 'NOSE')
        left_ankle = self._get_lm(landmarks, 'LEFT_ANKLE')
        right_ankle = self._get_lm(landmarks, 'RIGHT_ANKLE')

        # 计算身体中心
        body_center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        body_center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4

        # 计算身体高度（从鼻子到脚踝的平均距离）
        height = ((nose.y - left_ankle.y) + (nose.y - right_ankle.y)) / 2

        # 计算肩宽
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)

        # 计算身体对称性
        left_side = abs(left_shoulder.x - left_hip.x)
        right_side = abs(right_shoulder.x - right_hip.x)
        symmetry = 1 - abs(left_side - right_side) / max(left_side, right_side)

        return {
            'center_x': body_center_x,
            'center_y': body_center_y,
            'height': height,
            'shoulder_width': shoulder_width,
            'symmetry': symmetry
        }

    def analyze_frame(self, frame):
        """
        分析单帧图像

        Args:
            frame: 输入图像帧

        Returns:
            处理后的图像和分析结果
        """
        # 转换颜色空间
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 进行姿态检测
        detection_ok = False
        landmarks_list = None
        if not self.use_tasks:
            results = self.pose.process(image)
            detection_ok = bool(results.pose_landmarks)
            if detection_ok:
                landmarks_list = results.pose_landmarks.landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.landmarker is not None:
                results = self.landmarker.detect(mp_image)
                detection_ok = bool(results.pose_landmarks)
                if detection_ok and len(results.pose_landmarks) > 0:
                    landmarks_list = results.pose_landmarks[0]

        # 转回BGR用于显示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        action = "NO_PERSON_DETECTED"
        details = {}
        metrics = {}

        if detection_ok and landmarks_list:
            # 绘制姿态关键点
            if not self.use_tasks and self.mp_drawing is not None:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            elif self.use_tasks:
                for lm in landmarks_list:
                    cx, cy = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                    cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)

            # 检测动作
            if not self.use_tasks:
                action, details = self.detect_action(landmarks_list)
            else:
                # 使用索引映射构建与solutions一致的访问方式
                action, details = self.detect_action(landmarks_list)

            # 计算身体指标
            metrics = self.calculate_body_metrics(landmarks_list)

            # 在图像上显示信息
            self._draw_info(image, action, details, metrics)

        # 记录分析结果
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.analysis_results.append({
            'timestamp': timestamp,
            'action': action,
            'details': details,
            'metrics': metrics
        })

        return image, action, details

    def _draw_info(self, image, action, details, metrics):
        """在图像上绘制信息"""
        h, w, _ = image.shape

        # 动作类型
        action_text = self._translate_action(action)
        cv2.rectangle(image, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(image, f'Action: {action_text}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 计数信息
        y_offset = 80
        if 'squat_count' in details:
            cv2.putText(image, f"Squats: {details['squat_count']}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30

        # 角度信息
        if 'knee_angle' in details:
            cv2.putText(image, f"Knee Angle: {details['knee_angle']:.1f}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # 身体指标
        if 'symmetry' in metrics:
            cv2.putText(image, f"Symmetry: {metrics['symmetry']:.2f}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _translate_action(self, action):
        """翻译动作名称"""
        translations = {
            'STANDING': '站立',
            'SQUATTING': '深蹲',
            'PUSH_UP': '俯卧撑',
            'RAISING_HANDS': '举手',
            'HANDS_UP': '双手举起',
            'SITTING': '坐姿',
            'JUMPING': '跳跃',
            'WALKING': '行走',
            'BENDING_FORWARD': '前倾',
            'NO_PERSON_DETECTED': '未检测到人',
            'UNKNOWN': '未知动作'
        }
        return translations.get(action, action)
    
    def _get_lm(self, landmarks, name):
        """统一获取关键点（兼容solutions与tasks）"""
        if not self.use_tasks:
            # solutions 枚举
            return landmarks[getattr(self.mp_pose.PoseLandmark, name).value]
        else:
            # tasks 索引
            idx = self.pose_idx[name]
            return landmarks[idx]

    def generate_report(self, output_path='motion_analysis_report.json'):
        """生成分析报告"""
        # 统计动作频率
        action_frequency = {}
        for action in self.action_history:
            action_frequency[action] = action_frequency.get(action, 0) + 1

        # 计算动作转换
        transitions = []
        for i in range(len(self.analysis_results) - 1):
            if self.analysis_results[i]['action'] != self.analysis_results[i + 1]['action']:
                transitions.append({
                    'from': self.analysis_results[i]['action'],
                    'to': self.analysis_results[i + 1]['action'],
                    'timestamp': self.analysis_results[i + 1]['timestamp']
                })

        report = {
            'summary': {
                'total_frames': len(self.analysis_results),
                'action_frequency': action_frequency,
                'action_count': self.action_count,
                'squat_count': self.squat_counter,
                'pushup_count': self.pushup_counter,
                'jump_count': self.jump_counter
            },
            'transitions': transitions,
            'detailed_results': self.analysis_results[-100:]  # 最后100帧的详细数据
        }

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n分析报告已保存至: {output_path}")
        self._print_summary(report['summary'])

        return report

    def _print_summary(self, summary):
        """打印摘要信息"""
        print("\n" + "=" * 50)
        print("动作分析摘要")
        print("=" * 50)
        print(f"总帧数: {summary['total_frames']}")
        print(f"\n深蹲次数: {summary['squat_count']}")
        print(f"俯卧撑次数: {summary['pushup_count']}")
        print(f"跳跃次数: {summary['jump_count']}")
        print("\n动作频率:")
        for action, count in sorted(summary['action_frequency'].items(),
                                    key=lambda x: x[1], reverse=True):
            percentage = (count / summary['total_frames']) * 100
            print(f"  {self._translate_action(action)}: {count} ({percentage:.1f}%)")
        print("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='人体动作分析系统')
    parser.add_argument('--source', type=str, default='0',
                        help='视频源：0为摄像头，或视频文件路径')
    parser.add_argument('--record', type=str, default=None,
                        help='输出视频文件路径')
    parser.add_argument('--report', type=str, default='motion_report.json',
                        help='分析报告输出路径')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                        help='最小检测置信度')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                        help='最小跟踪置信度')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示窗口，适合命令行/远程运行')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最多处理的帧数，达到后自动结束')
    parser.add_argument('--skip-model', action='store_true',
                        help='跳过模型下载/加载（无检测模式运行）')

    args = parser.parse_args()

    # 初始化分析器
    analyzer = MotionAnalyzer(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        skip_model=args.skip_model
    )

    # 打开视频源
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {args.source}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height} @ {fps}fps")

    # 初始化视频写入器
    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args.record, fourcc, fps, (width, height))
        print(f"录制至: {args.record}")

    print("\n开始分析... (按 'q' 退出, 'r' 生成报告)")

    # 主循环
    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("视频结束或读取失败")
                break

            # 分析帧
            analyzed_frame, action, details = analyzer.analyze_frame(frame)

            # 显示FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(analyzed_frame, f'FPS: {current_fps:.1f}',
                        (width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 保存帧
            if writer:
                writer.write(analyzed_frame)

            # 显示结果
            if not args.no_display:
                cv2.imshow('Motion Analysis', analyzed_frame)
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    analyzer.generate_report(args.report)

            # 自动结束条件
            if args.max_frames is not None and frame_count >= args.max_frames:
                print(f"达到最大帧数 {args.max_frames}，自动结束。")
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        # 生成最终报告
        print("\n生成最终分析报告...")
        analyzer.generate_report(args.report)

        # 清理资源
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print("\n分析完成！")


if __name__ == '__main__':
    main()
