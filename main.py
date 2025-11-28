import cv2
from modules.acquisition import VideoAcquisition
from modules.preprocessing import FramePreprocessor
from modules.detection import ObjectDetector
from modules.tracking import ObjectTracker
from modules.reporting import Reporter

def main():
    # Initialize modules
    # Configuration
    # Set video_source to 0 for webcam, or provide a string path for a video file
    # Example: video_source = "traffic_video.mp4"
    video_source = 0 
    acquisition = VideoAcquisition(video_source)
    preprocessor = FramePreprocessor()
    detector = ObjectDetector('yolov8n.pt') # You can change model here
    tracker = ObjectTracker()
    reporter = Reporter()

    print("Starting Automated Object Detection and Tracking System...")
    print("Press 'q' to exit.")

    try:
        while True:
            # 1. Acquisition
            ret, frame = acquisition.read_frame()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            # 2. Preprocessing
            # frame = preprocessor.resize(frame, width=640) # Optional resize
            # frame = preprocessor.apply_gaussian_blur(frame) # Optional blur

            # 3. Detection
            detections = detector.detect(frame)

            # 4. Tracking
            tracks = tracker.update(detections)

            # 5. Reporting/Visualization
            frame = reporter.draw_tracks(frame, tracks)
            frame = reporter.display_metrics(frame)

            # Display
            cv2.imshow('Object Detection and Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write(traceback.format_exc())
        print(f"An error occurred: {e}")
    finally:
        acquisition.release()
        cv2.destroyAllWindows()
        print("System stopped.")

if __name__ == "__main__":
    main()
