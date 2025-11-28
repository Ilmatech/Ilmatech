import cv2
import time

class Reporter:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def draw_tracks(self, frame, tracks):
        """
        Draw tracking results on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame.
            tracks (numpy.ndarray): Tracks [x1, y1, x2, y2, track_id]
            
        Returns:
            numpy.ndarray: Frame with drawn tracks.
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(frame, str(track_id), (x1, y1 - 10), self.font, 0.9, (0, 255, 0), 2)
            
        return frame

    def display_metrics(self, frame):
        """
        Calculate and display FPS on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame.
            
        Returns:
            numpy.ndarray: Frame with FPS.
        """
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = self.new_frame_time
        
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        return frame
