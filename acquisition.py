import cv2

class VideoAcquisition:
    def __init__(self, source=0):
        """
        Initialize the video acquisition module.
        
        Args:
            source (int or str): Video source. 0 for webcam, or path to video file.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
            
    def read_frame(self):
        """
        Read a single frame from the video source.
        
        Returns:
            ret (bool): True if frame is read correctly, False otherwise.
            frame (numpy.ndarray): The captured frame.
        """
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """
        Release the video capture resource.
        """
        if self.cap.isOpened():
            self.cap.release()
