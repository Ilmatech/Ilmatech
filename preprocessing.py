import cv2

class FramePreprocessor:
    def __init__(self):
        pass

    def apply_gaussian_blur(self, frame, kernel_size=(5, 5)):
        """
        Apply Gaussian Blur to the frame.
        
        Args:
            frame (numpy.ndarray): Input frame.
            kernel_size (tuple): Kernel size for blurring.
            
        Returns:
            numpy.ndarray: Blurred frame.
        """
        return cv2.GaussianBlur(frame, kernel_size, 0)

    def resize(self, frame, width=None, height=None):
        """
        Resize the frame to specified width and height.
        If only one is provided, maintains aspect ratio.
        
        Args:
            frame (numpy.ndarray): Input frame.
            width (int): Target width.
            height (int): Target height.
            
        Returns:
            numpy.ndarray: Resized frame.
        """
        if width is None and height is None:
            return frame
            
        h, w = frame.shape[:2]
        
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)
            
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
