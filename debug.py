import sys
import traceback

try:
    with open("debug_status.txt", "w") as f:
        f.write("Starting imports...\n")
        
    import cv2
    import ultralytics
    import filterpy
    import scipy
    import numpy
    
    with open("debug_status.txt", "a") as f:
        f.write("Library imports successful.\n")
    
    from modules.acquisition import VideoAcquisition
    from modules.preprocessing import FramePreprocessor
    from modules.detection import ObjectDetector
    from modules.tracking import ObjectTracker
    from modules.reporting import Reporter
    
    with open("debug_status.txt", "a") as f:
        f.write("Module imports successful.\n")

except Exception:
    with open("error_log.txt", "w") as f:
        f.write(traceback.format_exc())
