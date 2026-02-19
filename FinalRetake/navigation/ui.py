from __future__ import annotations
import cv2
import numpy as np

def pick_points_interactive(
    image_bgr: np.ndarray, 
    num_points: int = 2, 
    window_name: str = "Click to select points (ESC to cancel)"
) -> list[tuple[int, int]]:
    """
    Opens an OpenCV window allowing the user to click `num_points` on the image.
    Returns a list of (x, y) tuples.
    """
    pts = []
    disp = image_bgr.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < num_points:
                pts.append((x, y))
                # Draw a marker to show the click
                color = (0, 255, 0) if len(pts) == 1 else (0, 0, 255)
                label = "A" if len(pts) == 1 else "B"
                cv2.circle(disp, (x, y), 6, color, -1, cv2.LINE_AA)
                cv2.circle(disp, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(disp, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.imshow(window_name, disp)

    cv2.imshow(window_name, disp)
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"Please click {num_points} point(s) on the image window...")
    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # ESC key
            print("Selection cancelled.")
            break
        if len(pts) == num_points:
            cv2.waitKey(600)  # Brief pause so user sees their last click
            break

    cv2.destroyWindow(window_name)
    return pts