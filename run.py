from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import *
import argparse
import cv2
import io
import itertools
import math
import numpy as np
import random
import time

CursorPosition = Tuple[int, int]

@dataclass
class Circle:
  x: int
  y: int
  radius: int

class SleepController:
  def __init__(self, interval: float):
    self.interval = interval
    self._initialized = False
    self._last_time = None

  def begin(self):
    if not self._initialized:
      self._last_time = time.time()
      self._initialized = True

  def end(self):
    curr = time.time()
    elapsed_time = curr - self._last_time
    if self.interval - elapsed_time > 0:
      time.sleep(self.interval - elapsed_time)
    self._last_time = curr

class DetectCursor:
  def __init__(
    self,
    video_path:
    str,
    *,
    cursor_radius: int
  ) -> None:
    # Parameters
    self._cursor_radius: int = cursor_radius
    self._cursor_radius_deviation: int = 3 # `float` works too?
    self._cursor_bright_detect_size_factor: float = 0.9
    # self._cursor_brightness_accept_threshold: float = 0.92
    self._cursor_brightness_accept_threshold: float = 0.86
    self._blur_radius = 5
    # State
    self._state_last_cursor: Optional[Circle] = None
    self._state_reading: bool = False
    # Open video file
    self._cv_video_capture = cv2.VideoCapture(video_path)
    if not self._cv_video_capture.isOpened():
      raise RuntimeError(f"Cannot open video file '{path}'")
    self._video_fps  = self._cv_video_capture.get(cv2.CAP_PROP_FPS)
    self._video_width  = int(self._cv_video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    self._video_height = int(self._cv_video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self._frame_i = -1
    self._frame = None

  @property
  def _frame_t(self) -> float:
    return self._frame_i / self._video_fps

  def _detect_cursor_sized_circles(self, frame: np.ndarray) -> list[Circle]:
    # Detect circles on frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.blur(gray_frame, (self._blur_radius, self._blur_radius))
    circles = cv2.HoughCircles(
      blurred_frame,
      cv2.HOUGH_GRADIENT,
      1, 10,
      param1=50, param2=30,
      minRadius=self._cursor_radius - self._cursor_radius_deviation,
      maxRadius=self._cursor_radius + self._cursor_radius_deviation,
      )
    # Postprocess the output of HoughCircles
    circles = [] if circles is None else circles[0, :]
    circles = [ Circle(*np.around(circle).astype(np.int32)) for circle in circles ]
    return circles

  def _is_circle_bright_enough(self, c: Circle) -> bool:
    # if the region is bright => its probably the cursor since the cursor is bright yellow
    r = round(c.radius * self._cursor_bright_detect_size_factor)
    region = self._frame[c.y-r:c.y+r+1,c.x-r:c.x+r+1]
    if region.size == 0: # sometimes it just happens and I don't know why
      return False
    _, _, brightness_pixels = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HSV))
    brightness = np.average(brightness_pixels) / 255.0
    # Only allow really bright circles to pass
    bright_enough = brightness >= self._cursor_brightness_accept_threshold
    return bright_enough

  def _detect_cursor(self) -> Iterator[Optional[Tuple[int, CursorPosition]]]:
    cands = self._detect_cursor_sized_circles(self._frame)
    cands = list(filter(self._is_circle_bright_enough, cands))
    match cands:
      case []:
        # No cursor found
        return self._state_last_cursor # Can yield none
      case [cand]:
        # Cursor found
        self._state_last_cursor = cand
        return cand
      case _:
        # Multiple cursors detected
        raise NotImplementedError(f"Multiple candidates found: {cands}")

  def _read_frame(self):
    _, frame = self._cv_video_capture.read()
    if frame is None:
      return False
    self._frame = frame
    self._frame_i += 1
    return True

  def cursor_to_coordinate(self, cursor: Circle) -> Tuple[float, float]:
    height_scaler = 0.8

    osu_size = np.asarray([640, 480])
    field_center = np.asarray([256, 192])
    field_size = field_center * 2
    screen_size = np.asarray([self._video_width, self._video_height])
    screen_cursor = np.asarray([cursor.x, cursor.y])

    h = screen_size[1] * height_scaler
    w = h / 3 * 4
    what = np.asarray([w, h])

    norm = (screen_cursor - screen_size / 2) / ( what / 2 ) * (field_size / 2) + (field_size / 2)
    return (norm[0], norm[1])

  def start(self) -> Iterator[Tuple[int, float, Optional[CursorPosition]]]:
    while self._read_frame():
      cursor = self._detect_cursor()
      yield (self._frame_i, self._frame_t, cursor)

  def show(self) -> None:
    cv_window_name: str = "DetectCursor.show()"
    cv2.namedWindow(cv_window_name)

    cv_quit_key: str = 'q'

    sleep_controller = SleepController(1.0 / self._video_fps)

    start_time = time.time()
    while self._read_frame():
      sleep_controller.begin()
      c = self._detect_cursor()

      draw_frame = self._frame.copy()

      if c is not None:
        cv2.circle(draw_frame, (c.x, c.y), c.radius, (255, 0, 0), 2)
        cv2.line(draw_frame, (0, c.y), (c.x, c.y), (0, 0, 255), 1)
        cv2.line(draw_frame, (c.x, 0), (c.x, c.y), (0, 0, 255), 1)
        cv2.putText(draw_frame, f"{(c.x,c.y)}", (c.x + c.radius, c.y + c.radius), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

      # Draw other information
      true_time = time.time() - start_time
      cv2.putText(
        draw_frame,
        f"frame #{self._frame_i} / fps={self._video_fps} / frame_t={self._frame_t:.3f} / true_t={true_time:.3f} / ratio={self._frame_t / true_time}",
        (0, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        bottomLeftOrigin=False
      )

      cv2.imshow(cv_window_name, draw_frame)

      if cv2.waitKey(1) == ord(cv_quit_key):
        print(f"'{cv_quit_key}' key pressed.")
        break
      # Sleep to maintain stable FPS if necessary
      sleep_controller.end()

    cv2.destroyWindow(cv_window_name)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cursor-radius", required=True, type=int) # required to be an int by HoughCircles
  parser.add_argument("replay_render_path", type=str)
  args = parser.parse_args()

  detector = DetectCursor(
    args.replay_render_path,
    cursor_radius = args.cursor_radius,
    )
  detector.show()

if __name__ == "__main__":
  main()
