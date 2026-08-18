export const HALPE26_NAMES = [
  "nose",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
  "head",
  "neck",
  "hip",
  "left_big_toe",
  "right_big_toe",
  "left_small_toe",
  "right_small_toe",
  "left_heel",
  "right_heel"
] as const;

export const HALPE26_LINKS = [
  [15, 13], [13, 11], [11, 19],
  [16, 14], [14, 12], [12, 19],
  [17, 18], [18, 19],
  [18, 5], [5, 7], [7, 9],
  [18, 6], [6, 8], [8, 10],
  [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
  [15, 20], [15, 22], [15, 24],
  [16, 21], [16, 23], [16, 25]
] as const;

export type Halpe26Name = typeof HALPE26_NAMES[number];
