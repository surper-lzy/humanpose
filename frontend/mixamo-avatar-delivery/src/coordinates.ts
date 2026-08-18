import * as THREE from "three";

import type { Joint3 } from "./types.js";


/** Convert right-handed application Z-up metres to Three.js Y-up metres. */
export function applicationJointToThree(
  joint: Exclude<Joint3, null>,
  target: THREE.Vector3 = new THREE.Vector3()
): THREE.Vector3 {
  return target.set(joint[0], joint[2], -joint[1]);
}

export function applicationJointsToThree(
  joints: Joint3[]
): Array<THREE.Vector3 | null> {
  return joints.map((joint) => (
    joint === null ? null : applicationJointToThree(joint)
  ));
}
