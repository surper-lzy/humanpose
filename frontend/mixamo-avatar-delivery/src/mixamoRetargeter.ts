import * as THREE from "three";

import { applicationJointsToThree } from "./coordinates.js";
import type { StickmanPayloadV1 } from "./types.js";


const REQUIRED_BONES = [
  "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
  "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
  "RightShoulder", "RightArm", "RightForeArm", "RightHand",
  "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
  "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"
] as const;

const TRACKING_RESET_STATUSES = new Set([
  "partial_person_out_of_frame",
  "awaiting_full_reentry"
]);

export interface MixamoRetargeterOptions {
  /** Map canonical Mixamo names to model-specific bone names. */
  boneAliases?: Record<string, string>;
  scaleMultiplier?: number;
  autoScale?: boolean;
  minimumScaleSamples?: number;
  minimumScale?: number;
  maximumScale?: number;
  maximumRotationSpeedDegS?: number;
  rotationResponse?: number;
}

interface BodyBasis {
  quaternion: THREE.Quaternion;
  lateral: THREE.Vector3;
  depth: THREE.Vector3;
  up: THREE.Vector3;
}

function canonicalBoneName(name: string): string {
  const parts = name.split(":");
  const withoutNamespace = parts[parts.length - 1] ?? name;
  return withoutNamespace.trim();
}

function unit(value: THREE.Vector3): THREE.Vector3 | null {
  const length = value.length();
  if (!Number.isFinite(length) || length <= 1e-8) return null;
  return value.clone().multiplyScalar(1 / length);
}

function frameQuaternion(
  primaryValue: THREE.Vector3,
  secondaryValue: THREE.Vector3
): THREE.Quaternion | null {
  const primary = unit(primaryValue);
  const secondaryInput = unit(secondaryValue);
  if (primary === null || secondaryInput === null) return null;
  const secondary = secondaryInput.addScaledVector(
    primary,
    -secondaryInput.dot(primary)
  );
  if (unit(secondary) === null) return null;
  secondary.normalize();
  const tertiary = new THREE.Vector3().crossVectors(primary, secondary);
  if (unit(tertiary) === null) return null;
  tertiary.normalize();
  secondary.crossVectors(tertiary, primary).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(primary, secondary, tertiary)
  ).normalize();
}

function bodyBasis(
  upValue: THREE.Vector3,
  lateralValue: THREE.Vector3
): BodyBasis | null {
  const up = unit(upValue);
  const lateralInput = unit(lateralValue);
  if (up === null || lateralInput === null) return null;
  const lateral = lateralInput.addScaledVector(up, -lateralInput.dot(up));
  if (unit(lateral) === null) return null;
  lateral.normalize();
  const depth = new THREE.Vector3().crossVectors(up, lateral);
  if (unit(depth) === null) return null;
  depth.normalize();
  lateral.crossVectors(depth, up).normalize();
  return {
    quaternion: new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(lateral, depth, up)
    ).normalize(),
    lateral,
    depth,
    up
  };
}

function composeFrameTarget(
  targetFrame: THREE.Quaternion,
  bindFrame: THREE.Quaternion,
  bindWorld: THREE.Quaternion
): THREE.Quaternion {
  return targetFrame.clone()
    .multiply(bindFrame.clone().invert())
    .multiply(bindWorld)
    .normalize();
}

function limitedRotation(
  previous: THREE.Quaternion,
  target: THREE.Quaternion,
  maximumStepRad: number,
  response: number
): THREE.Quaternion {
  const angle = previous.angleTo(target);
  if (!Number.isFinite(angle) || angle <= 1e-8) return target.clone();
  const fraction = Math.min(1, maximumStepRad / angle) * response;
  return previous.clone().slerp(target, fraction).normalize();
}

function midpoint(a: THREE.Vector3, b: THREE.Vector3): THREE.Vector3 {
  return a.clone().add(b).multiplyScalar(0.5);
}

function isBone(object: THREE.Object3D | null): object is THREE.Bone {
  return object !== null && (object as THREE.Bone).isBone === true;
}

export class MixamoRetargeter {
  readonly root: THREE.Object3D;

  private readonly boneAliases: Record<string, string>;
  private readonly scaleMultiplier: number;
  private readonly autoScale: boolean;
  private readonly minimumScaleSamples: number;
  private readonly minimumScale: number;
  private readonly maximumScale: number;
  private readonly maximumRotationSpeedDegS: number;
  private readonly rotationResponse: number;
  private readonly boneByCanonical = new Map<string, THREE.Bone>();
  private readonly orderedBones: THREE.Bone[];
  private readonly controlledBones = new Set<THREE.Bone>();
  private readonly bindLocalPosition = new Map<THREE.Bone, THREE.Vector3>();
  private readonly bindLocalQuaternion = new Map<THREE.Bone, THREE.Quaternion>();
  private readonly bindLocalScale = new Map<THREE.Bone, THREE.Vector3>();
  private readonly bindWorldPosition = new Map<THREE.Bone, THREE.Vector3>();
  private readonly bindWorldQuaternion = new Map<THREE.Bone, THREE.Quaternion>();
  private readonly bindFrames = new Map<string, THREE.Quaternion>();
  private readonly bindBodyBasis: BodyBasis;
  private readonly bindStature: number;
  private readonly previousLocal = new Map<THREE.Bone, THREE.Quaternion>();
  private readonly previousGlobal = new Map<THREE.Bone, THREE.Quaternion>();
  private readonly scaleSamples: number[] = [];
  private calibratedScale: number | null = null;
  private lastTimestampMs: number | null = null;

  constructor(root: THREE.Object3D, options: MixamoRetargeterOptions = {}) {
    this.root = root;
    this.boneAliases = options.boneAliases ?? {};
    this.scaleMultiplier = options.scaleMultiplier ?? 1;
    this.autoScale = options.autoScale ?? true;
    this.minimumScaleSamples = options.minimumScaleSamples ?? 5;
    this.minimumScale = options.minimumScale ?? 0.65;
    this.maximumScale = options.maximumScale ?? 1.5;
    this.maximumRotationSpeedDegS = options.maximumRotationSpeedDegS ?? 180;
    this.rotationResponse = options.rotationResponse ?? 0.78;
    if (!(this.scaleMultiplier > 0)) throw new Error("scaleMultiplier must be positive");
    if (!(this.minimumScaleSamples >= 1)) throw new Error("minimumScaleSamples must be positive");
    if (!(this.minimumScale > 0 && this.maximumScale >= this.minimumScale)) {
      throw new Error("invalid Mixamo scale range");
    }
    if (!(this.maximumRotationSpeedDegS > 0)) {
      throw new Error("maximumRotationSpeedDegS must be positive");
    }
    if (!(this.rotationResponse > 0 && this.rotationResponse <= 1)) {
      throw new Error("rotationResponse must be in (0, 1]");
    }

    this.root.updateMatrixWorld(true);
    const allBones: THREE.Bone[] = [];
    const boneByRawName = new Map<string, THREE.Bone>();
    root.traverse((object) => {
      if (!isBone(object)) return;
      allBones.push(object);
      boneByRawName.set(object.name, object);
      const canonical = canonicalBoneName(object.name);
      if (!this.boneByCanonical.has(canonical)) {
        this.boneByCanonical.set(canonical, object);
      }
    });
    for (const [canonical, actual] of Object.entries(this.boneAliases)) {
      const bone = boneByRawName.get(actual) ??
        allBones.find((candidate) => canonicalBoneName(candidate.name) === actual);
      if (bone === undefined) {
        throw new Error(`Mixamo bone alias ${canonical} -> ${actual} was not found`);
      }
      this.boneByCanonical.set(canonical, bone);
    }
    const missing = REQUIRED_BONES.filter((name) => !this.boneByCanonical.has(name));
    if (missing.length > 0) {
      throw new Error(`Mixamo skeleton is missing bones: ${missing.join(", ")}`);
    }

    const hips = this.bone("Hips");
    this.orderedBones = [];
    const visit = (bone: THREE.Bone): void => {
      this.orderedBones.push(bone);
      for (const child of bone.children) {
        if (isBone(child)) visit(child);
      }
    };
    visit(hips);

    for (const bone of this.orderedBones) {
      this.bindLocalPosition.set(bone, bone.position.clone());
      this.bindLocalQuaternion.set(bone, bone.quaternion.clone());
      this.bindLocalScale.set(bone, bone.scale.clone());
      this.bindWorldPosition.set(bone, bone.getWorldPosition(new THREE.Vector3()));
      this.bindWorldQuaternion.set(
        bone,
        bone.getWorldQuaternion(new THREE.Quaternion()).normalize()
      );
    }

    const bindHips = this.bindPosition("Hips");
    this.bindBodyBasis = bodyBasis(
      this.bindPosition("Neck").sub(bindHips),
      this.bindPosition("LeftArm").sub(this.bindPosition("RightArm"))
    ) ?? (() => { throw new Error("Mixamo bind torso frame is degenerate"); })();
    this.buildBindFrames();

    const headTop = this.boneByCanonical.has("HeadTop_End")
      ? this.bindPosition("HeadTop_End")
      : this.bindPosition("Head");
    const footCenter = midpoint(
      this.bindPosition("LeftToeBase"),
      this.bindPosition("RightToeBase")
    );
    this.bindStature = headTop.distanceTo(footCenter);
    if (!(this.bindStature > 1e-6)) {
      throw new Error("Mixamo bind stature is degenerate");
    }
  }

  get avatarScale(): number {
    return (this.calibratedScale ?? 1) * this.scaleMultiplier;
  }

  solve(payload: StickmanPayloadV1): boolean {
    if (TRACKING_RESET_STATUSES.has(payload.status)) {
      this.resetTracking(true);
      return false;
    }
    const points = applicationJointsToThree(payload.joints);
    const rootPosition = points[19] ?? (
      points[11] !== null && points[12] !== null
        ? midpoint(points[11], points[12])
        : null
    );
    if (rootPosition === null) {
      this.resetTracking(false);
      return false;
    }

    let deltaTimeS = 1 / 15;
    if (this.lastTimestampMs !== null) {
      const measured = (payload.timestamp_ms - this.lastTimestampMs) / 1000;
      if (measured > 0 && Number.isFinite(measured)) {
        deltaTimeS = THREE.MathUtils.clamp(measured, 1 / 120, 0.25);
      } else {
        this.previousGlobal.clear();
        this.previousLocal.clear();
      }
    }
    this.lastTimestampMs = payload.timestamp_ms;
    this.updateScale(points);

    const liveBody = points[18] !== null
      ? this.liveBodyBasis(points, rootPosition)
      : null;
    const hips = this.bone("Hips");
    const bindHipsWorld = this.bindQuaternion("Hips");
    let rootGlobal: THREE.Quaternion;
    if (liveBody !== null) {
      const rawRoot = composeFrameTarget(
        liveBody.quaternion,
        this.bindBodyBasis.quaternion,
        bindHipsWorld
      );
      const previous = this.previousGlobal.get(hips);
      rootGlobal = previous === undefined
        ? rawRoot
        : limitedRotation(
          previous,
          rawRoot,
          THREE.MathUtils.degToRad(this.maximumRotationSpeedDegS * deltaTimeS),
          this.rotationResponse
        );
    } else {
      rootGlobal = this.previousGlobal.get(hips)?.clone() ?? bindHipsWorld;
    }

    const bindToRoot = rootGlobal.clone().multiply(bindHipsWorld.clone().invert());
    const bodyDepth = liveBody?.depth.clone() ??
      this.bindBodyBasis.depth.clone().applyQuaternion(bindToRoot);
    const bodyUp = liveBody?.up.clone() ??
      this.bindBodyBasis.up.clone().applyQuaternion(bindToRoot);
    const targetGlobal = new Map<THREE.Bone, THREE.Quaternion>();
    targetGlobal.set(hips, rootGlobal);

    const addTarget = (
      boneName: string,
      primary: THREE.Vector3,
      secondary: THREE.Vector3
    ): void => {
      const targetFrame = frameQuaternion(primary, secondary);
      const bindFrame = this.bindFrames.get(boneName);
      if (targetFrame === null || bindFrame === undefined) return;
      targetGlobal.set(
        this.bone(boneName),
        composeFrameTarget(
          targetFrame,
          bindFrame,
          this.bindQuaternion(boneName)
        )
      );
    };

    if (points[18] !== null && points[5] !== null) {
      addTarget("LeftShoulder", points[5].clone().sub(points[18]), bodyUp);
    }
    if (points[18] !== null && points[6] !== null) {
      addTarget("RightShoulder", points[6].clone().sub(points[18]), bodyUp);
    }
    if (points[5] !== null && points[7] !== null) {
      const secondary = points[9] !== null
        ? points[9].clone().sub(points[7])
        : bodyDepth;
      addTarget("LeftArm", points[7].clone().sub(points[5]), secondary);
    }
    if (points[6] !== null && points[8] !== null) {
      const secondary = points[10] !== null
        ? points[10].clone().sub(points[8])
        : bodyDepth;
      addTarget("RightArm", points[8].clone().sub(points[6]), secondary);
    }
    if (points[7] !== null && points[9] !== null) {
      addTarget("LeftForeArm", points[9].clone().sub(points[7]), bodyDepth);
    }
    if (points[8] !== null && points[10] !== null) {
      addTarget("RightForeArm", points[10].clone().sub(points[8]), bodyDepth);
    }
    if (points[11] !== null && points[13] !== null) {
      const secondary = points[15] !== null
        ? points[15].clone().sub(points[13])
        : bodyDepth;
      addTarget("LeftUpLeg", points[13].clone().sub(points[11]), secondary);
    }
    if (points[12] !== null && points[14] !== null) {
      const secondary = points[16] !== null
        ? points[16].clone().sub(points[14])
        : bodyDepth;
      addTarget("RightUpLeg", points[14].clone().sub(points[12]), secondary);
    }
    if (points[13] !== null && points[15] !== null) {
      addTarget("LeftLeg", points[15].clone().sub(points[13]), bodyDepth);
    }
    if (points[14] !== null && points[16] !== null) {
      addTarget("RightLeg", points[16].clone().sub(points[14]), bodyDepth);
    }
    if (points[18] !== null && points[17] !== null) {
      addTarget("Neck", points[17].clone().sub(points[18]), bodyDepth);
    }
    if (points[24] !== null && points[20] !== null && points[22] !== null) {
      addTarget(
        "LeftFoot",
        midpoint(points[20], points[22]).sub(points[24]),
        bodyUp
      );
    }
    if (points[25] !== null && points[21] !== null && points[23] !== null) {
      addTarget(
        "RightFoot",
        midpoint(points[21], points[23]).sub(points[25]),
        bodyUp
      );
    }

    const desiredGlobal = new Map<THREE.Bone, THREE.Quaternion>();
    const maximumStep = THREE.MathUtils.degToRad(
      this.maximumRotationSpeedDegS * deltaTimeS
    );
    for (const bone of this.orderedBones) {
      const parentBone = isBone(bone.parent)
        ? bone.parent
        : null;
      const parentGlobal = parentBone !== null
        ? desiredGlobal.get(parentBone)?.clone()
        : bone.parent?.getWorldQuaternion(new THREE.Quaternion()).normalize();
      if (parentGlobal === undefined) {
        throw new Error(`Cannot resolve parent rotation for ${bone.name}`);
      }

      let global: THREE.Quaternion;
      let local: THREE.Quaternion;
      const rawTarget = targetGlobal.get(bone);
      if (rawTarget !== undefined) {
        const previous = this.previousGlobal.get(bone);
        global = previous === undefined || bone === hips
          ? rawTarget.clone()
          : limitedRotation(
            previous,
            rawTarget,
            maximumStep,
            this.rotationResponse
          );
        local = parentGlobal.clone().invert().multiply(global).normalize();
      } else {
        local = this.controlledBones.has(bone) && this.previousLocal.has(bone)
          ? this.previousLocal.get(bone)!.clone()
          : this.bindLocalQuaternion.get(bone)!.clone();
        global = parentGlobal.clone().multiply(local).normalize();
      }
      bone.quaternion.copy(local);
      bone.position.copy(this.bindLocalPosition.get(bone)!);
      bone.scale.copy(this.bindLocalScale.get(bone)!);
      desiredGlobal.set(bone, global);
      this.previousLocal.set(bone, local.clone());
      this.previousGlobal.set(bone, global.clone());
    }

    const rootLocalPosition = rootPosition.clone();
    if (hips.parent !== null) hips.parent.worldToLocal(rootLocalPosition);
    hips.position.copy(rootLocalPosition);
    hips.scale.copy(this.bindLocalScale.get(hips)!).multiplyScalar(this.avatarScale);
    this.root.updateMatrixWorld(true);
    return true;
  }

  resetTracking(resetScale = false): void {
    this.previousLocal.clear();
    this.previousGlobal.clear();
    this.lastTimestampMs = null;
    if (resetScale) {
      this.scaleSamples.length = 0;
      this.calibratedScale = null;
    }
    for (const bone of this.orderedBones) {
      bone.position.copy(this.bindLocalPosition.get(bone)!);
      bone.quaternion.copy(this.bindLocalQuaternion.get(bone)!);
      bone.scale.copy(this.bindLocalScale.get(bone)!);
    }
    this.root.updateMatrixWorld(true);
  }

  private bone(name: string): THREE.Bone {
    const bone = this.boneByCanonical.get(name);
    if (bone === undefined) throw new Error(`Mixamo bone not found: ${name}`);
    return bone;
  }

  private bindPosition(name: string): THREE.Vector3 {
    return this.bindWorldPosition.get(this.bone(name))!.clone();
  }

  private bindQuaternion(name: string): THREE.Quaternion {
    return this.bindWorldQuaternion.get(this.bone(name))!.clone();
  }

  private buildBindFrames(): void {
    const depth = this.bindBodyBasis.depth;
    const up = this.bindBodyBasis.up;
    const specs: Array<[string, string, THREE.Vector3]> = [
      ["LeftShoulder", "LeftArm", up],
      ["RightShoulder", "RightArm", up],
      ["LeftArm", "LeftForeArm", depth],
      ["RightArm", "RightForeArm", depth],
      ["LeftForeArm", "LeftHand", depth],
      ["RightForeArm", "RightHand", depth],
      ["LeftUpLeg", "LeftLeg", depth],
      ["RightUpLeg", "RightLeg", depth],
      ["LeftLeg", "LeftFoot", depth],
      ["RightLeg", "RightFoot", depth],
      ["Neck", "Head", depth]
    ];
    for (const [boneName, childName, secondary] of specs) {
      const frame = frameQuaternion(
        this.bindPosition(childName).sub(this.bindPosition(boneName)),
        secondary
      );
      if (frame === null) {
        throw new Error(`Mixamo bind frame is degenerate for ${boneName}`);
      }
      this.bindFrames.set(boneName, frame);
      this.controlledBones.add(this.bone(boneName));
    }
    // Mixamo's Foot node is located at the ankle, so Foot -> ToeBase points
    // both forward and downward even though the skinned sole is flat in the
    // bind pose. Halpe's heel -> toe vector describes the sole itself. Build
    // the foot bind frame from the horizontal projection of Foot -> ToeBase;
    // otherwise aligning it to a level live sole pitches the whole mesh up.
    for (const [footName, toeName] of [
      ["LeftFoot", "LeftToeBase"],
      ["RightFoot", "RightToeBase"]
    ] as const) {
      const soleForward = this.bindPosition(toeName).sub(
        this.bindPosition(footName)
      );
      soleForward.addScaledVector(up, -soleForward.dot(up));
      const frame = frameQuaternion(soleForward, up);
      if (frame === null) {
        throw new Error(`Mixamo bind sole frame is degenerate for ${footName}`);
      }
      this.bindFrames.set(footName, frame);
      this.controlledBones.add(this.bone(footName));
    }
    this.controlledBones.add(this.bone("Hips"));
  }

  private liveBodyBasis(
    points: Array<THREE.Vector3 | null>,
    root: THREE.Vector3
  ): BodyBasis | null {
    if (points[18] === null) return null;
    const lateral = new THREE.Vector3();
    if (points[11] !== null && points[12] !== null) {
      const hips = unit(points[11].clone().sub(points[12]));
      if (hips !== null) lateral.add(hips);
    }
    if (points[5] !== null && points[6] !== null) {
      const shoulders = unit(points[5].clone().sub(points[6]));
      if (shoulders !== null) lateral.add(shoulders);
    }
    return bodyBasis(points[18].clone().sub(root), lateral);
  }

  private updateScale(points: Array<THREE.Vector3 | null>): void {
    if (!this.autoScale || this.calibratedScale !== null) return;
    if (points[17] === null || points[15] === null || points[16] === null) return;
    const observed = points[17].distanceTo(midpoint(points[15], points[16]));
    if (!(observed >= 1.2 && observed <= 2.2)) return;
    this.scaleSamples.push(observed / this.bindStature);
    if (this.scaleSamples.length < this.minimumScaleSamples) return;
    const sorted = [...this.scaleSamples].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    if (median >= this.minimumScale && median <= this.maximumScale) {
      this.calibratedScale = median;
    }
  }
}
