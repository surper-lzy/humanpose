import * as THREE from "three";

import { applicationJointsToThree } from "./coordinates.js";
import { HALPE26_LINKS } from "./halpe26.js";
import type { StickmanPayloadV1 } from "./types.js";


export interface StickmanRendererOptions {
  jointRadius?: number;
  boneRadius?: number;
  jointColor?: THREE.ColorRepresentation;
  boneColor?: THREE.ColorRepresentation;
}

const Y_AXIS = new THREE.Vector3(0, 1, 0);

export class StickmanRenderer {
  readonly group = new THREE.Group();

  private readonly jointGeometry: THREE.SphereGeometry;
  private readonly boneGeometry: THREE.CylinderGeometry;
  private readonly jointMaterial: THREE.MeshBasicMaterial;
  private readonly boneMaterial: THREE.MeshBasicMaterial;
  private readonly joints: THREE.Mesh[];
  private readonly bones: THREE.Mesh[];

  constructor(
    parent: THREE.Object3D,
    options: StickmanRendererOptions = {}
  ) {
    this.group.name = "Halpe26Stickman";
    parent.add(this.group);
    this.jointGeometry = new THREE.SphereGeometry(
      options.jointRadius ?? 0.025,
      10,
      8
    );
    this.boneGeometry = new THREE.CylinderGeometry(
      options.boneRadius ?? 0.012,
      options.boneRadius ?? 0.012,
      1,
      8
    );
    this.jointMaterial = new THREE.MeshBasicMaterial({
      color: options.jointColor ?? 0x101010
    });
    this.boneMaterial = new THREE.MeshBasicMaterial({
      color: options.boneColor ?? 0x101010
    });
    this.joints = Array.from({ length: 26 }, () => {
      const mesh = new THREE.Mesh(this.jointGeometry, this.jointMaterial);
      mesh.visible = false;
      this.group.add(mesh);
      return mesh;
    });
    this.bones = HALPE26_LINKS.map(() => {
      const mesh = new THREE.Mesh(this.boneGeometry, this.boneMaterial);
      mesh.visible = false;
      this.group.add(mesh);
      return mesh;
    });
    this.group.visible = false;
  }

  update(payload: StickmanPayloadV1): void {
    const points = applicationJointsToThree(payload.joints);
    for (let index = 0; index < this.joints.length; index += 1) {
      const point = points[index];
      const mesh = this.joints[index];
      mesh.visible = point !== null;
      if (point !== null) mesh.position.copy(point);
    }

    for (let index = 0; index < HALPE26_LINKS.length; index += 1) {
      const [start, end] = HALPE26_LINKS[index];
      const a = points[start];
      const b = points[end];
      const mesh = this.bones[index];
      mesh.visible = a !== null && b !== null;
      if (a === null || b === null) continue;
      const direction = b.clone().sub(a);
      const length = direction.length();
      if (length <= 1e-6) {
        mesh.visible = false;
        continue;
      }
      mesh.position.copy(a).add(b).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(Y_AXIS, direction.multiplyScalar(1 / length));
      mesh.scale.set(1, length, 1);
    }
    this.group.visible = true;
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  dispose(): void {
    this.group.removeFromParent();
    this.jointGeometry.dispose();
    this.boneGeometry.dispose();
    this.jointMaterial.dispose();
    this.boneMaterial.dispose();
  }
}
