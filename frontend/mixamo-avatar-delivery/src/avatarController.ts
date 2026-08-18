import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import {
  AVATARS,
  avatarDefinition,
  type AvatarDefinition
} from "./avatarRegistry.js";
import {
  MixamoRetargeter,
  type MixamoRetargeterOptions
} from "./mixamoRetargeter.js";
import {
  StickmanRenderer,
  type StickmanRendererOptions
} from "./stickmanRenderer.js";
import type { StickmanPayloadV1 } from "./types.js";


export type DisplayMode = "stickman" | "mixamo";

export interface AvatarControllerOptions {
  avatars?: readonly AvatarDefinition[];
  defaultAvatarId?: string;
  staleAfterMs?: number;
  stickman?: StickmanRendererOptions;
  retargeter?: Omit<MixamoRetargeterOptions, "boneAliases" | "scaleMultiplier">;
  loader?: GLTFLoader;
}

interface LoadedAvatar {
  definition: AvatarDefinition;
  scene: THREE.Group;
  retargeter: MixamoRetargeter;
}

const RESET_STATUSES = new Set([
  "partial_person_out_of_frame",
  "awaiting_full_reentry"
]);

function disposeMaterial(material: THREE.Material): void {
  for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture) value.dispose();
  }
  material.dispose();
}

export class AvatarController {
  readonly stickman: StickmanRenderer;

  private readonly parent: THREE.Object3D;
  private readonly avatars: readonly AvatarDefinition[];
  private readonly staleAfterMs: number;
  private readonly loader: GLTFLoader;
  private readonly retargeterOptions: AvatarControllerOptions["retargeter"];
  private readonly loaded = new Map<string, LoadedAvatar>();
  private readonly loading = new Map<string, Promise<LoadedAvatar>>();
  private mode: DisplayMode = "stickman";
  private selectedAvatarId: string;
  private latestPose: StickmanPayloadV1 | null = null;
  private latestArrivalMs = 0;
  private staleHidden = false;

  constructor(parent: THREE.Object3D, options: AvatarControllerOptions = {}) {
    this.parent = parent;
    this.avatars = options.avatars ?? AVATARS;
    if (this.avatars.length === 0) throw new Error("At least one avatar is required");
    const ids = new Set(this.avatars.map((avatar) => avatar.id));
    if (ids.size !== this.avatars.length) throw new Error("Avatar ids must be unique");
    this.selectedAvatarId = options.defaultAvatarId ?? this.avatars[0].id;
    avatarDefinition(this.selectedAvatarId, this.avatars);
    this.staleAfterMs = options.staleAfterMs ?? 500;
    this.loader = options.loader ?? new GLTFLoader();
    this.retargeterOptions = options.retargeter;
    this.stickman = new StickmanRenderer(parent, options.stickman);
  }

  get displayMode(): DisplayMode {
    return this.mode;
  }

  get avatarId(): string {
    return this.selectedAvatarId;
  }

  async setDisplayMode(mode: DisplayMode): Promise<void> {
    this.mode = mode;
    if (mode === "stickman") {
      this.hideAvatars();
      if (
        this.latestPose !== null &&
        !this.isStale() &&
        !RESET_STATUSES.has(this.latestPose.status)
      ) {
        this.stickman.update(this.latestPose);
      }
      return;
    }

    this.stickman.setVisible(false);
    const avatar = await this.ensureAvatar(this.selectedAvatarId);
    if (this.mode !== "mixamo" || avatar.definition.id !== this.selectedAvatarId) {
      return;
    }
    avatar.scene.visible = true;
    this.applyLatestToAvatar(avatar);
  }

  async selectAvatar(id: string): Promise<void> {
    avatarDefinition(id, this.avatars);
    this.selectedAvatarId = id;
    this.hideAvatars();
    const avatar = await this.ensureAvatar(id);
    if (this.mode !== "mixamo" || this.selectedAvatarId !== id) return;
    avatar.retargeter.resetTracking(false);
    avatar.scene.visible = true;
    this.applyLatestToAvatar(avatar);
  }

  /** Consume each accepted WebSocket pose exactly once. */
  acceptPose(payload: StickmanPayloadV1, arrivalMs = performance.now()): void {
    this.latestPose = payload;
    this.latestArrivalMs = arrivalMs;
    this.staleHidden = false;
    if (RESET_STATUSES.has(payload.status)) {
      this.hideAll();
      for (const avatar of this.loaded.values()) {
        avatar.retargeter.resetTracking(true);
      }
      return;
    }

    if (this.mode === "stickman") {
      this.hideAvatars();
      this.stickman.update(payload);
      return;
    }

    this.stickman.setVisible(false);
    const avatar = this.loaded.get(this.selectedAvatarId);
    if (avatar === undefined) return;
    avatar.scene.visible = avatar.retargeter.solve(payload);
  }

  /** Call from requestAnimationFrame to enforce the shared stale timeout. */
  tick(nowMs = performance.now()): void {
    if (!this.isStale(nowMs) || this.staleHidden) return;
    this.staleHidden = true;
    this.hideAll();
    this.loaded.get(this.selectedAvatarId)?.retargeter.resetTracking(false);
  }

  hideAll(): void {
    this.stickman.setVisible(false);
    this.hideAvatars();
  }

  dispose(): void {
    this.stickman.dispose();
    for (const avatar of this.loaded.values()) {
      avatar.scene.traverse((object) => {
        const mesh = object as THREE.Mesh;
        if (!mesh.isMesh) return;
        mesh.geometry?.dispose();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach(disposeMaterial);
        } else if (mesh.material !== undefined) {
          disposeMaterial(mesh.material);
        }
      });
      avatar.scene.removeFromParent();
    }
    this.loaded.clear();
    this.loading.clear();
  }

  private async ensureAvatar(id: string): Promise<LoadedAvatar> {
    const existing = this.loaded.get(id);
    if (existing !== undefined) return existing;
    const inFlight = this.loading.get(id);
    if (inFlight !== undefined) return inFlight;

    const definition = avatarDefinition(id, this.avatars);
    const promise = this.loader.loadAsync(definition.modelUrl).then((gltf) => {
      const scene = gltf.scene;
      scene.name = `Avatar:${definition.id}`;
      if (definition.rotationOffsetEuler !== undefined) {
        scene.rotation.set(...definition.rotationOffsetEuler);
      }
      scene.visible = false;
      scene.traverse((object) => {
        const mesh = object as THREE.SkinnedMesh;
        if (mesh.isSkinnedMesh) mesh.frustumCulled = false;
      });
      this.parent.add(scene);
      scene.updateMatrixWorld(true);
      const retargeter = new MixamoRetargeter(scene, {
        ...this.retargeterOptions,
        boneAliases: definition.boneAliases,
        scaleMultiplier: definition.scaleMultiplier
      });
      const loaded = { definition, scene, retargeter };
      this.loaded.set(id, loaded);
      this.loading.delete(id);
      return loaded;
    }).catch((error: unknown) => {
      this.loading.delete(id);
      throw error;
    });
    this.loading.set(id, promise);
    return promise;
  }

  private applyLatestToAvatar(avatar: LoadedAvatar): void {
    if (this.latestPose === null || this.isStale()) {
      avatar.scene.visible = false;
      return;
    }
    avatar.scene.visible = avatar.retargeter.solve(this.latestPose);
  }

  private hideAvatars(): void {
    for (const avatar of this.loaded.values()) avatar.scene.visible = false;
  }

  private isStale(nowMs = performance.now()): boolean {
    return this.latestPose === null || nowMs - this.latestArrivalMs > this.staleAfterMs;
  }
}
