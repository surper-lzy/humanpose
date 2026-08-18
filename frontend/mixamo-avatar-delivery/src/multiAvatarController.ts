import * as THREE from "three";

import {
  AvatarController,
  type AvatarControllerOptions,
  type DisplayMode
} from "./avatarController.js";
import type {
  StickmanPayloadV1,
  StickmenPayloadV1,
  TrackedStickmanV1
} from "./types.js";


export type AvatarDisplayLimit = number | "all";

export interface MultiAvatarControllerOptions {
  displayLimit?: AvatarDisplayLimit;
  removedTrackRetentionMs?: number;
  avatar?: AvatarControllerOptions;
}

interface TrackController {
  group: THREE.Group;
  controller: AvatarController;
  lastArrivalMs: number;
}

/** Own one AvatarController per track and expose a stable display limit. */
export class MultiAvatarController {
  private readonly parent: THREE.Object3D;
  private readonly options: AvatarControllerOptions;
  private readonly removedTrackRetentionMs: number;
  private readonly tracks = new Map<number, TrackController>();
  private readonly selectedTrackIds = new Set<number>();
  private limit: AvatarDisplayLimit;
  private mode: DisplayMode = "stickman";
  private avatarId: string | null = null;
  private streamId: string | null = null;
  private latest: StickmenPayloadV1 | null = null;

  constructor(
    parent: THREE.Object3D,
    options: MultiAvatarControllerOptions = {}
  ) {
    this.parent = parent;
    this.options = options.avatar ?? {};
    this.removedTrackRetentionMs = options.removedTrackRetentionMs ?? 1_500;
    if (!(Number.isFinite(this.removedTrackRetentionMs) &&
      this.removedTrackRetentionMs >= 0)) {
      throw new Error("removedTrackRetentionMs must be non-negative");
    }
    this.limit = options.displayLimit ?? "all";
    this.assertDisplayLimit(this.limit);
  }

  get displayLimit(): AvatarDisplayLimit {
    return this.limit;
  }

  get visibleTrackIds(): readonly number[] {
    return [...this.selectedTrackIds].sort((first, second) => first - second);
  }

  setDisplayLimit(limit: AvatarDisplayLimit): void {
    this.assertDisplayLimit(limit);
    this.limit = limit;
    if (this.latest !== null) this.updateSelection(this.latest.persons);
  }

  async setDisplayMode(mode: DisplayMode): Promise<void> {
    this.mode = mode;
    await Promise.all(
      [...this.tracks.values()].map(({ controller }) =>
        controller.setDisplayMode(mode)
      )
    );
  }

  async selectAvatar(id: string): Promise<void> {
    this.avatarId = id;
    await Promise.all(
      [...this.tracks.values()].map(({ controller }) =>
        controller.selectAvatar(id)
      )
    );
  }

  acceptPoses(
    payload: StickmenPayloadV1,
    arrivalMs = performance.now()
  ): void {
    if (this.streamId !== null && this.streamId !== payload.stream_id) {
      this.clearTracks();
    }
    this.streamId = payload.stream_id;
    this.latest = payload;
    const presentIds = new Set<number>();
    for (const person of payload.persons) {
      presentIds.add(person.track_id);
      const track = this.ensureTrack(person.track_id);
      track.lastArrivalMs = arrivalMs;
      track.controller.acceptPose(
        this.asSinglePayload(payload, person),
        arrivalMs
      );
    }
    this.removeExpiredTracks(presentIds, arrivalMs);
    this.updateSelection(payload.persons);
  }

  /** Call from requestAnimationFrame, like the single-person controller. */
  tick(nowMs = performance.now()): void {
    for (const track of this.tracks.values()) track.controller.tick(nowMs);
    const presentIds = new Set(
      this.latest?.persons.map((person) => person.track_id) ?? []
    );
    this.removeExpiredTracks(presentIds, nowMs);
  }

  dispose(): void {
    this.clearTracks();
    this.latest = null;
    this.streamId = null;
  }

  private ensureTrack(trackId: number): TrackController {
    const existing = this.tracks.get(trackId);
    if (existing !== undefined) return existing;
    const group = new THREE.Group();
    group.name = `TrackedAvatar:${trackId}`;
    group.visible = false;
    this.parent.add(group);
    const controller = new AvatarController(group, this.options);
    const track = { group, controller, lastArrivalMs: 0 };
    this.tracks.set(trackId, track);
    void controller.setDisplayMode(this.mode);
    if (this.avatarId !== null) void controller.selectAvatar(this.avatarId);
    return track;
  }

  private updateSelection(persons: readonly TrackedStickmanV1[]): void {
    const available = new Set(persons.map((person) => person.track_id));
    const maximum = this.limit === "all" ? persons.length : this.limit;
    const next: number[] = [...this.selectedTrackIds]
      .filter((trackId) => available.has(trackId))
      .slice(0, maximum);
    const alreadySelected = new Set(next);
    const candidates = [...persons].sort((first, second) => {
      if (first.observed_in_frame !== second.observed_in_frame) {
        return first.observed_in_frame ? -1 : 1;
      }
      return first.track_id - second.track_id;
    });
    for (const person of candidates) {
      if (next.length >= maximum) break;
      if (alreadySelected.has(person.track_id)) continue;
      next.push(person.track_id);
      alreadySelected.add(person.track_id);
    }
    this.selectedTrackIds.clear();
    next.forEach((trackId) => this.selectedTrackIds.add(trackId));
    for (const [trackId, track] of this.tracks) {
      track.group.visible = this.selectedTrackIds.has(trackId);
    }
  }

  private removeExpiredTracks(
    presentIds: ReadonlySet<number>,
    nowMs: number
  ): void {
    for (const [trackId, track] of this.tracks) {
      if (presentIds.has(trackId)) continue;
      if (nowMs - track.lastArrivalMs <= this.removedTrackRetentionMs) continue;
      this.disposeTrack(trackId, track);
    }
  }

  private disposeTrack(trackId: number, track: TrackController): void {
    track.controller.dispose();
    track.group.removeFromParent();
    this.tracks.delete(trackId);
    this.selectedTrackIds.delete(trackId);
  }

  private clearTracks(): void {
    for (const [trackId, track] of this.tracks) {
      this.disposeTrack(trackId, track);
    }
    this.selectedTrackIds.clear();
  }

  private asSinglePayload(
    payload: StickmenPayloadV1,
    person: TrackedStickmanV1
  ): StickmanPayloadV1 {
    return {
      schema_version: 1,
      keypoint_format: "halpe26",
      coordinate_system: payload.coordinate_system,
      source_id: `${payload.source_id}:${person.track_id}`,
      frame_number: payload.frame_number,
      timestamp_ms: payload.timestamp_ms,
      status: person.status,
      joints: person.joints
    };
  }

  private assertDisplayLimit(limit: AvatarDisplayLimit): void {
    if (limit === "all") return;
    if (!Number.isInteger(limit) || limit <= 0) {
      throw new Error("displayLimit must be a positive integer or 'all'");
    }
  }
}
