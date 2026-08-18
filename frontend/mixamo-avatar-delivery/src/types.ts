export type Joint3 = [number, number, number] | null;

export interface ApplicationCoordinateSystem {
  name: "application";
  handedness: "right";
  unit: "meter";
  up_axis: "+z";
  ground_z_m: number;
}

export interface StickmanPayloadV1 {
  schema_version: 1;
  keypoint_format: "halpe26";
  coordinate_system: ApplicationCoordinateSystem;
  source_id: string;
  frame_number: number;
  timestamp_ms: number;
  status: string;
  joints: Joint3[];
}

export interface StickmanEventV1 {
  type: "event";
  event: "avatar.stickman.updated";
  topics: string[];
  payload: StickmanPayloadV1;
  message_id?: string;
  timestamp?: string;
  source_type?: string;
  source_id?: string;
}

export type IdentityMethod = "geometry" | "shadow";

export interface TrackedStickmanV1 {
  track_id: number;
  status: string;
  observed_in_frame: boolean;
  joints: Joint3[];
}

export interface StickmenPayloadV1 {
  schema_version: 1;
  keypoint_format: "halpe26";
  coordinate_system: ApplicationCoordinateSystem;
  source_id: string;
  stream_id: string;
  frame_number: number;
  timestamp_ms: number;
  status: string;
  identity_method: IdentityMethod;
  identity_fallback: boolean;
  detected_person_count: number;
  published_person_count: number;
  persons: TrackedStickmanV1[];
}

export interface StickmenEventV1 {
  type: "event";
  event: "avatar.stickmen.updated";
  topics: string[];
  payload: StickmenPayloadV1;
  message_id?: string;
  timestamp?: string;
  source_type?: string;
  source_id?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}

export function isJoint3(value: unknown): value is Joint3 {
  return value === null || (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every(Number.isFinite)
  );
}

export function isStickmanPayloadV1(
  value: unknown
): value is StickmanPayloadV1 {
  if (!isRecord(value) || !isRecord(value.coordinate_system)) return false;
  const coordinate = value.coordinate_system;
  return value.schema_version === 1 &&
    value.keypoint_format === "halpe26" &&
    coordinate.name === "application" &&
    coordinate.handedness === "right" &&
    coordinate.unit === "meter" &&
    coordinate.up_axis === "+z" &&
    Number.isFinite(coordinate.ground_z_m) &&
    typeof value.source_id === "string" &&
    Number.isInteger(value.frame_number) &&
    Number.isFinite(value.timestamp_ms) &&
    typeof value.status === "string" &&
    Array.isArray(value.joints) &&
    value.joints.length === 26 &&
    value.joints.every(isJoint3);
}

export function isStickmanEventV1(
  value: unknown,
  topic: string
): value is StickmanEventV1 {
  return isRecord(value) &&
    value.type === "event" &&
    value.event === "avatar.stickman.updated" &&
    Array.isArray(value.topics) &&
    value.topics.includes(topic) &&
    isStickmanPayloadV1(value.payload);
}

export function isTrackedStickmanV1(
  value: unknown
): value is TrackedStickmanV1 {
  return isRecord(value) &&
    Number.isInteger(value.track_id) &&
    Number(value.track_id) > 0 &&
    typeof value.status === "string" &&
    typeof value.observed_in_frame === "boolean" &&
    Array.isArray(value.joints) &&
    value.joints.length === 26 &&
    value.joints.every(isJoint3);
}

export function isStickmenPayloadV1(
  value: unknown
): value is StickmenPayloadV1 {
  if (!isRecord(value) || !isRecord(value.coordinate_system)) return false;
  const coordinate = value.coordinate_system;
  const persons = value.persons;
  if (!(value.schema_version === 1 &&
    value.keypoint_format === "halpe26" &&
    coordinate.name === "application" &&
    coordinate.handedness === "right" &&
    coordinate.unit === "meter" &&
    coordinate.up_axis === "+z" &&
    Number.isFinite(coordinate.ground_z_m) &&
    typeof value.source_id === "string" &&
    typeof value.stream_id === "string" &&
    value.stream_id.length > 0 &&
    Number.isInteger(value.frame_number) &&
    Number.isFinite(value.timestamp_ms) &&
    typeof value.status === "string" &&
    (value.identity_method === "geometry" ||
      value.identity_method === "shadow") &&
    typeof value.identity_fallback === "boolean" &&
    Number.isInteger(value.detected_person_count) &&
    Number(value.detected_person_count) >= 0 &&
    Number.isInteger(value.published_person_count) &&
    Number(value.published_person_count) >= 0 &&
    Array.isArray(persons) &&
    persons.every(isTrackedStickmanV1) &&
    value.published_person_count === persons.length)) {
    return false;
  }
  const trackedPersons = persons as TrackedStickmanV1[];
  const ids = trackedPersons.map((person) => person.track_id);
  return ids.length === new Set(ids).size;
}

export function isStickmenEventV1(
  value: unknown,
  topic: string
): value is StickmenEventV1 {
  return isRecord(value) &&
    value.type === "event" &&
    value.event === "avatar.stickmen.updated" &&
    Array.isArray(value.topics) &&
    value.topics.includes(topic) &&
    isStickmenPayloadV1(value.payload);
}
