export interface AvatarDefinition {
  id: string;
  label: string;
  modelUrl: string;
  rigType: "mixamo";
  /** Canonical Mixamo bone name -> actual GLB bone name. */
  boneAliases?: Record<string, string>;
  /** Optional final correction after automatic person/model scale fitting. */
  scaleMultiplier?: number;
  /** Optional GLB root correction in radians, applied before rig inspection. */
  rotationOffsetEuler?: [number, number, number];
}

export const AVATARS: readonly AvatarDefinition[] = [
  {
    id: "character-a",
    label: "Character A",
    modelUrl: "/avatars/character-a.glb",
    rigType: "mixamo"
  }
];

export function avatarDefinition(
  id: string,
  avatars: readonly AvatarDefinition[] = AVATARS
): AvatarDefinition {
  const definition = avatars.find((candidate) => candidate.id === id);
  if (definition === undefined) throw new Error(`Unknown avatar: ${id}`);
  return definition;
}
