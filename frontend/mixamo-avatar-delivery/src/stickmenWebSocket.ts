import {
  isStickmenEventV1,
  type StickmenPayloadV1
} from "./types.js";


export interface StickmenWebSocketOptions {
  url: string;
  topic: string;
  clientId?: string;
  reconnectInitialMs?: number;
  reconnectMaximumMs?: number;
  onPoses: (payload: StickmenPayloadV1) => void;
  onConnectionChange?: (connected: boolean) => void;
  onError?: (error: unknown) => void;
}

function messageId(prefix: string): string {
  const randomId = globalThis.crypto?.randomUUID?.();
  return randomId ?? `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/** WebSocket client for the additive avatar.stickmen.updated event. */
export class StickmenWebSocketClient {
  private readonly options: Required<Pick<
    StickmenWebSocketOptions,
    "clientId" | "reconnectInitialMs" | "reconnectMaximumMs"
  >> & StickmenWebSocketOptions;

  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelayMs: number;
  private stopped = true;
  private connected = false;
  private latest: StickmenPayloadV1 | null = null;

  constructor(options: StickmenWebSocketOptions) {
    this.options = {
      clientId: "mixamo-multi-avatar-ui",
      reconnectInitialMs: 500,
      reconnectMaximumMs: 10_000,
      ...options
    };
    this.reconnectDelayMs = this.options.reconnectInitialMs;
  }

  get isConnected(): boolean {
    return this.connected;
  }

  get latestPoses(): StickmenPayloadV1 | null {
    return this.latest;
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.socket?.close();
    this.socket = null;
    this.setConnected(false);
  }

  private connectionUrl(): string {
    const url = new URL(this.options.url, globalThis.location?.href);
    url.searchParams.set("client_type", "browser");
    url.searchParams.set("client_id", this.options.clientId);
    url.searchParams.set("topics", this.options.topic);
    return url.toString();
  }

  private connect(): void {
    if (this.stopped) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(this.connectionUrl());
    } catch (error) {
      this.options.onError?.(error);
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;

    socket.addEventListener("open", () => {
      if (socket !== this.socket) return;
      this.reconnectDelayMs = this.options.reconnectInitialMs;
      this.setConnected(true);
      socket.send(JSON.stringify({
        type: "hello",
        message_id: messageId("frontend-multi-hello"),
        client_type: "browser",
        client_id: this.options.clientId,
        topics: [this.options.topic]
      }));
    });

    socket.addEventListener("message", (event) => {
      if (socket !== this.socket || typeof event.data !== "string") return;
      let message: unknown;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (!isStickmenEventV1(message, this.options.topic)) return;
      this.latest = message.payload;
      this.options.onPoses(message.payload);
    });

    socket.addEventListener("error", (event) => {
      this.options.onError?.(event);
      socket.close();
    });

    socket.addEventListener("close", () => {
      if (socket !== this.socket) return;
      this.socket = null;
      this.setConnected(false);
      this.scheduleReconnect();
    });
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer !== null) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, this.reconnectDelayMs);
    this.reconnectDelayMs = Math.min(
      this.reconnectDelayMs * 2,
      this.options.reconnectMaximumMs
    );
  }

  private setConnected(connected: boolean): void {
    if (this.connected === connected) return;
    this.connected = connected;
    this.options.onConnectionChange?.(connected);
  }
}
