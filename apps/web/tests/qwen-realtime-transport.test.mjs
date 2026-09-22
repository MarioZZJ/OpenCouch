/** Exercise the actual session connector with mocked WebRTC and HTTP transports. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

async function loadConnector() {
  const directory = await mkdtemp(join(tmpdir(), "opencouch-voice-test-"));
  const modules = ["api", "realtime-voice-session", "realtime-voice-events", "realtime-voice-turn-record",
    "realtime-voice-finalization", "realtime-voice-tool-flow", "qwen-realtime-protocol"];
  for (const name of modules) {
    const source = await readFile(new URL(`../src/lib/${name}.ts`, import.meta.url), "utf8");
    const compiled = ts.transpileModule(source, { compilerOptions: {
      target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022,
    } }).outputText.replace(/from "\.\/([\w-]+)"/g, 'from "./$1.mjs"');
    await writeFile(join(directory, `${name}.mjs`), compiled);
  }
  const module = await import(pathToFileURL(join(directory, "realtime-voice-session.mjs")).href);
  return { ...module, cleanup: () => rm(directory, { recursive: true, force: true }) };
}

class Channel extends EventTarget {
  constructor(label) { super(); this.label = label; this.readyState = "open"; this.sent = []; }
  send(value) { this.sent.push(JSON.parse(value)); }
  close() { this.readyState = "closed"; }
  receive(value) { this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify(value) })); }
}

test("Qwen sends policy on server txt channel, gates microphone, and preserves safety/turn APIs", async (t) => {
  const connector = await loadConnector();
  t.after(connector.cleanup);
  const track = { enabled: true, stop() { this.stopped = true; } };
  const audio = { muted: false, srcObject: null, pause() {} };
  const sentRequests = [], status = [], transcripts = [], errors = [];
  const oldFetch = globalThis.fetch;
  const oldPeer = globalThis.RTCPeerConnection;
  const oldNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  t.after(() => {
    globalThis.fetch = oldFetch; globalThis.RTCPeerConnection = oldPeer;
    if (oldNavigator) Object.defineProperty(globalThis, "navigator", oldNavigator);
    else delete globalThis.navigator;
  });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: {
    mediaDevices: { getUserMedia: async () => ({ getTracks: () => [track], getAudioTracks: () => [track] }) },
  } });
  let peer;
  globalThis.RTCPeerConnection = class {
    constructor() { peer = this; this.iceGatheringState = "complete"; }
    addTrack() {}
    createDataChannel(label) { this.local = new Channel(label); return this.local; }
    async createOffer() { return { type: "offer", sdp: "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n" }; }
    async setLocalDescription(offer) { this.localDescription = offer; }
    async setRemoteDescription(answer) {
      this.remoteDescription = answer;
      assert.equal(track.enabled, false);
      assert.equal(audio.muted, true);
      this.local.dispatchEvent(new Event("open"));
      this.remote = new Channel("txt");
      this.ondatachannel({ channel: this.remote });
    }
    close() { this.closed = true; }
  };
  globalThis.fetch = async (url, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : {};
    sentRequests.push({ url: String(url), body, headers: options.headers });
    assert.ok(!String(url).includes("api.openai.com"));
    assert.equal(options.headers?.Authorization, undefined);
    if (String(url).endsWith("/realtime/session")) return Response.json({
      provider: "qwen", client_secret: "a".repeat(43), thread_id: "t", user_id: null,
      memory_mode: "incognito", message_count: 0,
      session_config: { model: "qwen3.5-omni-flash-realtime", instructions: "policy", tools: [] },
    });
    if (String(url).endsWith("/qwen/sdp")) return Response.json({ sdp: "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n" });
    if (String(url).endsWith("/safety/check")) return Response.json({
      client_turn_id: body.client_turn_id, status: "assessed", reason: "test", action: "continue", risk_level: 0,
    });
    if (String(url).endsWith("/realtime/turn")) return Response.json({ recorded: true, thread_id: "t", message_count: 2 });
    return Response.json({});
  };
  const handle = await connector.connectRealtimeVoiceSession({
    threadId: "t", memoryMode: "incognito", audioElement: audio,
    onStatus: value => status.push(value), onTranscript: value => transcripts.push(value),
    onError: value => errors.push(value),
  });
  t.after(() => handle.disconnect({ finalize: false }));
  assert.equal(peer.local.sent.length, 0);
  assert.equal(peer.remote.sent[0].type, "session.update");
  assert.equal(track.enabled, false);
  assert.ok(!status.includes("connected"));
  peer.remote.receive({ type: "session.updated", event_id: "ready" });
  assert.equal(track.enabled, true);
  assert.equal(audio.muted, false);
  assert.ok(status.includes("connected"));
  peer.remote.receive({ type: "input_audio_buffer.committed", item_id: "user-1" });
  peer.remote.receive({ type: "conversation.item.input_audio_transcription.completed", item_id: "user-1", transcript: "今天有点累" });
  peer.remote.receive({ type: "response.created", response: { id: "response-1" } });
  peer.remote.receive({ type: "response.text.done", item_id: "assistant-1", response_id: "response-1", text: "听起来你需要休息" });
  peer.remote.receive({ type: "response.done", response: { id: "response-1", output: [] } });
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.ok(transcripts.some(value => value.role === "assistant" && value.text === "听起来你需要休息"));
  assert.ok(sentRequests.some(value => value.url.endsWith("/safety/check")));
  assert.ok(sentRequests.some(value => value.url.endsWith("/realtime/turn") && value.body.assistant_text === "听起来你需要休息"));
  assert.equal(errors.length, 0);
  await handle.disconnect({ finalize: false });
  assert.equal(track.stopped, true);
});
