import assert from "node:assert/strict";
import { test } from "node:test";
import { normalizeQwenRealtimeEvent, qwenClientEvent, QwenFollowUpResponses } from "../src/lib/qwen-realtime-protocol.ts";
import { parseRealtimeServerEvent, buildResponseCreateEvent } from "../src/lib/realtime-voice-events.ts";

for (const type of ["response.text.done", "response.audio_transcript.done"]) {
  test(`normalizes ${type} into the existing transcript and persistence path`, () => {
    const parsed = parseRealtimeServerEvent(normalizeQwenRealtimeEvent({
      type, text: "我听到了", response_id: "r", item_id: "i",
    }));
    assert.deepEqual(parsed.transcript, { role: "assistant", responseId: "r", itemId: "i", text: "我听到了", final: true });
  });
}
test("snapshot ASR previews are not appended as duplicated user speech", () => {
  const parsed = parseRealtimeServerEvent(normalizeQwenRealtimeEvent({
    type: "conversation.item.input_audio_transcription.delta", text: "你好", stash: "吗",
  }));
  assert.equal(parsed.transcript.text, "");
});
test("Qwen response.create omits unsupported OpenAI metadata", () => {
  assert.deepEqual(qwenClientEvent(buildResponseCreateEvent(null, "req-1")), {
    type: "response.create", event_id: "req-1",
  });
});
test("correlates tool follow-ups locally without mutating provider events", () => {
  const tracker = new QwenFollowUpResponses();
  tracker.expect("req-1");
  const raw = { type: "response.created", response: { id: "r" } };
  const parsed = parseRealtimeServerEvent(tracker.responseCreated(raw));
  assert.equal(parsed.responseRequestId, "req-1");
  assert.equal(raw.response.metadata, undefined);
});
test("barge-in invalidates pending follow-ups and their generation", () => {
  const tracker = new QwenFollowUpResponses();
  const initial = tracker.generation;
  tracker.expect("req-1");
  assert.deepEqual(tracker.interrupt(), ["req-1"]);
  assert.notEqual(tracker.generation, initial);
  assert.equal(parseRealtimeServerEvent(tracker.responseCreated({ type: "response.created", response: { id: "r2" } })).responseRequestId, undefined);
});
