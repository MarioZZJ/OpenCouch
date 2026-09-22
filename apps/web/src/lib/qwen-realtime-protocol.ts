/** Qwen WebRTC wire compatibility. This module has no browser/SDK dependencies. */
type JsonRecord = Record<string, unknown>;
const object = (value: unknown): JsonRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonRecord : {};

export function normalizeQwenRealtimeEvent(event: JsonRecord): JsonRecord {
  const aliases: Record<string, string> = {
    "response.text.delta": "response.output_audio_transcript.delta",
    "response.text.done": "response.output_audio_transcript.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
    "response.audio.done": "response.output_audio.done",
  };
  const type = aliases[String(event.type)] ?? event.type;
  if (type === "response.output_audio_transcript.done") {
    return { ...event, type, transcript: event.transcript ?? event.text ?? "" };
  }
  // Qwen's ASR `text` + `stash` preview is a snapshot, not an append delta.
  // The authoritative completed transcript drives tools, safety and persistence.
  if (type === "conversation.item.input_audio_transcription.delta" && !event.delta) {
    return { ...event, type, delta: "" };
  }
  return { ...event, type };
}

export function qwenClientEvent(event: JsonRecord): JsonRecord {
  if (event.type === "response.create") {
    // Qwen documents type/event_id, not OpenAI response.metadata/instructions.
    return { type: "response.create", ...(event.event_id ? { event_id: event.event_id } : {}) };
  }
  return event;
}

/** Correlate tool follow-up responses only inside the same uninterrupted turn. */
export class QwenFollowUpResponses {
  private pending: string[] = [];
  generation = 0;

  expect(id: string): void { this.pending.push(id); }
  forget(id: string): void { this.pending = this.pending.filter(value => value !== id); }
  interrupt(): string[] {
    this.generation += 1;
    const cancelled = this.pending;
    this.pending = [];
    return cancelled;
  }
  responseCreated(event: JsonRecord): JsonRecord {
    if (event.type !== "response.created" || !this.pending.length) return event;
    const id = this.pending.shift();
    const response = object(event.response);
    return { ...event, response: { ...response, metadata: {
      ...object(response.metadata), opencouch_response_request_id: id,
    } } };
  }
}
