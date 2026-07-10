export type TrainingSample = {
  encodedState: Float32Array;
  legalMask: Uint8Array;
  actionIndex: number;
  reward?: number;
  policyTarget?: Float32Array;
  priority?: number;
  source?: "teacher" | "selfplay" | "search" | "alphazero";
  terminal?: boolean;
  wallMove?: boolean;
  loopPenalty?: number;
  pathAdvantage?: number;
  pathProgress?: number;
};

export class ReplayBuffer {
  private samples: TrainingSample[] = [];

  constructor(private capacity = 50000) {}

  add(sample: TrainingSample): void {
    this.samples.push(sample);
    if (this.samples.length > this.capacity) this.samples.shift();
  }

  addMany(samples: TrainingSample[]): void {
    for (const sample of samples) this.add(sample);
  }

  size(): number {
    return this.samples.length;
  }

  sample(batchSize: number): TrainingSample[] {
    const count = Math.min(batchSize, this.samples.length);
    const batch: TrainingSample[] = [];
    const recentStart = Math.max(0, this.samples.length - Math.max(1000, Math.floor(this.samples.length * 0.25)));
    this.addWeightedCategory(batch, Math.floor(count * 0.25), (sample) => sample.terminal === true || Math.abs(sample.reward ?? 0) > 0.85);
    this.addWeightedCategory(batch, Math.floor(count * 0.18), (sample) => sample.wallMove === true);
    this.addWeightedCategory(batch, Math.floor(count * 0.12), (sample) => (sample.loopPenalty ?? 0) > 0);
    this.addWeightedCategory(batch, Math.floor(count * 0.25), (_sample, index) => index >= recentStart);
    this.addWeightedCategory(batch, count - batch.length, () => true);
    return batch;
  }

  all(): TrainingSample[] {
    return this.samples.slice();
  }

  private addWeightedCategory(batch: TrainingSample[], requested: number, predicate: (sample: TrainingSample, index: number) => boolean): void {
    const count = Math.min(requested, this.samples.length);
    if (count <= 0) return;
    const candidates: Array<{ sample: TrainingSample; weight: number }> = [];
    for (let index = 0; index < this.samples.length; index++) {
      const sample = this.samples[index];
      if (!predicate(sample, index)) continue;
      candidates.push({ sample, weight: clampPriority(sample.priority ?? priorityFromReward(sample.reward)) });
    }
    if (candidates.length === 0) return;
    const totalWeight = candidates.reduce((sum, item) => sum + item.weight, 0);
    for (let i = 0; i < count; i++) {
      if (!Number.isFinite(totalWeight) || totalWeight <= 0) {
        batch.push(candidates[Math.floor(Math.random() * candidates.length)].sample);
        continue;
      }
      let pick = Math.random() * totalWeight;
      let selected = candidates[candidates.length - 1].sample;
      for (const item of candidates) {
        pick -= item.weight;
        if (pick <= 0) {
          selected = item.sample;
          break;
        }
      }
      batch.push(selected);
    }
  }
}

function priorityFromReward(reward: number | undefined): number {
  if (reward === undefined || !Number.isFinite(reward)) return 1;
  return 1 + Math.min(1, Math.abs(reward));
}

function clampPriority(priority: number): number {
  if (!Number.isFinite(priority)) return 1;
  return Math.max(0.05, Math.min(5, priority));
}

export function compactSample(sample: TrainingSample): {
  encodedState: string;
  legalMask: string;
  actionIndex: number;
  reward?: number;
  policyTarget?: string;
  priority?: number;
  source?: TrainingSample["source"];
  terminal?: boolean;
  wallMove?: boolean;
  loopPenalty?: number;
  pathAdvantage?: number;
  pathProgress?: number;
} {
  return {
    encodedState: Buffer.from(sample.encodedState.buffer, sample.encodedState.byteOffset, sample.encodedState.byteLength).toString("base64"),
    legalMask: Buffer.from(sample.legalMask).toString("base64"),
    actionIndex: sample.actionIndex,
    reward: sample.reward,
    priority: sample.priority,
    source: sample.source,
    terminal: sample.terminal,
    wallMove: sample.wallMove,
    loopPenalty: sample.loopPenalty,
    pathAdvantage: sample.pathAdvantage,
    pathProgress: sample.pathProgress,
    policyTarget: sample.policyTarget
      ? Buffer.from(sample.policyTarget.buffer, sample.policyTarget.byteOffset, sample.policyTarget.byteLength).toString("base64")
      : undefined
  };
}

export function expandSample(sample: ReturnType<typeof compactSample>): TrainingSample {
  const encodedBuffer = Buffer.from(sample.encodedState, "base64");
  const policyBuffer = sample.policyTarget ? Buffer.from(sample.policyTarget, "base64") : null;
  return {
    encodedState: new Float32Array(encodedBuffer.buffer.slice(encodedBuffer.byteOffset, encodedBuffer.byteOffset + encodedBuffer.byteLength)),
    legalMask: new Uint8Array(Buffer.from(sample.legalMask, "base64")),
    actionIndex: sample.actionIndex,
    reward: sample.reward,
    priority: sample.priority,
    source: sample.source,
    terminal: sample.terminal,
    wallMove: sample.wallMove,
    loopPenalty: sample.loopPenalty,
    pathAdvantage: sample.pathAdvantage,
    pathProgress: sample.pathProgress,
    policyTarget: policyBuffer
      ? new Float32Array(policyBuffer.buffer.slice(policyBuffer.byteOffset, policyBuffer.byteOffset + policyBuffer.byteLength))
      : undefined
  };
}
