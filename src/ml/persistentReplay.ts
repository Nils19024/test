import fs from "node:fs/promises";
import path from "node:path";
import { compactSample, expandSample, TrainingSample } from "./replayBuffer";

export class PersistentReplayStore {
  private filePath: string;

  constructor(private dir = "data/replay", private maxLoadSamples = 20000) {
    this.filePath = path.join(this.dir, "samples.jsonl");
  }

  async load(): Promise<TrainingSample[]> {
    try {
      const content = await fs.readFile(this.filePath, "utf8");
      const lines = content.trim().split("\n").filter(Boolean).slice(-this.maxLoadSamples);
      return lines.map((line) => expandSample(JSON.parse(line)));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
  }

  async append(samples: TrainingSample[]): Promise<void> {
    if (samples.length === 0) return;
    await fs.mkdir(this.dir, { recursive: true });
    const lines = samples.map((sample) => JSON.stringify(compactSample(sample))).join("\n") + "\n";
    await fs.appendFile(this.filePath, lines);
  }
}
