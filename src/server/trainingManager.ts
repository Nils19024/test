import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";
import { exportLatestModel } from "../ml/exportModel";
import { TrainingConfig, TrainingProgress, defaultTrainingConfig, train } from "../ml/train";
import { TrainingBackendInfo } from "../ml/tfBackend";
import { spawnReplayGeneration } from "./replaySpawner";

export type EvaluationResult = unknown;

export type TrainingStatus = {
  running: boolean;
  mode: string;
  gamesPlayed: number;
  samplesCollected: number;
  currentLoss: number | null;
  policyLoss: number | null;
  valueLoss: number | null;
  rewardMean: number | null;
  rewardMin: number | null;
  rewardMax: number | null;
  policyEntropyMean: number | null;
  policyTopProbMean: number | null;
  actionDiversity: number | null;
  avgSamplesPerGame: number | null;
  terminalRate: number | null;
  loopRate: number | null;
  wallMoveRate: number | null;
  forwardProgressRate: number | null;
  pathAdvantageMean: number | null;
  avgGameLength: number | null;
  generationSource: string | null;
  latestEvaluation: EvaluationResult | null;
  modelPath: string | null;
  latestReplayPath: string | null;
  backend: TrainingBackendInfo | null;
  logs: string[];
};

export class TrainingManager extends EventEmitter {
  private stopRequested = false;
  private replayGenerationRunning = false;
  private pendingReplayGeneration: { game: number; modelPath: string; label: string } | null = null;
  private runId: string | null = null;
  private status: TrainingStatus = {
    running: false,
    mode: defaultTrainingConfig.mode,
    gamesPlayed: 0,
    samplesCollected: 0,
    currentLoss: null,
    policyLoss: null,
    valueLoss: null,
    rewardMean: null,
    rewardMin: null,
    rewardMax: null,
    policyEntropyMean: null,
    policyTopProbMean: null,
    actionDiversity: null,
    avgSamplesPerGame: null,
    terminalRate: null,
    loopRate: null,
    wallMoveRate: null,
    forwardProgressRate: null,
    pathAdvantageMean: null,
    avgGameLength: null,
    generationSource: null,
    latestEvaluation: null,
    modelPath: null,
    latestReplayPath: null,
    backend: null,
    logs: []
  };

  getStatus(): TrainingStatus {
    return { ...this.status, logs: this.status.logs.slice(-200) };
  }

  private log(message: string): void {
    const line = `[${new Date().toLocaleTimeString()}] ${message}`;
    this.status.logs.push(line);
    this.status.logs = this.status.logs.slice(-500);
    this.emit("status", this.getStatus());
  }

  private appendMetricSnapshot(progress: TrainingProgress): void {
    const evalAny = progress.evaluation as any;
    const outDir = path.join(process.cwd(), "data/training-metrics");
    const outPath = path.join(outDir, "history.jsonl");
    const record = {
      at: new Date().toISOString(),
      runId: this.runId,
      game: progress.game,
      samples: progress.samples,
      mode: this.status.mode,
      generationSource: progress.generationSource,
      backend: progress.backend?.activeBackend ?? null,
      loss: progress.loss,
      policyLoss: progress.policyLoss,
      valueLoss: progress.valueLoss,
      rewardMean: progress.rewardMean,
      rewardMin: progress.rewardMin,
      rewardMax: progress.rewardMax,
      terminalRate: progress.terminalRate,
      loopRate: progress.loopRate,
      wallMoveRate: progress.wallMoveRate,
      forwardProgressRate: progress.forwardProgressRate,
      pathAdvantageMean: progress.pathAdvantageMean,
      avgGameLength: progress.avgGameLength,
      evaluation: progress.evaluation ? {
        random: evalAny?.random?.winrateA ?? null,
        greedy: evalAny?.greedy?.winrateA ?? null,
        alpha20: evalAny?.alpha20?.winrateA ?? null,
        greedyAgree: evalAny?.greedyAgreement?.agreement ?? null,
        greedyTop3: evalAny?.greedyAgreement?.top3Agreement ?? null,
        greedyProb: evalAny?.greedyAgreement?.greedyProbMean ?? null,
        bestResp: evalAny?.greedyBestResponse?.agreement ?? null,
        bestRespTop3: evalAny?.greedyBestResponse?.top3Agreement ?? null,
        bestRespProb: evalAny?.greedyBestResponse?.teacherProbMean ?? null
      } : null
    };
    try {
      fs.mkdirSync(outDir, { recursive: true });
      fs.appendFileSync(outPath, `${JSON.stringify(record)}\n`);
    } catch (error) {
      this.log(`Metric history append failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private startReplayGeneration(request: { game: number; modelPath: string; label: string }): void {
    if (!fs.existsSync(path.join(request.modelPath, "model.json"))) {
      this.log(`Skipped replay for ${request.label} ${request.game}: no saved model available yet`);
      return;
    }
    if (this.replayGenerationRunning) {
      this.pendingReplayGeneration = request;
      this.log(`Replay generation queued for ${request.label} ${request.game}`);
      return;
    }
    this.replayGenerationRunning = true;
    this.log(`Starting replay generation for ${request.label} ${request.game}`);
    spawnReplayGeneration({
      root: process.cwd(),
      modelPath: request.modelPath,
      checkpointGame: request.game,
      onDone: (message) => {
        this.status.latestReplayPath = `/replays/latest.json?v=${Date.now()}`;
        this.log(message);
        this.finishReplayGeneration();
      },
      onError: (message) => {
        this.log(message);
        this.finishReplayGeneration();
      }
    });
  }

  private finishReplayGeneration(): void {
    this.replayGenerationRunning = false;
    const next = this.pendingReplayGeneration;
    this.pendingReplayGeneration = null;
    if (next) this.startReplayGeneration(next);
  }

  async start(config: Partial<TrainingConfig> = {}): Promise<void> {
    if (this.status.running) throw new Error("Training is already running");
    this.stopRequested = false;
    const merged = { ...defaultTrainingConfig, games: 50, saveEveryGames: 100, replayEveryGames: 10, evaluateEveryGames: 100, ...config };
    this.runId = new Date().toISOString().replace(/[:.]/g, "-");
    this.status = {
      ...this.status,
      running: true,
      mode: merged.mode,
      gamesPlayed: 0,
      samplesCollected: 0,
      currentLoss: null,
      policyLoss: null,
      valueLoss: null,
      rewardMean: null,
      rewardMin: null,
      rewardMax: null,
      policyEntropyMean: null,
      policyTopProbMean: null,
      actionDiversity: null,
      avgSamplesPerGame: null,
      terminalRate: null,
      loopRate: null,
      wallMoveRate: null,
      forwardProgressRate: null,
      pathAdvantageMean: null,
      avgGameLength: null,
      generationSource: null,
      latestEvaluation: null,
      latestReplayPath: null,
      backend: null
    };
    this.emit("status", this.getStatus());
    train(merged, {
      shouldStop: () => this.stopRequested,
      onLog: (message) => this.log(message),
      onProgress: (progress: TrainingProgress) => {
        this.status.gamesPlayed = progress.game;
        this.status.samplesCollected = progress.samples;
        this.status.currentLoss = progress.loss;
        this.status.policyLoss = progress.policyLoss;
        this.status.valueLoss = progress.valueLoss;
        this.status.rewardMean = progress.rewardMean;
        this.status.rewardMin = progress.rewardMin;
        this.status.rewardMax = progress.rewardMax;
        this.status.policyEntropyMean = progress.policyEntropyMean;
        this.status.policyTopProbMean = progress.policyTopProbMean;
        this.status.actionDiversity = progress.actionDiversity;
        this.status.avgSamplesPerGame = progress.avgSamplesPerGame;
        this.status.terminalRate = progress.terminalRate;
        this.status.loopRate = progress.loopRate;
        this.status.wallMoveRate = progress.wallMoveRate;
        this.status.forwardProgressRate = progress.forwardProgressRate;
        this.status.pathAdvantageMean = progress.pathAdvantageMean;
        this.status.avgGameLength = progress.avgGameLength;
        this.status.generationSource = progress.generationSource;
        this.status.modelPath = progress.modelPath;
        this.status.backend = progress.backend;
        if (progress.evaluation) this.status.latestEvaluation = progress.evaluation;
        this.appendMetricSnapshot(progress);
        this.emit("status", this.getStatus());
      },
      onCheckpoint: (checkpoint) => {
        this.startReplayGeneration({ game: checkpoint.game, modelPath: checkpoint.modelPath, label: "checkpoint game" });
      },
      onReplayModel: (replayModel) => {
        this.startReplayGeneration({ game: replayModel.game, modelPath: replayModel.modelPath, label: "game" });
      }
    }).then((result) => {
      this.status.running = false;
      this.status.modelPath = result.modelPath;
      this.status.backend = result.backend;
      this.log(`Training completed with ${result.samples} samples`);
      this.runId = null;
    }).catch((error) => {
      this.status.running = false;
      this.log(`Training failed: ${error instanceof Error ? error.message : String(error)}`);
      this.runId = null;
    });
  }

  stop(): void {
    this.stopRequested = true;
    this.log("Stop requested");
  }

  async exportBrowser(): Promise<string> {
    const target = await exportLatestModel(this.status.modelPath ?? "models/checkpoints/latest", "public/models/latest");
    this.status.modelPath = target;
    this.log(`Exported browser model to ${target}`);
    return target;
  }
}
