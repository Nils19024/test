import fs from "node:fs/promises";
import path from "node:path";
import { TrainingStatus } from "./trainingManager";

type ReplaySummary = {
  id: string;
  checkpointGame: number;
  matchup: string;
  winner: 0 | 1 | null;
  steps: number;
  firstMoves: unknown[];
  wallCounts: Record<string, number>;
} | null;

type MetricHistoryRecord = {
  at: string;
  runId: string | null;
  game: number;
  samples: number;
  mode: string;
  generationSource: string | null;
  backend: string | null;
  loss: number | null;
  policyLoss: number | null;
  valueLoss: number | null;
  rewardMean: number | null;
  terminalRate: number | null;
  loopRate: number | null;
  wallMoveRate: number | null;
  forwardProgressRate: number | null;
  pathAdvantageMean: number | null;
  avgGameLength: number | null;
  evaluation: {
    random: number | null;
    greedy: number | null;
    alpha20: number | null;
    greedyAgree: number | null;
    greedyTop3: number | null;
    greedyProb: number | null;
    bestResp: number | null;
    bestRespTop3: number | null;
    bestRespProb: number | null;
  } | null;
};

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function replaySummary(root: string): Promise<ReplaySummary> {
  const latestPath = path.join(root, "public/replays/latest.json");
  if (!(await pathExists(latestPath))) return null;
  const replay = JSON.parse(await fs.readFile(latestPath, "utf8"));
  const wallCounts: Record<string, number> = {};
  for (const step of replay.steps ?? []) {
    if (step.move?.type === "pawn") continue;
    const key = `p${step.player}-${step.move.type}-${step.move.row}-${step.move.col}`;
    wallCounts[key] = (wallCounts[key] ?? 0) + 1;
  }
  return {
    id: replay.id,
    checkpointGame: replay.checkpointGame,
    matchup: replay.matchup,
    winner: replay.winner,
    steps: replay.steps?.length ?? 0,
    firstMoves: (replay.steps ?? []).slice(0, 12).map((step: any) => ({
      ply: step.ply,
      player: step.player,
      agent: step.agent,
      move: step.move
    })),
    wallCounts
  };
}

async function replayBufferInfo(root: string) {
  const replayPath = path.join(root, "data/replay/samples.jsonl");
  if (!(await pathExists(replayPath))) return { exists: false, samplesApprox: 0, bytes: 0 };
  const stat = await fs.stat(replayPath);
  const content = await fs.readFile(replayPath, "utf8");
  const lines = content.split("\n").filter(Boolean);
  return {
    exists: true,
    samplesApprox: lines.length,
    bytes: stat.size,
    lastSamplePreview: lines.at(-1)?.slice(0, 500) ?? null
  };
}

async function metricHistory(root: string): Promise<MetricHistoryRecord[]> {
  const historyPath = path.join(root, "data/training-metrics/history.jsonl");
  if (!(await pathExists(historyPath))) return [];
  const content = await fs.readFile(historyPath, "utf8");
  return content
    .split("\n")
    .filter(Boolean)
    .slice(-1000)
    .flatMap((line) => {
      try {
        const record = JSON.parse(line) as MetricHistoryRecord;
        return Number.isFinite(record.game) ? [record] : [];
      } catch {
        return [];
      }
    });
}

function deriveSignals(status: TrainingStatus, replay: ReplaySummary) {
  const evalAny = status.latestEvaluation as any;
  const random = evalAny?.random?.winrateA ?? null;
  const greedy = evalAny?.greedy?.winrateA ?? null;
  const alpha20 = evalAny?.alpha20?.winrateA ?? null;
  const warnings: string[] = [];
  const replayMatchesStatus = replay !== null && replay.checkpointGame === status.gamesPlayed && status.gamesPlayed > 0;
  if (status.gamesPlayed >= 300 && random === 0) warnings.push("random_winrate_zero_after_300_games");
  if (status.gamesPlayed >= 300 && greedy === 0) warnings.push("greedy_winrate_zero_after_300_games");
  if (status.valueLoss !== null && status.valueLoss > 0.5) warnings.push("high_value_loss");
  if (status.terminalRate !== null && status.gamesPlayed >= 300 && status.terminalRate < 0.15) warnings.push("low_terminal_rate");
  if (status.loopRate !== null && status.loopRate > 0.2) warnings.push("high_loop_rate");
  if (status.wallMoveRate !== null && status.wallMoveRate > 0.55) warnings.push("high_wall_move_rate");
  if (status.forwardProgressRate !== null && status.forwardProgressRate < 0.35) warnings.push("low_forward_progress_rate");
  if (replayMatchesStatus && replay.winner === 1) warnings.push("latest_replay_model_lost");
  if (replayMatchesStatus && replay.steps <= 25 && replay.winner === 1) warnings.push("latest_replay_fast_loss");
  return { random, greedy, alpha20, warnings };
}

function parseNumber(value: string | undefined): number | null {
  if (value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function slope(values: number[]): number | null {
  if (values.length < 2) return null;
  const n = values.length;
  const meanX = (n - 1) / 2;
  const meanY = values.reduce((sum, value) => sum + value, 0) / n;
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < n; index++) {
    numerator += (index - meanX) * (values[index] - meanY);
    denominator += (index - meanX) ** 2;
  }
  return denominator > 0 ? numerator / denominator : null;
}

function learningTrendFromLogs(logs: string[]) {
  const evaluations = logs.flatMap((line) => {
    const match = line.match(/Evaluation after (\d+): random=([\d.]+), greedy=([\d.]+), alpha20=([\d.]+).*bestResp=([\d.]+), bestRespTop3=([\d.]+), bestRespProb=([\d.]+)/);
    if (!match) return [];
    return [{
      game: Number(match[1]),
      random: Number(match[2]),
      greedy: Number(match[3]),
      alpha20: Number(match[4]),
      bestResp: Number(match[5]),
      bestRespTop3: Number(match[6]),
      bestRespProb: Number(match[7])
    }];
  });
  const timings = logs.flatMap((line) => {
    const match = line.match(/Games (\d+)(?:-\d+)? timings:.*rewardMean=([-\d.]+|-).*terminal=([-\d.]+|-).*wall=([-\d.]+|-).*forward=([-\d.]+|-)/);
    if (!match) return [];
    return [{
      game: Number(match[1]),
      rewardMean: parseNumber(match[2]),
      terminal: parseNumber(match[3]),
      wall: parseNumber(match[4]),
      forward: parseNumber(match[5])
    }];
  });
  const lastEvaluations = evaluations.slice(-8);
  const lastTimings = timings.slice(-20);
  const bestRespValues = lastEvaluations.map((item) => item.bestResp);
  const bestRespTop3Values = lastEvaluations.map((item) => item.bestRespTop3);
  const bestRespProbValues = lastEvaluations.map((item) => item.bestRespProb);
  const greedyValues = lastEvaluations.map((item) => item.greedy);
  const rewardValues = lastTimings.map((item) => item.rewardMean).filter((value): value is number => value !== null);
  const forwardValues = lastTimings.map((item) => item.forward).filter((value): value is number => value !== null);
  const terminalValues = lastTimings.map((item) => item.terminal).filter((value): value is number => value !== null);
  const latestEvaluation = lastEvaluations.at(-1) ?? null;
  const latestTiming = lastTimings.at(-1) ?? null;
  const bestRespSlope = slope(bestRespValues);
  const bestRespTop3Slope = slope(bestRespTop3Values);
  const bestRespProbSlope = slope(bestRespProbValues);
  const greedySlope = slope(greedyValues);
  const rewardSlope = slope(rewardValues);
  const plausibleGreedySoon =
    latestEvaluation !== null
    && (
      latestEvaluation.greedy > 0
      || latestEvaluation.bestResp >= 0.35
      || latestEvaluation.bestRespTop3 >= 0.65
      || ((bestRespSlope ?? 0) > 0.03 && latestEvaluation.bestResp >= 0.2)
    );
  const flags: string[] = [];
  if (lastEvaluations.length >= 3 && (bestRespSlope ?? 0) <= 0) flags.push("best_response_not_improving");
  if (lastEvaluations.length >= 3 && (bestRespProbSlope ?? 0) <= 0) flags.push("teacher_probability_not_improving");
  if (latestEvaluation && latestEvaluation.bestRespTop3 < 0.2 && latestEvaluation.game >= 50) flags.push("low_best_response_top3_after_50_games");
  if (latestTiming?.forward !== null && latestTiming && latestTiming.forward < 0.5) flags.push("latest_forward_rate_low");
  if (rewardValues.length >= 3 && (rewardSlope ?? 0) < -0.02) flags.push("reward_trending_down");
  return {
    evaluations: lastEvaluations,
    timings: lastTimings,
    slopes: {
      bestResp: bestRespSlope,
      bestRespTop3: bestRespTop3Slope,
      bestRespProb: bestRespProbSlope,
      greedy: greedySlope,
      rewardMean: rewardSlope,
      forward: slope(forwardValues),
      terminal: slope(terminalValues)
    },
    latestEvaluation,
    latestTiming,
    plausibleGreedySoon,
    flags
  };
}

function learningTrendFromHistory(history: MetricHistoryRecord[]) {
  const recent = history.slice(-250);
  const evaluationRecords = recent.filter((record) => record.evaluation);
  const lastEvaluations = evaluationRecords.slice(-12).map((record) => ({
    game: record.game,
    random: record.evaluation?.random ?? 0,
    greedy: record.evaluation?.greedy ?? 0,
    alpha20: record.evaluation?.alpha20 ?? 0,
    bestResp: record.evaluation?.bestResp ?? 0,
    bestRespTop3: record.evaluation?.bestRespTop3 ?? 0,
    bestRespProb: record.evaluation?.bestRespProb ?? 0
  }));
  const lastTimings = recent.slice(-40).map((record) => ({
    game: record.game,
    rewardMean: record.rewardMean,
    terminal: record.terminalRate,
    wall: record.wallMoveRate,
    forward: record.forwardProgressRate
  }));
  const bestRespValues = lastEvaluations.map((item) => item.bestResp);
  const bestRespTop3Values = lastEvaluations.map((item) => item.bestRespTop3);
  const bestRespProbValues = lastEvaluations.map((item) => item.bestRespProb);
  const greedyValues = lastEvaluations.map((item) => item.greedy);
  const rewardValues = lastTimings.map((item) => item.rewardMean).filter((value): value is number => value !== null);
  const forwardValues = lastTimings.map((item) => item.forward).filter((value): value is number => value !== null);
  const terminalValues = lastTimings.map((item) => item.terminal).filter((value): value is number => value !== null);
  const latestEvaluation = lastEvaluations.at(-1) ?? null;
  const latestTiming = lastTimings.at(-1) ?? null;
  const bestRespSlope = slope(bestRespValues);
  const bestRespTop3Slope = slope(bestRespTop3Values);
  const bestRespProbSlope = slope(bestRespProbValues);
  const greedySlope = slope(greedyValues);
  const rewardSlope = slope(rewardValues);
  const plausibleGreedySoon =
    latestEvaluation !== null
    && (
      latestEvaluation.greedy > 0
      || latestEvaluation.bestResp >= 0.35
      || latestEvaluation.bestRespTop3 >= 0.65
      || ((bestRespSlope ?? 0) > 0.03 && latestEvaluation.bestResp >= 0.2)
    );
  const flags: string[] = [];
  if (lastEvaluations.length >= 3 && (bestRespSlope ?? 0) <= 0) flags.push("best_response_not_improving");
  if (lastEvaluations.length >= 3 && (bestRespProbSlope ?? 0) <= 0) flags.push("teacher_probability_not_improving");
  if (latestEvaluation && latestEvaluation.bestRespTop3 < 0.2 && latestEvaluation.game >= 50) flags.push("low_best_response_top3_after_50_games");
  if (latestTiming?.forward !== null && latestTiming && latestTiming.forward < 0.5) flags.push("latest_forward_rate_low");
  if (rewardValues.length >= 3 && (rewardSlope ?? 0) < -0.02) flags.push("reward_trending_down");
  return {
    source: "history",
    records: recent.length,
    evaluations: lastEvaluations,
    timings: lastTimings,
    slopes: {
      bestResp: bestRespSlope,
      bestRespTop3: bestRespTop3Slope,
      bestRespProb: bestRespProbSlope,
      greedy: greedySlope,
      rewardMean: rewardSlope,
      forward: slope(forwardValues),
      terminal: slope(terminalValues)
    },
    latestEvaluation,
    latestTiming,
    plausibleGreedySoon,
    flags
  };
}

export async function createTrainingDiagnostics(root: string, status: TrainingStatus) {
  const replay = await replaySummary(root);
  const buffer = await replayBufferInfo(root);
  const history = await metricHistory(root);
  const signals = deriveSignals(status, replay);
  const learningTrend = history.length > 0
    ? learningTrendFromHistory(history)
    : { source: "logs", ...learningTrendFromLogs(status.logs) };
  const report = {
    createdAt: new Date().toISOString(),
    status: {
      running: status.running,
      mode: status.mode,
      gamesPlayed: status.gamesPlayed,
      samplesCollected: status.samplesCollected,
      currentLoss: status.currentLoss,
      policyLoss: status.policyLoss,
      valueLoss: status.valueLoss,
      rewardMean: status.rewardMean,
      terminalRate: status.terminalRate,
      loopRate: status.loopRate,
      wallMoveRate: status.wallMoveRate,
      forwardProgressRate: status.forwardProgressRate,
      pathAdvantageMean: status.pathAdvantageMean,
      avgGameLength: status.avgGameLength,
      generationSource: status.generationSource,
      backend: status.backend,
      modelPath: status.modelPath,
      latestReplayPath: status.latestReplayPath,
      latestEvaluation: status.latestEvaluation
    },
    signals,
    learningTrend,
    metricHistory: {
      exists: history.length > 0,
      records: history.length,
      latest: history.at(-1) ?? null
    },
    replay,
    replayBuffer: buffer,
    recentLogs: status.logs.slice(-80)
  };
  const outDir = path.join(root, "diagnostics");
  await fs.mkdir(outDir, { recursive: true });
  const outPath = path.join(outDir, "latest-training-diagnostics.json");
  await fs.writeFile(outPath, JSON.stringify(report, null, 2));
  return { path: outPath, report };
}
