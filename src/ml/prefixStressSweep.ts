import * as fs from "node:fs/promises";
import path from "node:path";
import { BenchmarkResult } from "../benchmark/benchmark";
import { runPrefixStress } from "../benchmark/prefixStress";
import { runStrengthGate } from "../benchmark/strengthGate";
import { isMainModule } from "../util/isMain";
import { exportLatestModel } from "./exportModel";
import { runPrefixStressFineTune } from "./prefixStressFineTune";
import { TrainingBackendPreference } from "./tfBackend";

type SweepCandidate = {
  name: string;
  games: number;
  rolloutPlies: number;
  updates: number;
  batchSize: number;
  teacherTimeMs: number;
  teacherMaxDepth: number;
  anchorGames: number;
  anchorMaxMovesPerGame: number;
  learningRate: number;
};

type SweepConfig = {
  sourceDir: string;
  targetDir: string;
  outDir: string;
  backend: TrainingBackendPreference;
  promote: boolean;
  minPrefixScoreGain: number;
  maxMatchupScoreDrop: number;
  prefixRepeats: number;
  profile: "smoke" | "conservative";
};

type CandidateReport = {
  name: string;
  candidateDir: string;
  train: unknown;
  strengthPassed: boolean;
  strength: BenchmarkResult[];
  prefix?: BenchmarkResult[];
  prefixRuns?: BenchmarkResult[][];
  prefixScore?: number;
  prefixGain?: number;
  nonRegressed?: boolean;
};

const conservativeCandidates: SweepCandidate[] = [
  {
    name: "lite-lr20-u6-a12",
    games: 8,
    rolloutPlies: 5,
    updates: 6,
    batchSize: 64,
    teacherTimeMs: 120,
    teacherMaxDepth: 5,
    anchorGames: 12,
    anchorMaxMovesPerGame: 80,
    learningRate: 0.00002
  },
  {
    name: "balanced-lr30-u10-a12",
    games: 8,
    rolloutPlies: 5,
    updates: 10,
    batchSize: 64,
    teacherTimeMs: 120,
    teacherMaxDepth: 5,
    anchorGames: 12,
    anchorMaxMovesPerGame: 80,
    learningRate: 0.00003
  },
  {
    name: "deep-prefix-lr20-u10-a16",
    games: 10,
    rolloutPlies: 6,
    updates: 10,
    batchSize: 64,
    teacherTimeMs: 160,
    teacherMaxDepth: 6,
    anchorGames: 16,
    anchorMaxMovesPerGame: 80,
    learningRate: 0.00002
  }
];

const smokeCandidates: SweepCandidate[] = [
  {
    name: "smoke",
    games: 1,
    rolloutPlies: 2,
    updates: 1,
    batchSize: 8,
    teacherTimeMs: 5,
    teacherMaxDepth: 2,
    anchorGames: 1,
    anchorMaxMovesPerGame: 8,
    learningRate: 0.00002
  }
];

function parseArgs(): SweepConfig {
  const parsed: Record<string, string> = {};
  for (const arg of process.argv.slice(2)) {
    if (!arg.startsWith("--")) continue;
    const [key, value] = arg.slice(2).split("=", 2);
    parsed[key] = value ?? "true";
  }
  return {
    sourceDir: parsed.sourceDir ?? "public/models/latest",
    targetDir: parsed.targetDir ?? "public/models/latest",
    outDir: parsed.outDir ?? "models/candidates/prefix-sweep",
    backend: (parsed.backend as TrainingBackendPreference | undefined) ?? "auto",
    promote: parsed.promote === "true",
    minPrefixScoreGain: parsed.minPrefixScoreGain ? Number(parsed.minPrefixScoreGain) : 0.05,
    maxMatchupScoreDrop: parsed.maxMatchupScoreDrop ? Number(parsed.maxMatchupScoreDrop) : 0,
    prefixRepeats: parsed.prefixRepeats ? Number(parsed.prefixRepeats) : 2,
    profile: parsed.profile === "smoke" ? "smoke" : "conservative"
  };
}

function matchupScore(result: BenchmarkResult): number {
  return (result.winsA + result.draws * 0.5) / Math.max(1, result.games);
}

function aggregatePrefixScore(results: BenchmarkResult[]): number {
  return results.reduce((sum, result) => sum + matchupScore(result), 0) / Math.max(1, results.length);
}

function strengthPassed(results: BenchmarkResult[]): boolean {
  return results.every((result) => result.illegalMoves === 0
    && result.winsB === 0
    && matchupScore(result) >= 0.75
    && result.avgMoveMsA < 10000);
}

function prefixMatchupsNonRegressed(current: BenchmarkResult[], candidate: BenchmarkResult[], maxDrop: number): boolean {
  return current.every((currentResult, index) => {
    const candidateResult = candidate[index];
    if (!candidateResult) return false;
    return matchupScore(candidateResult) + maxDrop >= matchupScore(currentResult);
  });
}

function averageBenchmarkResults(runs: BenchmarkResult[][]): BenchmarkResult[] {
  if (runs.length === 0) return [];
  return runs[0].map((first, index) => {
    const values = runs.map((run) => run[index]).filter(Boolean);
    const games = values.reduce((sum, value) => sum + value.games, 0);
    const winsA = values.reduce((sum, value) => sum + value.winsA, 0);
    const winsB = values.reduce((sum, value) => sum + value.winsB, 0);
    const draws = values.reduce((sum, value) => sum + value.draws, 0);
    return {
      matchup: `${first.matchup} avg${values.length}`,
      games,
      winsA,
      winsB,
      draws,
      winrateA: winsA / Math.max(1, games),
      avgMoveMsA: values.reduce((sum, value) => sum + value.avgMoveMsA, 0) / Math.max(1, values.length),
      avgMoveMsB: values.reduce((sum, value) => sum + value.avgMoveMsB, 0) / Math.max(1, values.length),
      avgGameLength: values.reduce((sum, value) => sum + value.avgGameLength, 0) / Math.max(1, values.length),
      illegalMoves: values.reduce((sum, value) => sum + value.illegalMoves, 0)
    };
  });
}

async function runRepeatedPrefixStress(modelDir: string, repeats: number): Promise<{ average: BenchmarkResult[]; runs: BenchmarkResult[][] }> {
  const runs: BenchmarkResult[][] = [];
  for (let repeat = 0; repeat < Math.max(1, repeats); repeat++) {
    runs.push(await runPrefixStress(modelDir) as BenchmarkResult[]);
  }
  return { average: averageBenchmarkResults(runs), runs };
}

export async function runPrefixStressSweep(config: Partial<SweepConfig> = {}) {
  const cfg: SweepConfig = {
    sourceDir: "public/models/latest",
    targetDir: "public/models/latest",
    outDir: "models/candidates/prefix-sweep",
    backend: "auto",
    promote: false,
    minPrefixScoreGain: 0.05,
    maxMatchupScoreDrop: 0,
    prefixRepeats: 2,
    profile: "conservative",
    ...config
  };
  const candidates = cfg.profile === "smoke" ? smokeCandidates : conservativeCandidates;
  await fs.mkdir(cfg.outDir, { recursive: true });

  const currentRepeatedPrefix = await runRepeatedPrefixStress(cfg.sourceDir, cfg.profile === "smoke" ? 1 : cfg.prefixRepeats);
  const currentPrefix = currentRepeatedPrefix.average;
  const currentPrefixScore = aggregatePrefixScore(currentPrefix);
  const reports: CandidateReport[] = [];

  for (const candidate of candidates) {
    const candidateDir = path.join(cfg.outDir, candidate.name);
    const train = await runPrefixStressFineTune({
      sourceDir: cfg.sourceDir,
      candidateDir,
      games: candidate.games,
      rolloutPlies: candidate.rolloutPlies,
      updates: candidate.updates,
      batchSize: candidate.batchSize,
      teacherTimeMs: candidate.teacherTimeMs,
      teacherMaxDepth: candidate.teacherMaxDepth,
      anchorGames: candidate.anchorGames,
      anchorMaxMovesPerGame: candidate.anchorMaxMovesPerGame,
      learningRate: candidate.learningRate,
      backend: cfg.backend,
      skipBenchmarks: true
    });
    const strengthGate = await runStrengthGate(candidateDir);
    const strength = strengthGate.results as BenchmarkResult[];
    const passed = strengthGate.passed && strengthPassed(strength);
    const report: CandidateReport = {
      name: candidate.name,
      candidateDir,
      train,
      strengthPassed: passed,
      strength
    };
    if (passed && cfg.profile !== "smoke") {
      const repeatedPrefix = await runRepeatedPrefixStress(candidateDir, cfg.prefixRepeats);
      const prefix = repeatedPrefix.average;
      const prefixScore = aggregatePrefixScore(prefix);
      report.prefix = prefix;
      report.prefixRuns = repeatedPrefix.runs;
      report.prefixScore = prefixScore;
      report.prefixGain = prefixScore - currentPrefixScore;
      report.nonRegressed = prefixMatchupsNonRegressed(currentPrefix, prefix, cfg.maxMatchupScoreDrop);
    }
    reports.push(report);
  }

  const promotable = reports
    .filter((report) => report.strengthPassed
      && report.nonRegressed
      && (report.prefixGain ?? -Infinity) >= cfg.minPrefixScoreGain)
    .sort((a, b) => (b.prefixScore ?? -Infinity) - (a.prefixScore ?? -Infinity))[0];

  let promoted = false;
  let reason = "no candidate passed promotion gates";
  if (promotable && cfg.promote) {
    await exportLatestModel(promotable.candidateDir, cfg.targetDir);
    promoted = true;
    reason = `promoted ${promotable.name}`;
  } else if (promotable) {
    reason = `best candidate ${promotable.name} passed promotion gates, dry-run only`;
  }

  const result = {
    sourceDir: cfg.sourceDir,
    targetDir: cfg.targetDir,
    promoted,
    reason,
    currentPrefixScore,
    currentPrefix,
    currentPrefixRuns: currentRepeatedPrefix.runs,
    reports
  };
  await fs.writeFile(path.join(cfg.outDir, "sweep-report.json"), JSON.stringify(result, null, 2));
  return result;
}

async function main(): Promise<void> {
  const cfg = parseArgs();
  const result = await runPrefixStressSweep(cfg);
  console.log(JSON.stringify(result, null, 2));
  if (cfg.promote && !result.promoted) process.exitCode = 1;
}

if (isMainModule(import.meta.url)) {
  void main();
}
