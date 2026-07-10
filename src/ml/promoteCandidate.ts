import { BenchmarkResult } from "../benchmark/benchmark";
import { runPrefixStress } from "../benchmark/prefixStress";
import { runStrengthGate } from "../benchmark/strengthGate";
import { isMainModule } from "../util/isMain";
import { exportLatestModel } from "./exportModel";

type PromoteConfig = {
  candidateDir: string;
  currentDir: string;
  targetDir: string;
  minPrefixScoreGain: number;
  maxMatchupScoreDrop: number;
  dryRun: boolean;
};

type PromotionReport = {
  promoted: boolean;
  reason: string;
  candidateDir: string;
  currentDir: string;
  targetDir: string;
  currentPrefixScore: number;
  candidatePrefixScore: number;
  candidateStrengthPassed: boolean;
  prefixMatchupsNonRegressed: boolean;
  currentPrefix: BenchmarkResult[];
  candidatePrefix: BenchmarkResult[];
  candidateStrength: BenchmarkResult[];
};

const defaultConfig: PromoteConfig = {
  candidateDir: "models/candidates/prefix-stress",
  currentDir: "public/models/latest",
  targetDir: "public/models/latest",
  minPrefixScoreGain: 0.05,
  maxMatchupScoreDrop: 0,
  dryRun: true
};

function parseArgs(): PromoteConfig {
  const parsed: Record<string, string> = {};
  for (const arg of process.argv.slice(2)) {
    if (!arg.startsWith("--")) continue;
    const [key, value] = arg.slice(2).split("=", 2);
    parsed[key] = value ?? "true";
  }
  return {
    candidateDir: parsed.candidateDir ?? defaultConfig.candidateDir,
    currentDir: parsed.currentDir ?? defaultConfig.currentDir,
    targetDir: parsed.targetDir ?? defaultConfig.targetDir,
    minPrefixScoreGain: parsed.minPrefixScoreGain ? Number(parsed.minPrefixScoreGain) : defaultConfig.minPrefixScoreGain,
    maxMatchupScoreDrop: parsed.maxMatchupScoreDrop ? Number(parsed.maxMatchupScoreDrop) : defaultConfig.maxMatchupScoreDrop,
    dryRun: parsed.promote !== "true"
  };
}

function matchupScore(result: BenchmarkResult): number {
  return (result.winsA + result.draws * 0.5) / Math.max(1, result.games);
}

function aggregatePrefixScore(results: BenchmarkResult[]): number {
  if (results.length === 0) return 0;
  return results.reduce((sum, result) => sum + matchupScore(result), 0) / results.length;
}

function strengthPassed(results: BenchmarkResult[]): boolean {
  return results.every((result) => {
    const scoreA = matchupScore(result);
    return result.illegalMoves === 0
      && result.winsB === 0
      && scoreA >= 0.75
      && result.avgMoveMsA < 10000;
  });
}

function prefixMatchupsNonRegressed(current: BenchmarkResult[], candidate: BenchmarkResult[], maxDrop: number): boolean {
  return current.every((currentResult, index) => {
    const candidateResult = candidate[index];
    if (!candidateResult) return false;
    return matchupScore(candidateResult) + maxDrop >= matchupScore(currentResult);
  });
}

export async function evaluateAndMaybePromoteCandidate(config: Partial<PromoteConfig> = {}): Promise<PromotionReport> {
  const cfg = { ...defaultConfig, ...config };
  const currentPrefix = await runPrefixStress(cfg.currentDir);
  const candidatePrefix = await runPrefixStress(cfg.candidateDir);
  const candidateStrengthGate = await runStrengthGate(cfg.candidateDir);
  const candidateStrength = candidateStrengthGate.results as BenchmarkResult[];
  const currentPrefixScore = aggregatePrefixScore(currentPrefix as BenchmarkResult[]);
  const candidatePrefixScore = aggregatePrefixScore(candidatePrefix as BenchmarkResult[]);
  const candidateStrengthPassed = candidateStrengthGate.passed && strengthPassed(candidateStrength);
  const nonRegressed = prefixMatchupsNonRegressed(currentPrefix as BenchmarkResult[], candidatePrefix as BenchmarkResult[], cfg.maxMatchupScoreDrop);
  const prefixImproved = candidatePrefixScore >= currentPrefixScore + cfg.minPrefixScoreGain;

  let promoted = false;
  let reason = "dry-run";
  if (!candidateStrengthPassed) {
    reason = "candidate failed strength gate";
  } else if (!nonRegressed) {
    reason = "candidate regressed at least one prefix matchup";
  } else if (!prefixImproved) {
    reason = `candidate prefix score ${candidatePrefixScore.toFixed(3)} did not beat current ${currentPrefixScore.toFixed(3)} by ${cfg.minPrefixScoreGain.toFixed(3)}`;
  } else if (cfg.dryRun) {
    reason = "candidate passed, dry-run only";
  } else {
    await exportLatestModel(cfg.candidateDir, cfg.targetDir);
    promoted = true;
    reason = `promoted ${cfg.candidateDir} to ${cfg.targetDir}`;
  }

  return {
    promoted,
    reason,
    candidateDir: cfg.candidateDir,
    currentDir: cfg.currentDir,
    targetDir: cfg.targetDir,
    currentPrefixScore,
    candidatePrefixScore,
    candidateStrengthPassed,
    prefixMatchupsNonRegressed: nonRegressed,
    currentPrefix: currentPrefix as BenchmarkResult[],
    candidatePrefix: candidatePrefix as BenchmarkResult[],
    candidateStrength
  };
}

async function main(): Promise<void> {
  const report = await evaluateAndMaybePromoteCandidate(parseArgs());
  console.log(JSON.stringify(report, null, 2));
  if (!report.promoted && report.reason !== "candidate passed, dry-run only") process.exitCode = 1;
}

if (isMainModule(import.meta.url)) {
  void main();
}
