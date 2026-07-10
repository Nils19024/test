import * as fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { alphaBetaAgent } from "../ai/alphabeta";
import { selectBestMoveAgainstGreedy } from "../ai/greedyBestResponse";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { searchBestMoveHybrid } from "../ai/hybrid";
import { QuoridorState } from "../core/state";
import { TfjsQuoridorPolicyModel } from "../ml/model";
import { createPrefixStressState } from "../ml/prefixStressSamples";
import { isMainModule } from "../util/isMain";
import { runMirroredMatchup } from "./benchmark";

async function loadBrowserModelFromDir(dir: string): Promise<tf.LayersModel> {
  const modelJson = JSON.parse(await fs.readFile(path.join(dir, "model.json"), "utf8"));
  const weightData = await fs.readFile(path.join(dir, "weights.bin"));
  return tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs: modelJson.weightsManifest[0].weights,
      weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength)
    })
  );
}

export function createDeterministicPrefixState(game: number, seedBase = 1000): QuoridorState {
  return createPrefixStressState(game, seedBase);
}

export async function runPrefixStress(
  modelDir = "public/models/latest",
  options: { only?: "greedy" | "best-response" | "alphabeta"; games?: number; profile?: "pro" | "elite-base" | "elite" } = {}
) {
  const model = new TfjsQuoridorPolicyModel(await loadBrowserModelFromDir(modelDir));
  const elite = options.profile === "elite" || options.profile === "elite-base";
  const adaptive = options.profile === "elite";
  const proHybrid = {
    name: adaptive ? "EliteHybrid adaptive 9500ms" : elite ? "EliteHybrid base 1200ms" : "ProHybrid 650ms",
    async selectMove(state: QuoridorState) {
      return (await searchBestMoveHybrid(state, model, {
        timeMs: adaptive ? 650 : elite ? 1200 : 650,
        maxDepth: adaptive ? 7 : elite ? 8 : 7,
        topKFromModel: 12,
        maxTimeMs: adaptive ? 9500 : undefined,
        criticalTimeMs: adaptive ? 1200 : undefined,
        criticalMaxDepth: adaptive ? 10 : undefined
      })).move;
    }
  };
  const initialState = (game: number) => createDeterministicPrefixState(game);
  const bestResponseVsGreedy = {
    name: "BestResponseVsGreedy",
    selectMove: selectBestMoveAgainstGreedy
  };
  const results = [];
  if (!options.only || options.only === "greedy") {
    results.push(await runMirroredMatchup({ agentA: proHybrid, agentB: GreedyShortestPathAgent, games: options.games ?? 6, maxMovesPerGame: 180, initialState }));
  }
  if (!options.only || options.only === "best-response") {
    results.push(await runMirroredMatchup({ agentA: proHybrid, agentB: bestResponseVsGreedy, games: options.games ?? 6, maxMovesPerGame: 180, initialState }));
  }
  if (!options.only || options.only === "alphabeta") {
    results.push(await runMirroredMatchup({ agentA: proHybrid, agentB: alphaBetaAgent(120, 6), games: options.games ?? 4, maxMovesPerGame: 180, initialState }));
  }
  return results;
}

async function main(): Promise<void> {
  const only = process.argv.find((value) => value.startsWith("--only="))?.slice("--only=".length) as "greedy" | "best-response" | "alphabeta" | undefined;
  const gamesValue = process.argv.find((value) => value.startsWith("--games="))?.slice("--games=".length);
  const profile = process.argv.find((value) => value.startsWith("--profile="))?.slice("--profile=".length) as "pro" | "elite-base" | "elite" | undefined;
  console.log(JSON.stringify(await runPrefixStress(process.argv[2] ?? "public/models/latest", {
    only,
    games: gamesValue ? Number(gamesValue) : undefined,
    profile
  }), null, 2));
}

if (isMainModule(import.meta.url)) {
  void main();
}
