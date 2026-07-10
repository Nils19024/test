import * as fs from "node:fs/promises";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { searchBestMoveHybrid } from "../ai/hybrid";
import { moveToActionIndex } from "../core/actions";
import { encodeStateForModel } from "../core/encode";
import { applyKnownLegalMove } from "../core/moves";
import { getModelLegalActionMask, modelActionIndexToMove } from "../core/modelPerspective";
import { shortestPathLength } from "../core/pathfinding";
import { QuoridorState, getWinner, otherPlayer } from "../core/state";
import { TfjsQuoridorPolicyModel, chooseBestLegalAction } from "../ml/model";
import { createDeterministicPrefixState } from "./prefixStress";

async function loadModel(dir: string): Promise<TfjsQuoridorPolicyModel> {
  const modelJson = JSON.parse(await fs.readFile(path.join(dir, "model.json"), "utf8"));
  const weightData = await fs.readFile(path.join(dir, "weights.bin"));
  const model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: modelJson.modelTopology,
    weightSpecs: modelJson.weightsManifest[0].weights,
    weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength)
  }));
  return new TfjsQuoridorPolicyModel(model);
}

function numberArg(name: string, fallback: number): number {
  const value = process.argv.find((arg) => arg.startsWith(`--${name}=`))?.split("=")[1];
  return value === undefined ? fallback : Number(value);
}

async function main(): Promise<void> {
  const game = numberArg("game", 0);
  const aiPlayer = numberArg("player", 0) === 1 ? 1 : 0;
  const timeMs = numberArg("timeMs", 650);
  const maxPlies = numberArg("maxPlies", 180);
  const model = await loadModel(process.argv[2] ?? "public/models/latest");
  let state: QuoridorState = createDeterministicPrefixState(game);
  const decisions: unknown[] = [];

  while (getWinner(state) === null && decisions.length < maxPlies) {
    if (state.turn !== aiPlayer) {
      state = applyKnownLegalMove(state, await GreedyShortestPathAgent.selectMove(state));
      continue;
    }

    const player = state.turn;
    const opponent = otherPlayer(player);
    const legalMask = getModelLegalActionMask(state);
    const prediction = await model.predict(encodeStateForModel(state), legalMask);
    const modelAction = chooseBestLegalAction(prediction.policy, legalMask);
    const modelMove = modelActionIndexToMove(state, modelAction);
    const result = await searchBestMoveHybrid(state, model, {
      timeMs,
      maxDepth: 7,
      topKFromModel: 12
    });
    const next = applyKnownLegalMove(state, result.move);
    decisions.push({
      moveNumber: state.moveNumber,
      modelConfidence: prediction.policy[modelAction],
      modelValue: prediction.value,
      modelMove,
      selectedMove: result.move,
      selectedModelMove: moveToActionIndex(modelMove) === moveToActionIndex(result.move),
      score: result.score,
      depth: result.depth,
      nodes: result.nodes,
      ms: result.ms,
      myPathBefore: shortestPathLength(state, player),
      opponentPathBefore: shortestPathLength(state, opponent),
      myPathAfter: shortestPathLength(next, player),
      opponentPathAfter: shortestPathLength(next, opponent),
      myWalls: state.players[player].wallsLeft,
      opponentWalls: state.players[opponent].wallsLeft
    });
    state = next;
  }

  console.log(JSON.stringify({ game, aiPlayer, winner: getWinner(state), moveNumber: state.moveNumber, decisions }, null, 2));
}

void main();
