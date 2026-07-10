import fs from "node:fs/promises";
import path from "node:path";
import { moveToActionIndex } from "../core/actions";
import { applyMove } from "../core/moves";
import { Move, QuoridorState, createInitialState, getWinner } from "../core/state";
import { GreedyShortestPathAgent } from "../ai/greedyAgent";
import { ModelAgent } from "../ai/modelAgent";
import { TfjsQuoridorPolicyModel } from "../ml/model";
import { loadModelFromDir } from "../ml/exportModel";
import { isMainModule } from "../util/isMain";

export type ReplayStep = {
  ply: number;
  player: 0 | 1;
  agent: string;
  state: QuoridorState;
  move: Move;
  actionIndex: number;
  ms: number;
};

export type ReplayRecord = {
  id: string;
  createdAt: string;
  checkpointGame: number;
  modelPath: string;
  matchup: string;
  initialState: QuoridorState;
  finalState: QuoridorState;
  winner: 0 | 1 | null;
  steps: ReplayStep[];
};

export async function generateModelReplay(config: {
  modelPath: string;
  outDir: string;
  checkpointGame: number;
  maxMoves?: number;
}): Promise<ReplayRecord> {
  const model = await loadModelFromDir(config.modelPath);
  const modelAgent = new ModelAgent(new TfjsQuoridorPolicyModel(model));
  const opponent = GreedyShortestPathAgent;
  const initialState = createInitialState();
  let state = initialState;
  const steps: ReplayStep[] = [];
  const maxMoves = config.maxMoves ?? 140;

  for (let ply = 0; ply < maxMoves && getWinner(state) === null; ply++) {
    const agent = state.turn === 0 ? modelAgent : opponent;
    const started = performance.now();
    const move = await agent.selectMove(state);
    const ms = performance.now() - started;
    steps.push({
      ply,
      player: state.turn,
      agent: agent.name,
      state,
      move,
      actionIndex: moveToActionIndex(move),
      ms
    });
    state = applyMove(state, move);
    if (ply % 20 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const replay: ReplayRecord = {
    id: `replay-game-${config.checkpointGame}-${Date.now()}`,
    createdAt: new Date().toISOString(),
    checkpointGame: config.checkpointGame,
    modelPath: config.modelPath,
    matchup: `${modelAgent.name} vs ${opponent.name}`,
    initialState,
    finalState: state,
    winner: getWinner(state),
    steps
  };

  await fs.mkdir(config.outDir, { recursive: true });
  const fileName = `${replay.id}.json`;
  await fs.writeFile(path.join(config.outDir, fileName), JSON.stringify(replay));
  await fs.writeFile(path.join(config.outDir, "latest.json"), JSON.stringify(replay));
  return replay;
}

function arg(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

if (isMainModule(import.meta.url)) {
  const replay = await generateModelReplay({
    modelPath: arg("modelPath", "models/checkpoints/latest"),
    outDir: arg("outDir", "public/replays"),
    checkpointGame: Number(arg("checkpointGame", "0")),
    maxMoves: Number(arg("maxMoves", "140"))
  });
  console.log(JSON.stringify({ id: replay.id, steps: replay.steps.length, winner: replay.winner }));
}
