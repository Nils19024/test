import { applyKnownLegalMove, getLegalMoves } from "../core/moves";
import { shortestPathLength } from "../core/pathfinding";
import { Move, QuoridorState, PlayerIndex, getWinner, otherPlayer } from "../core/state";
import { evaluateState } from "./evaluate";
import { GreedyShortestPathAgent } from "./greedyAgent";

type PlayerPosition = { row: number; col: number };

export async function selectBestMoveAgainstGreedy(state: QuoridorState, recentPositions: PlayerPosition[] = []): Promise<Move> {
  const legalMoves = getLegalMoves(state);
  const player: PlayerIndex = state.turn;
  const opponent = otherPlayer(player);
  const pawnMoves = legalMoves.filter((move) => move.type === "pawn");
  const wallMoves = legalMoves
    .filter((move) => move.type !== "pawn")
    .sort((a, b) => evaluateState(applyKnownLegalMove(state, b), player) - evaluateState(applyKnownLegalMove(state, a), player))
    .slice(0, 18);
  const candidates = [...pawnMoves, ...wallMoves];
  let bestMove = candidates[0] ?? legalMoves[0];
  let bestScore = -Infinity;

  for (let index = 0; index < candidates.length; index++) {
    const move = candidates[index];
    const next = applyKnownLegalMove(state, move);
    let score: number;
    if (getWinner(next) === player) {
      score = 100000;
    } else {
      const greedyReply = await GreedyShortestPathAgent.selectMove(next);
      const afterReply = applyKnownLegalMove(next, greedyReply);
      const ownBefore = shortestPathLength(state, player);
      const ownAfter = shortestPathLength(afterReply, player);
      const oppBefore = shortestPathLength(state, opponent);
      const oppAfter = shortestPathLength(afterReply, opponent);
      const pathProgress = Number.isFinite(ownBefore) && Number.isFinite(ownAfter) ? ownBefore - ownAfter : 0;
      const blockProgress = Number.isFinite(oppBefore) && Number.isFinite(oppAfter) ? oppAfter - oppBefore : 0;
      const afterPosition = next.players[player];
      const repeated = move.type === "pawn" && recentPositions.some((position) => position.row === afterPosition.row && position.col === afterPosition.col);
      score = evaluateState(afterReply, player)
        + pathProgress * 24
        + blockProgress * 10
        - (move.type === "pawn" ? 0 : 4)
        - (repeated ? 50 : 0);
    }
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
    if (index % 8 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return bestMove;
}
