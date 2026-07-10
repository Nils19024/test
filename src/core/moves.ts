import { ACTION_COUNT, Move, PlayerIndex, QuoridorState, cloneState, getWinner, inBoard, otherPlayer } from "./state";
import { hasPathToGoal } from "./pathfinding";
import { canMoveBetween, isWallPlacementFree } from "./walls";
import { moveToActionIndex } from "./actions";

const DIRS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1]
] as const;

const DIAGONALS_BY_DIR: Record<string, readonly (readonly [number, number])[]> = {
  "-1,0": [[0, -1], [0, 1]],
  "1,0": [[0, -1], [0, 1]],
  "0,-1": [[-1, 0], [1, 0]],
  "0,1": [[-1, 0], [1, 0]]
};

function sameSquare(a: { row: number; col: number }, row: number, col: number): boolean {
  return a.row === row && a.col === col;
}

function addPawnMove(moves: Move[], seen: Set<string>, row: number, col: number): void {
  const key = `${row},${col}`;
  if (!seen.has(key)) {
    seen.add(key);
    moves.push({ type: "pawn", row, col });
  }
}

export function getLegalPawnMoves(state: QuoridorState, playerIndex: PlayerIndex = state.turn): Move[] {
  const me = state.players[playerIndex];
  const opponent = state.players[otherPlayer(playerIndex)];
  const moves: Move[] = [];
  const seen = new Set<string>();

  for (const [dr, dc] of DIRS) {
    const nr = me.row + dr;
    const nc = me.col + dc;
    if (!canMoveBetween(state, me.row, me.col, nr, nc)) continue;
    if (!sameSquare(opponent, nr, nc)) {
      addPawnMove(moves, seen, nr, nc);
      continue;
    }

    const jumpRow = opponent.row + dr;
    const jumpCol = opponent.col + dc;
    if (canMoveBetween(state, opponent.row, opponent.col, jumpRow, jumpCol)) {
      addPawnMove(moves, seen, jumpRow, jumpCol);
      continue;
    }

    for (const [sdr, sdc] of DIAGONALS_BY_DIR[`${dr},${dc}`]) {
      const diagRow = opponent.row + sdr;
      const diagCol = opponent.col + sdc;
      if (inBoard(diagRow, diagCol) && canMoveBetween(state, opponent.row, opponent.col, diagRow, diagCol)) {
        addPawnMove(moves, seen, diagRow, diagCol);
      }
    }
  }

  return moves;
}

export function isWallLegal(state: QuoridorState, move: Move): boolean {
  if (move.type === "pawn") return false;
  if (state.players[state.turn].wallsLeft <= 0) return false;
  if (!isWallPlacementFree(state, move)) return false;
  const next = cloneState(state);
  if (move.type === "hwall") next.hWalls[move.row][move.col] = true;
  else next.vWalls[move.row][move.col] = true;
  return hasPathToGoal(next, 0) && hasPathToGoal(next, 1);
}

export function getLegalMoves(state: QuoridorState): Move[] {
  if (getWinner(state) !== null) return [];
  const moves = getLegalPawnMoves(state);
  if (state.players[state.turn].wallsLeft <= 0) return moves;

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const hMove: Move = { type: "hwall", row, col };
      if (isWallLegal(state, hMove)) moves.push(hMove);
      const vMove: Move = { type: "vwall", row, col };
      if (isWallLegal(state, vMove)) moves.push(vMove);
    }
  }
  return moves;
}

export function getLegalActionMask(state: QuoridorState): Uint8Array {
  const mask = new Uint8Array(ACTION_COUNT);
  for (const move of getLegalMoves(state)) mask[moveToActionIndex(move)] = 1;
  return mask;
}

export function applyMove(state: QuoridorState, move: Move): QuoridorState {
  const legal = getLegalMoves(state).some((candidate) => moveToActionIndex(candidate) === moveToActionIndex(move));
  if (!legal) throw new Error(`Illegal move: ${JSON.stringify(move)}`);
  return applyKnownLegalMove(state, move);
}

export function applyKnownLegalMove(state: QuoridorState, move: Move): QuoridorState {
  const next = cloneState(state);
  const player = next.players[next.turn];
  if (move.type === "pawn") {
    player.row = move.row;
    player.col = move.col;
  } else if (move.type === "hwall") {
    next.hWalls[move.row][move.col] = true;
    player.wallsLeft--;
  } else {
    next.vWalls[move.row][move.col] = true;
    player.wallsLeft--;
  }
  next.turn = otherPlayer(next.turn);
  next.moveNumber++;
  return next;
}
