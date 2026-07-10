import { BOARD_SIZE, PlayerIndex, QuoridorState } from "./state";
import { canMoveBetween } from "./walls";

const DIRS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1]
] as const;

export function shortestPathLength(state: QuoridorState, playerIndex: PlayerIndex): number {
  const start = state.players[playerIndex];
  const targetRow = playerIndex === 0 ? 0 : BOARD_SIZE - 1;
  const queueRows = new Int8Array(BOARD_SIZE * BOARD_SIZE);
  const queueCols = new Int8Array(BOARD_SIZE * BOARD_SIZE);
  const dist = new Int16Array(BOARD_SIZE * BOARD_SIZE);
  dist.fill(-1);
  let head = 0;
  let tail = 0;
  queueRows[tail] = start.row;
  queueCols[tail] = start.col;
  dist[start.row * BOARD_SIZE + start.col] = 0;
  tail++;

  while (head < tail) {
    const row = queueRows[head];
    const col = queueCols[head];
    head++;
    const currentDist = dist[row * BOARD_SIZE + col];
    if (row === targetRow) return currentDist;
    for (const [dr, dc] of DIRS) {
      const nr = row + dr;
      const nc = col + dc;
      if (!canMoveBetween(state, row, col, nr, nc)) continue;
      const index = nr * BOARD_SIZE + nc;
      if (dist[index] !== -1) continue;
      dist[index] = currentDist + 1;
      queueRows[tail] = nr;
      queueCols[tail] = nc;
      tail++;
    }
  }
  return Number.POSITIVE_INFINITY;
}

export function distanceMapToGoal(state: QuoridorState, playerIndex: PlayerIndex): Int16Array {
  const targetRow = playerIndex === 0 ? 0 : BOARD_SIZE - 1;
  const queueRows = new Int8Array(BOARD_SIZE * BOARD_SIZE);
  const queueCols = new Int8Array(BOARD_SIZE * BOARD_SIZE);
  const dist = new Int16Array(BOARD_SIZE * BOARD_SIZE);
  dist.fill(-1);
  let head = 0;
  let tail = 0;
  for (let col = 0; col < BOARD_SIZE; col++) {
    queueRows[tail] = targetRow;
    queueCols[tail] = col;
    dist[targetRow * BOARD_SIZE + col] = 0;
    tail++;
  }

  while (head < tail) {
    const row = queueRows[head];
    const col = queueCols[head];
    head++;
    const currentDist = dist[row * BOARD_SIZE + col];
    for (const [dr, dc] of DIRS) {
      const nr = row + dr;
      const nc = col + dc;
      if (!canMoveBetween(state, row, col, nr, nc)) continue;
      const index = nr * BOARD_SIZE + nc;
      if (dist[index] !== -1) continue;
      dist[index] = currentDist + 1;
      queueRows[tail] = nr;
      queueCols[tail] = nc;
      tail++;
    }
  }
  return dist;
}

export function hasPathToGoal(state: QuoridorState, playerIndex: PlayerIndex): boolean {
  return Number.isFinite(shortestPathLength(state, playerIndex));
}
