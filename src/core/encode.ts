import { BOARD_SIZE, QuoridorState, createEmptyWalls, otherPlayer } from "./state";
import { distanceMapToGoal, shortestPathLength } from "./pathfinding";

export type NormalizedState = QuoridorState;

function rotateIndex(value: number): number {
  return BOARD_SIZE - 1 - value;
}

function rotateWallIndex(value: number): number {
  return 7 - value;
}

export function normalizeStateForCurrentPlayer(state: QuoridorState): NormalizedState {
  if (state.turn === 0) return state;
  const hWalls = createEmptyWalls();
  const vWalls = createEmptyWalls();
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      hWalls[rotateWallIndex(row)][rotateWallIndex(col)] = state.hWalls[row][col];
      vWalls[rotateWallIndex(row)][rotateWallIndex(col)] = state.vWalls[row][col];
    }
  }
  return {
    turn: 0,
    players: [
      {
        row: rotateIndex(state.players[1].row),
        col: rotateIndex(state.players[1].col),
        wallsLeft: state.players[1].wallsLeft
      },
      {
        row: rotateIndex(state.players[0].row),
        col: rotateIndex(state.players[0].col),
        wallsLeft: state.players[0].wallsLeft
      }
    ],
    hWalls,
    vWalls,
    moveNumber: state.moveNumber
  };
}

function setChannel(data: Float32Array, row: number, col: number, channel: number, value: number): void {
  data[(row * BOARD_SIZE + col) * 12 + channel] = value;
}

export function encodeStateForModel(state: QuoridorState): Float32Array {
  state = normalizeStateForCurrentPlayer(state);
  const meIndex = state.turn;
  const oppIndex = otherPlayer(meIndex);
  const ownDist = shortestPathLength(state, meIndex);
  const oppDist = shortestPathLength(state, oppIndex);
  const encoded = new Float32Array(BOARD_SIZE * BOARD_SIZE * 12);
  const ownDistanceMap = distanceMapToGoal(state, meIndex);
  const oppDistanceMap = distanceMapToGoal(state, oppIndex);

  setChannel(encoded, state.players[meIndex].row, state.players[meIndex].col, 0, 1);
  setChannel(encoded, state.players[oppIndex].row, state.players[oppIndex].col, 1, 1);
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      if (state.hWalls[row][col]) {
        setChannel(encoded, row, col, 2, 1);
        setChannel(encoded, row, col + 1, 2, 1);
      }
      if (state.vWalls[row][col]) {
        setChannel(encoded, row, col, 3, 1);
        setChannel(encoded, row + 1, col, 3, 1);
      }
    }
  }

  const ownTargetRow = meIndex === 0 ? 0 : BOARD_SIZE - 1;
  const oppTargetRow = oppIndex === 0 ? 0 : BOARD_SIZE - 1;
  for (let col = 0; col < BOARD_SIZE; col++) {
    setChannel(encoded, ownTargetRow, col, 4, 1);
    setChannel(encoded, oppTargetRow, col, 5, 1);
  }

  const ownWalls = state.players[meIndex].wallsLeft / 10;
  const oppWalls = state.players[oppIndex].wallsLeft / 10;
  const ownDistNorm = Math.min(ownDist, 32) / 32;
  const oppDistNorm = Math.min(oppDist, 32) / 32;
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      setChannel(encoded, row, col, 6, ownWalls);
      setChannel(encoded, row, col, 7, oppWalls);
      setChannel(encoded, row, col, 8, ownDistNorm);
      setChannel(encoded, row, col, 9, oppDistNorm);
      const ownCellDist = ownDistanceMap[row * BOARD_SIZE + col];
      const oppCellDist = oppDistanceMap[row * BOARD_SIZE + col];
      setChannel(encoded, row, col, 10, ownCellDist >= 0 ? Math.min(ownCellDist, 32) / 32 : 1);
      setChannel(encoded, row, col, 11, oppCellDist >= 0 ? Math.min(oppCellDist, 32) / 32 : 1);
    }
  }

  return encoded;
}
