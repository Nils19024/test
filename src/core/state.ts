export type PlayerIndex = 0 | 1;

export type Player = {
  row: number;
  col: number;
  wallsLeft: number;
};

export type QuoridorState = {
  turn: PlayerIndex;
  players: [Player, Player];
  hWalls: boolean[][];
  vWalls: boolean[][];
  moveNumber: number;
};

export type Move =
  | { type: "pawn"; row: number; col: number }
  | { type: "hwall"; row: number; col: number }
  | { type: "vwall"; row: number; col: number };

export const BOARD_SIZE = 9;
export const WALL_GRID_SIZE = 8;
export const ACTION_COUNT = 209;

export function createEmptyWalls(): boolean[][] {
  return Array.from({ length: WALL_GRID_SIZE }, () => Array(WALL_GRID_SIZE).fill(false));
}

export function createInitialState(): QuoridorState {
  return {
    turn: 0,
    players: [
      { row: 8, col: 4, wallsLeft: 10 },
      { row: 0, col: 4, wallsLeft: 10 }
    ],
    hWalls: createEmptyWalls(),
    vWalls: createEmptyWalls(),
    moveNumber: 0
  };
}

export function cloneState(state: QuoridorState): QuoridorState {
  return {
    turn: state.turn,
    players: [
      { ...state.players[0] },
      { ...state.players[1] }
    ],
    hWalls: state.hWalls.map((row) => row.slice()),
    vWalls: state.vWalls.map((row) => row.slice()),
    moveNumber: state.moveNumber
  };
}

export function otherPlayer(player: PlayerIndex): PlayerIndex {
  return player === 0 ? 1 : 0;
}

export function inBoard(row: number, col: number): boolean {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

export function inWallGrid(row: number, col: number): boolean {
  return row >= 0 && row < WALL_GRID_SIZE && col >= 0 && col < WALL_GRID_SIZE;
}

export function getWinner(state: QuoridorState): PlayerIndex | null {
  if (state.players[0].row === 0) return 0;
  if (state.players[1].row === BOARD_SIZE - 1) return 1;
  return null;
}
