// Push-optimal Sokoban solver for make.fun's sokoban-solver-01.
// Plain JavaScript (runs under Node 22's type stripping). Deterministic: every search is budgeted by state counts,
// never by the clock, and every expansion order is fixed.
//
// Approach: split the level into rooms (the hallway's dead squares cut the board), solve each room with a
// breadth-first search over pushes (so its first solution is push-optimal), with the other rooms' boxes as walls,
// and chain the rooms' move strings. Fallback: a move-based A* over the whole level.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const ROOM_STATE_CAP = 2000000; // per room of up to 6 boxes
const BIG_ROOM_STATE_CAP = 200000; // whole-level fallback room with more boxes (memory bound)
const ASTAR_STORE_CAP = 1000000; // A* stored-state bound (memory: ~300 B a state, well under 512 MB)
const ASTAR_MAX_BOXES = 6; // move-based A* is hopeless (and memory-hungry) on more boxes; skip it
const ASTAR_CAP = 2000000;
const LETTERS = ['u', 'd', 'l', 'r'];
// One work budget for the whole run (the 60 s wall is for the whole set), counted in flood-filled squares (A*: a fixed
// charge per expansion), never by the clock. A generated set uses ~1e7 of it; a generated room at most ~1.3e8.
const ASTAR_EXPANSION_COST = 100;
let workLeft = 4e8;

// ---------- level ----------

function parse(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  if (rows.length === 0) throw new Error('empty level');
  const Hr = rows.length;
  const Wr = Math.max(...rows.map((r) => r.length));
  // Pad one wall cell on every side so index arithmetic never leaves the board; cells missing from short rows are
  // walls, as in the checker.
  const W = Wr + 2, H = Hr + 2, N = W * H;
  const wall = new Uint8Array(N).fill(1);
  const goal = new Uint8Array(N);
  const boxes = [];
  let player = -1;
  for (let y = 0; y < Hr; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = (y + 1) * W + (x + 1);
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  if (player < 0) throw new Error('no player');
  // The level's outermost frame is wall too (as the generator's parser reads it), unless something stands on it.
  const busy = new Uint8Array(N);
  for (const b of boxes) busy[b] = 1;
  busy[player] = 1;
  const frame = (i) => { if (!busy[i] && !goal[i]) wall[i] = 1; };
  for (let x = 1; x <= Wr; x++) { frame(W + x); frame(Hr * W + x); }
  for (let y = 1; y <= Hr; y++) { frame(y * W + 1); frame(y * W + Wr); }
  const goalsList = [];
  for (let i = 0; i < N; i++) if (goal[i]) goalsList.push(i);
  return { W, H, N, wall, goal, goalsList, boxes, player, dirs: [-W, W, -1, 1] };
}

// Live squares: a lone box there can be pushed onto some goal (reverse pulls from every goal over open floor).
function liveSquares(L) {
  const { N, wall, goalsList, dirs } = L;
  const live = new Uint8Array(N);
  const q = [];
  for (const g of goalsList) if (!live[g]) { live[g] = 1; q.push(g); }
  for (let h = 0; h < q.length; h++) {
    const b = q[h];
    for (const d of dirs) {
      // pull: player at b-d steps to b-2d, box from b to b-d
      const nb = b - d, np = b - 2 * d;
      if (np < 0 || np >= N) continue;
      if (wall[nb] || wall[np] || live[nb]) continue;
      live[nb] = 1;
      q.push(nb);
    }
  }
  return live;
}

// Rooms: cut the dead squares of the player's row and flood-fill the rest; components holding boxes are rooms.
function decompose(L, live) {
  const { W, N, wall, goal, boxes, player, dirs } = L;
  const row = Math.floor(player / W);
  const cut = new Uint8Array(N);
  for (let x = 0; x < W; x++) {
    const i = row * W + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  const comp = new Int32Array(N).fill(-1);
  const comps = [];
  for (let s = 0; s < N; s++) {
    if (wall[s] || cut[s] || comp[s] >= 0) continue;
    const id = comps.length;
    const cells = [s];
    comp[s] = id;
    for (let h = 0; h < cells.length; h++) {
      const c = cells[h];
      for (const d of dirs) {
        const n = c + d;
        if (!wall[n] && !cut[n] && comp[n] < 0) { comp[n] = id; cells.push(n); }
      }
    }
    comps.push(cells);
  }
  const whole = [boxes.slice().sort((a, b) => a - b)];
  const boxCount = new Array(comps.length).fill(0);
  const goalCount = new Array(comps.length).fill(0);
  for (const b of boxes) {
    if (cut[b]) return whole;
    boxCount[comp[b]]++;
  }
  for (let i = 0; i < N; i++) if (goal[i]) {
    if (comp[i] < 0) return whole;
    goalCount[comp[i]]++;
  }
  const rooms = [];
  for (let k = 0; k < comps.length; k++) {
    if (boxCount[k] !== goalCount[k]) return whole;
    if (boxCount[k] > 0) rooms.push(boxes.filter((b) => comp[b] === k).sort((a, b) => a - b));
  }
  return rooms.length > 0 ? rooms : whole; // comps are numbered in scan order
}

// ---------- per-room push BFS ----------

function solveRoom(L, live, obstacle, roomBoxes, start, cap) {
  const { N, goal, dirs } = L;
  const k = roomBoxes.length;
  const occ = new Uint8Array(N); // walls + foreign boxes (static) + room boxes (dynamic)
  for (let i = 0; i < N; i++) occ[i] = obstacle[i];
  const isBox = new Uint8Array(N);
  const mark = new Int32Array(N);
  const hmark = new Int32Array(N); // head state's player region, tagged head+1
  let stamp = 0;
  const reach = [];
  const useNum = Math.pow(N, k + 1) < 9007199254740991;
  const keyOf = (bx, p) => {
    if (useNum) { let v = 0; for (let j = 0; j < k; j++) v = v * N + bx[j]; return v * N + p; }
    return bx.join(',') + ':' + p;
  };
  // flood-fill the player's region; returns its lowest square; leaves reach[] and mark[] filled
  const flood = (p) => {
    stamp++;
    reach.length = 0;
    reach.push(p);
    mark[p] = stamp;
    let lo = p;
    for (let h = 0; h < reach.length; h++) {
      const c = reach[h];
      for (let j = 0; j < 4; j++) {
        const n = c + dirs[j];
        if (occ[n] || mark[n] === stamp) continue;
        mark[n] = stamp;
        reach.push(n);
        if (n < lo) lo = n;
      }
    }
    workLeft -= reach.length;
    return lo;
  };
  const solved = (bx) => { for (let j = 0; j < k; j++) if (!goal[bx[j]]) return false; return true; };
  // frozen 2x2: some 2x2 block containing square b made of obstacles/boxes with a box off its goal
  const frozen2x2 = (b) => {
    for (const [dx, dy] of [[-1, -L.W], [-1, 0], [0, -L.W], [0, 0]]) {
      const a = b + dx + dy;
      const cells = [a, a + 1, a + L.W, a + L.W + 1];
      let ok = true, offGoal = false;
      for (const c of cells) {
        if (!occ[c]) { ok = false; break; }
        if (isBox[c] && !goal[c]) offGoal = true;
      }
      if (ok && offGoal) return true;
    }
    return false;
  };

  const startBoxes = roomBoxes.slice().sort((a, b) => a - b);
  if (solved(startBoxes)) return [];
  // state storage
  const boxStore = new Int32Array(cap * k);
  const repStore = new Int32Array(cap);
  const parent = new Int32Array(cap);
  const pushFrom = new Int32Array(cap);
  const pushDir = new Int8Array(cap);
  const seen = new Map();
  const setBoxes = (bx, on) => { for (let j = 0; j < k; j++) { occ[bx[j]] = on; isBox[bx[j]] = on; } };
  setBoxes(startBoxes, 1);
  const rep0 = flood(start);
  setBoxes(startBoxes, 0);
  for (let j = 0; j < k; j++) boxStore[j] = startBoxes[j];
  repStore[0] = rep0;
  parent[0] = -1;
  seen.set(keyOf(startBoxes, rep0), 0);
  let count = 1;
  let goalIdx = -1;
  const bx = new Array(k);
  const nbx = new Array(k);
  outer:
  for (let head = 0; head < count; head++) {
    if (workLeft < 0) return null;
    for (let j = 0; j < k; j++) bx[j] = boxStore[head * k + j];
    setBoxes(bx, 1);
    flood(repStore[head]);
    const tag = head + 1;
    for (let r = 0; r < reach.length; r++) hmark[reach[r]] = tag;
    for (let j = 0; j < k; j++) {
      const b = bx[j];
      for (let dIdx = 0; dIdx < 4; dIdx++) {
        const d = dirs[dIdx];
        const stand = b - d, to = b + d;
        if (hmark[stand] !== tag) continue;
        if (occ[to] || !live[to]) continue;
        // apply push
        occ[b] = 0; isBox[b] = 0; occ[to] = 1; isBox[to] = 1;
        let pruned = frozen2x2(to);
        let key = null, rep = 0;
        if (!pruned) {
          for (let m = 0; m < k; m++) nbx[m] = m === j ? to : bx[m];
          nbx.sort((a, c) => a - c);
          rep = flood(b);
          key = keyOf(nbx, rep);
          if (seen.has(key)) pruned = true;
        }
        // undo push
        occ[to] = 0; isBox[to] = 0; occ[b] = 1; isBox[b] = 1;
        if (pruned) continue;
        if (count >= cap) { setBoxes(bx, 0); return null; }
        const id = count++;
        for (let m = 0; m < k; m++) boxStore[id * k + m] = nbx[m];
        repStore[id] = rep;
        parent[id] = head;
        pushFrom[id] = b;
        pushDir[id] = dIdx;
        seen.set(key, id);
        if (solved(nbx)) { goalIdx = id; setBoxes(bx, 0); break outer; }
      }
    }
    setBoxes(bx, 0);
  }
  if (goalIdx < 0) return null;
  const pushes = [];
  for (let s = goalIdx; parent[s] >= 0; s = parent[s]) pushes.push([pushFrom[s], pushDir[s]]);
  pushes.reverse();
  return pushes;
}

// ---------- move strings ----------

// shortest walk (BFS, fixed direction order) from p to t with occ as obstacles; returns letters or null
function walk(L, occ, p, t) {
  if (p === t) return '';
  const { N, dirs } = L;
  const prev = new Int32Array(N).fill(-1);
  const how = new Int8Array(N);
  prev[p] = p;
  const q = [p];
  for (let h = 0; h < q.length; h++) {
    const c = q[h];
    for (let j = 0; j < 4; j++) {
      const n = c + dirs[j];
      if (occ[n] || prev[n] >= 0) continue;
      prev[n] = c; how[n] = j; q.push(n);
      if (n === t) {
        let s = '';
        for (let x = t; x !== p; x = prev[x]) s = LETTERS[how[x]] + s;
        return s;
      }
    }
  }
  return null;
}

// ---------- fallback: move-based A* over the whole level ----------

function astar(L, cap) {
  if (workLeft < 0) return null;
  const { N, goal, goalsList, wall, dirs, W } = L;
  const live = liveSquares(L);
  const k = L.boxes.length;
  const gx = goalsList.map((g) => g % W), gy = goalsList.map((g) => Math.floor(g / W));
  const hDist = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    let best = 1e9;
    const x = i % W, y = Math.floor(i / W);
    for (let j = 0; j < gx.length; j++) best = Math.min(best, Math.abs(x - gx[j]) + Math.abs(y - gy[j]));
    hDist[i] = best;
  }
  const keyOf = (bx, p) => bx.join(',') + ':' + p;
  const states = []; // {bx, p, g, parent, move}
  const heap = []; // [f, seq, idx]
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const push = (e) => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) { const pi = (i - 1) >> 1; if (less(heap[i], heap[pi])) { [heap[i], heap[pi]] = [heap[pi], heap[i]]; i = pi; } else break; }
  };
  const pop = () => {
    const top = heap[0], last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && less(heap[l], heap[m])) m = l;
        if (r < heap.length && less(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]]; i = m;
      }
    }
    return top;
  };
  const hOf = (bx) => { let s = 0; for (const b of bx) s += hDist[b]; return s; };
  const best = new Map();
  const b0 = L.boxes.slice().sort((a, b) => a - b);
  states.push({ bx: b0, p: L.player, g: 0, parent: -1, move: '' });
  best.set(keyOf(b0, L.player), 0);
  let seq = 0;
  push([hOf(b0), seq++, 0]);
  let expanded = 0;
  while (heap.length > 0 && expanded < cap && states.length < ASTAR_STORE_CAP && workLeft >= 0) {
    const [, , idx] = pop();
    const s = states[idx];
    if (best.get(keyOf(s.bx, s.p)) < s.g) continue;
    if (s.bx.every((b) => goal[b])) {
      let out = '';
      for (let x = idx; states[x].parent >= 0; x = states[x].parent) out = states[x].move + out;
      return out;
    }
    expanded++;
    workLeft -= ASTAR_EXPANSION_COST;
    const boxSet = new Set(s.bx);
    for (let j = 0; j < 4; j++) {
      const d = dirs[j];
      const n = s.p + d;
      if (wall[n]) continue;
      let nbx = s.bx, mv = LETTERS[j];
      if (boxSet.has(n)) {
        const to = n + d;
        if (wall[to] || boxSet.has(to) || !live[to]) continue;
        nbx = s.bx.map((b) => (b === n ? to : b)).sort((a, b) => a - b);
        mv = mv.toUpperCase();
      }
      const key = keyOf(nbx, n);
      const g = s.g + 1;
      const old = best.get(key);
      if (old !== undefined && old <= g) continue;
      best.set(key, g);
      states.push({ bx: nbx, p: n, g, parent: idx, move: mv });
      push([g + hOf(nbx), seq++, states.length - 1]);
    }
  }
  void k;
  return null;
}

// ---------- replay (verification) ----------

function replay(L, moves) {
  const { N, wall, goal, W } = L;
  const box = new Uint8Array(N);
  for (const b of L.boxes) box[b] = 1;
  let p = L.player, pushes = 0;
  const step = { u: -W, d: W, l: -1, r: 1 };
  for (const ch of moves) {
    const d = step[ch.toLowerCase()];
    const n = p + d;
    if (wall[n]) return { ok: false, pushes };
    if (box[n]) {
      if (wall[n + d] || box[n + d]) return { ok: false, pushes };
      box[n] = 0; box[n + d] = 1; pushes++;
    }
    p = n;
  }
  for (let i = 0; i < N; i++) if (box[i] && !goal[i]) return { ok: false, pushes };
  return { ok: true, pushes };
}

// ---------- top level ----------

function solveRooms(L) {
  const live = liveSquares(L);
  const rooms = decompose(L, live);
  const { N, wall, goal, dirs } = L;
  const current = new Uint8Array(N); // where every box stands now
  for (const b of L.boxes) current[b] = 1;
  let player = L.player;
  let out = '';
  for (const room of rooms) {
    const inRoom = new Set(room);
    const obstacle = new Uint8Array(N);
    for (let i = 0; i < N; i++) obstacle[i] = wall[i] || (current[i] && !inRoom.has(i)) ? 1 : 0;
    const pushes = solveRoom(L, live, obstacle, room, player, room.length <= 6 ? ROOM_STATE_CAP : BIG_ROOM_STATE_CAP);
    if (pushes === null) return null;
    // emit moves for this room
    const occ = new Uint8Array(N);
    for (let i = 0; i < N; i++) occ[i] = wall[i] || current[i] ? 1 : 0;
    for (const [from, dIdx] of pushes) {
      const d = dirs[dIdx];
      const w = walk(L, occ, player, from - d);
      if (w === null) throw new Error('walk failed');
      out += w + LETTERS[dIdx].toUpperCase();
      occ[from] = 0; occ[from + d] = 1;
      current[from] = 0; current[from + d] = 1;
      player = from;
    }
    void goal;
  }
  return out;
}

function solve(text) {
  const L = parse(text);
  let moves = null;
  try { moves = solveRooms(L); } catch { moves = null; }
  if (moves !== null && replay(L, moves).ok) return moves;
  const alt = L.boxes.length <= ASTAR_MAX_BOXES ? astar(L, ASTAR_CAP) : null;
  if (alt !== null && replay(L, alt).ok) return alt;
  return '';
}

const ids = readdirSync(I).sort();
// Write an empty (valid) answer for every instance first, so that a fatal error later (e.g. an out-of-memory abort,
// which no try/catch can stop) still leaves every instance answered.
const write = (id, moves) => {
  try {
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  } catch {}
};
for (const id of ids) write(id, '');
for (const id of ids) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  if (typeof moves !== 'string' || moves.length > 65535) moves = '';
  write(id, moves);
}
