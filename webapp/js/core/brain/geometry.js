/**
 * Geometria pura (array tipizzati, niente three.js): curve di Catmull-Rom, tubi con sezione variabile,
 * ellissoidi, normali e unione di mesh. La usano la forma del cervello (shape.js), le strutture
 * profonde e i percorsi (anatomy.js, pathways.js); three.js riceve solo i buffer finiti (toBufferGeometry
 * in anatomy.js), così la costruzione si può provare in Node (tests/brain.test.mjs).
 *
 * Una mesh qui è { positions: Float32Array (xyz), normals?: Float32Array, indices: Uint32Array, ... }.
 */

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a) => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/**
 * Punto di una Catmull-Rom centripeta (alpha = 0,5: niente cappi né cuspidi tra punti a distanze
 * diverse) per t in [0, 1] su tutta la curva.
 */
export function catmullRom(points, t) {
  const n = points.length;
  if (n === 1) return points[0].slice();
  const segs = n - 1;
  const s = Math.min(segs - 1e-9, Math.max(0, t * segs));
  const i = Math.floor(s);
  const u = s - i;
  const p1 = points[i];
  const p2 = points[i + 1];
  // estremi: punti fantasma riflessi, così la curva passa dal primo e dall'ultimo punto
  const p0 = i > 0 ? points[i - 1] : sub(scale(p1, 2), p2);
  const p3 = i + 2 < n ? points[i + 2] : sub(scale(p2, 2), p1);
  const knot = (a, b) => Math.max(1e-6, Math.sqrt(len(sub(b, a))));
  const t0 = 0;
  const t1 = t0 + knot(p0, p1);
  const t2 = t1 + knot(p1, p2);
  const t3 = t2 + knot(p2, p3);
  const tt = t1 + (t2 - t1) * u;
  const lerp = (a, b, ta, tb) => {
    const w = (tt - ta) / (tb - ta);
    return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
  };
  const a1 = lerp(p0, p1, t0, t1);
  const a2 = lerp(p1, p2, t1, t2);
  const a3 = lerp(p2, p3, t2, t3);
  const b1 = lerp(a1, a2, t0, t2);
  const b2 = lerp(a2, a3, t1, t3);
  return lerp(b1, b2, t1, t2);
}

/**
 * Campiona una curva in `count` punti equidistanti lungo la lunghezza (non lungo il parametro):
 * gli impulsi viaggiano a velocità costante e la sezione del tubo non si stira.
 * @returns {{ points: number[][], lengths: number[], total: number }} lengths = ascissa curvilinea di ogni punto
 */
export function sampleCurve(controlPoints, count) {
  const dense = [];
  const fine = Math.max(count * 4, 32);
  for (let i = 0; i <= fine; i++) dense.push(catmullRom(controlPoints, i / fine));
  const cumulative = [0];
  for (let i = 1; i < dense.length; i++) cumulative.push(cumulative[i - 1] + len(sub(dense[i], dense[i - 1])));
  const total = cumulative[cumulative.length - 1];
  const points = [];
  const lengths = [];
  let j = 1;
  for (let k = 0; k < count; k++) {
    const target = (total * k) / Math.max(1, count - 1);
    while (j < cumulative.length - 1 && cumulative[j] < target) j++;
    const span = cumulative[j] - cumulative[j - 1] || 1;
    const w = Math.min(1, Math.max(0, (target - cumulative[j - 1]) / span));
    const a = dense[j - 1];
    const b = dense[j];
    points.push([a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w]);
    lengths.push(target);
  }
  return { points, lengths, total };
}

/**
 * Tubo lungo una polilinea con sezione ellittica variabile e sistema di riferimento a trasporto
 * parallelo (la sezione non si attorciglia lungo la curva).
 * @param {number[][]} points centri delle sezioni
 * @param {object} o
 * @param {(t: number, angle: number) => number | [number, number]} o.radius raggio (o [rLaterale, rAltro])
 *   in funzione di t ∈ [0, 1] e dell'angolo della sezione (per bozzi e digitazioni)
 * @param {number} [o.radial] lati della sezione
 * @param {number[]} [o.side] direzione preferita del primo asse della sezione (es. [1, 0, 0])
 * @param {boolean} [o.caps] chiude le estremità
 * @param {number[]} [o.lengths] ascissa curvilinea dei punti: diventa l'attributo `arc` (impulsi)
 * @returns {{ positions: Float32Array, normals: Float32Array, indices: Uint32Array, arc: Float32Array }}
 */
export function sweepTube(points, { radius, radial = 12, side = null, caps = true, lengths = null }) {
  const n = points.length;
  const tangents = points.map((p, i) => norm(sub(points[Math.min(n - 1, i + 1)], points[Math.max(0, i - 1)])));
  // primo asse: quello preferito (proiettato sul piano della sezione) o uno qualsiasi perpendicolare
  let u = side ? sub(side, scale(tangents[0], dot(side, tangents[0]))) : cross(tangents[0], Math.abs(tangents[0][1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]);
  if (len(u) < 1e-6) u = cross(tangents[0], [0, 0, 1]);
  u = norm(u);
  const frames = [];
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      // trasporto parallelo: si toglie a u la componente lungo la nuova tangente
      u = norm(sub(u, scale(tangents[i], dot(u, tangents[i]))));
      if (side) {
        // con un asse preferito (es. il corpo calloso largo in x) si torna dolcemente verso di lui
        const want = sub(side, scale(tangents[i], dot(side, tangents[i])));
        if (len(want) > 1e-3) u = norm([u[0] * 0.6 + norm(want)[0] * 0.4, u[1] * 0.6 + norm(want)[1] * 0.4, u[2] * 0.6 + norm(want)[2] * 0.4]);
      }
    }
    const v = norm(cross(tangents[i], u));
    frames.push([u, v]);
  }
  const ring = radial + 1; // un vertice doppio sulla cucitura (normali e coordinate pulite)
  const capVerts = caps ? 2 : 0;
  const positions = new Float32Array((n * ring + capVerts) * 3);
  const normals = new Float32Array((n * ring + capVerts) * 3);
  const arc = new Float32Array(n * ring + capVerts);
  const lengthsOrIndex = lengths ?? points.map((_, i) => i);
  for (let i = 0; i < n; i++) {
    const [fu, fv] = frames[i];
    const t = n > 1 ? i / (n - 1) : 0;
    for (let k = 0; k < ring; k++) {
      const angle = ((k % radial) / radial) * Math.PI * 2;
      const r = radius(t, angle);
      const ru = Array.isArray(r) ? r[0] : r;
      const rv = Array.isArray(r) ? r[1] : r;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      const o = (i * ring + k) * 3;
      positions[o] = points[i][0] + fu[0] * c * ru + fv[0] * s * rv;
      positions[o + 1] = points[i][1] + fu[1] * c * ru + fv[1] * s * rv;
      positions[o + 2] = points[i][2] + fu[2] * c * ru + fv[2] * s * rv;
      // normale dell'ellisse: gradiente (c/ru, s/rv) nel piano della sezione
      const nu = c / Math.max(ru, 1e-6);
      const nv = s / Math.max(rv, 1e-6);
      const nn = norm([fu[0] * nu + fv[0] * nv, fu[1] * nu + fv[1] * nv, fu[2] * nu + fv[2] * nv]);
      normals[o] = nn[0];
      normals[o + 1] = nn[1];
      normals[o + 2] = nn[2];
      arc[i * ring + k] = lengthsOrIndex[i];
    }
  }
  const indices = [];
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < radial; k++) {
      const a = i * ring + k;
      const b = a + ring;
      // antiorario visto da fuori (l'angolo cresce da u verso v = t × u)
      indices.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  if (caps) {
    const first = n * ring;
    const last = first + 1;
    const set = (vi, p, nrm, s) => {
      positions.set(p, vi * 3);
      normals.set(nrm, vi * 3);
      arc[vi] = s;
    };
    set(first, points[0], scale(tangents[0], -1), lengthsOrIndex[0]);
    set(last, points[n - 1], tangents[n - 1], lengthsOrIndex[n - 1]);
    for (let k = 0; k < radial; k++) {
      indices.push(first, k + 1, k);
      const base = (n - 1) * ring;
      indices.push(last, base + k, base + k + 1);
    }
  }
  return { positions, normals, indices: Uint32Array.from(indices), arc };
}

/**
 * Ellissoide (sfera UV deformata) centrato in `center`, con semiassi `radii` e rotazione facoltativa
 * (matrice 3×3 per righe). `bump(nx, ny, nz)` moltiplica il raggio (forme meno perfette).
 */
export function ellipsoidMesh(center, radii, { widthSegments = 24, heightSegments = 16, rotation = null, bump = null } = {}) {
  const positions = [];
  const normals = [];
  for (let iy = 0; iy <= heightSegments; iy++) {
    const v = iy / heightSegments;
    const theta = v * Math.PI;
    for (let ix = 0; ix <= widthSegments; ix++) {
      const phi = (ix / widthSegments) * Math.PI * 2;
      const dx = Math.sin(theta) * Math.cos(phi);
      const dy = Math.cos(theta);
      const dz = Math.sin(theta) * Math.sin(phi);
      const b = bump ? bump(dx, dy, dz) : 1;
      let p = [dx * radii[0] * b, dy * radii[1] * b, dz * radii[2] * b];
      let nrm = norm([dx / radii[0], dy / radii[1], dz / radii[2]]);
      if (rotation) {
        p = mat3(rotation, p);
        nrm = mat3(rotation, nrm);
      }
      positions.push(center[0] + p[0], center[1] + p[1], center[2] + p[2]);
      normals.push(...nrm);
    }
  }
  const indices = [];
  const row = widthSegments + 1;
  for (let iy = 0; iy < heightSegments; iy++) {
    for (let ix = 0; ix < widthSegments; ix++) {
      const a = iy * row + ix;
      const b = a + row;
      if (iy > 0) indices.push(a, a + 1, b);
      if (iy < heightSegments - 1) indices.push(a + 1, b + 1, b);
    }
  }
  const mesh = { positions: Float32Array.from(positions), normals: Float32Array.from(normals), indices: Uint32Array.from(indices) };
  if (bump) computeNormals(mesh); // il bozzo cambia la forma: le normali analitiche non bastano più
  return mesh;
}

/** Matrice 3×3 (per righe) per vettore. */
export function mat3(m, p) {
  return [m[0] * p[0] + m[1] * p[1] + m[2] * p[2], m[3] * p[0] + m[4] * p[1] + m[5] * p[2], m[6] * p[0] + m[7] * p[1] + m[8] * p[2]];
}

/** Rotazione (per righe) attorno a un asse unitario di `angle` radianti. */
export function rotationMatrix(axis, angle) {
  const [x, y, z] = norm(axis);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}

/** Normali per vertice pesate con l'area dei triangoli (in place). */
export function computeNormals(mesh) {
  const { positions, indices } = mesh;
  const normals = new Float32Array(positions.length);
  for (let f = 0; f < indices.length; f += 3) {
    const a = indices[f] * 3;
    const b = indices[f + 1] * 3;
    const c = indices[f + 2] * 3;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const o of [a, b, c]) {
      normals[o] += nx;
      normals[o + 1] += ny;
      normals[o + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const l = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= l;
    normals[i + 1] /= l;
    normals[i + 2] /= l;
  }
  mesh.normals = normals;
  return mesh;
}

/**
 * Unisce più mesh in una (meno oggetti da disegnare: con il rendering software ogni disegno costa).
 * Gli attributi extra per vertice (es. `arc`) si uniscono se ci sono in tutte.
 */
export function mergeMeshes(meshes, extra = []) {
  let vertexCount = 0;
  let indexCount = 0;
  for (const m of meshes) {
    vertexCount += m.positions.length / 3;
    indexCount += m.indices.length;
  }
  const out = { positions: new Float32Array(vertexCount * 3), normals: new Float32Array(vertexCount * 3), indices: new Uint32Array(indexCount) };
  for (const name of extra) out[name] = new Float32Array(vertexCount * (meshes[0][name].length / (meshes[0].positions.length / 3)));
  let v = 0;
  let k = 0;
  for (const m of meshes) {
    const count = m.positions.length / 3;
    out.positions.set(m.positions, v * 3);
    if (m.normals) out.normals.set(m.normals, v * 3);
    for (const name of extra) out[name].set(m[name], v * (m[name].length / count));
    for (let i = 0; i < m.indices.length; i++) out.indices[k + i] = m.indices[i] + v;
    v += count;
    k += m.indices.length;
  }
  return out;
}

/** Box [min, max] di una lista di punti. */
export function boundsOf(positions) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  return { min, max };
}

export const vec = { sub, cross, dot, len, scale, norm };
