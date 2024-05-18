import numpy as np
from typing import Optional, Callable
import drawsvg as draw
import sys
from queue import Queue

EPS = 1e-9

def inv2by2(x: np.ndarray) -> Optional[np.ndarray]:
  (a, b), (c, d) = x
  denom = a * d - b * c
  if abs(denom) < EPS:
    return None
  return (1 / denom) * np.array([[d, -b], [-c, a]])

def in01(x: float) -> float:
  return EPS < x and x < 1 - EPS

def in01exact(x: float) -> float:
  return -EPS <= x and x <= 1 + EPS

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Optional[np.ndarray]:
  da = a2 - a1
  db = b2 - b1
  dp = b1 - a1

  T = np.vstack([da, -db])
  inv = inv2by2(T)

  if inv is None:
    return None
  t, s = dp @ inv

  if (in01(t) and in01exact(s)) or (in01exact(t) and in01(s)):
  # if in01(t) and in01(s):
    return da * t + a1
  return None

def barycentric(v1, v2, v3, vt):
  v13 = v1 - v3
  v23 = v2 - v3
  vt3 = vt - v3
  T = np.vstack([v13, v23])
  T_inv = np.linalg.inv(T)
  return vt3 @ T_inv

def interp(v1, v2, v3, uv):
  u, v = uv
  return u * v1 + v * v2 + (1 - u - v) * v3

def topo_sort(x: list[int], comp: Callable[[int, int], int]) -> list[int]:
  n = len(x)
  adj = [[] for _ in range(n)]
  indeg = [0 for _ in range(n)]
  q = Queue()
  for i in range(n):
    for j in range(i + 1, n):
      c = comp(x[i], x[j])
      if c < 0:
        adj[i].append(j)
        indeg[j] += 1
      elif c > 0:
        adj[j].append(i)
        indeg[i] += 1
  for i in range(n):
    if indeg[i] == 0:
      q.put(i)
  res = []
  error = 0
  while len(res) < n:
    if q.empty():
      mn = n + 1
      mn_idx = -1
      for i in range(n):
        if indeg[i] > 0 and indeg[i] < mn:
          mn = indeg[i]
          mn_idx = i
      assert(mn_idx != -1)
      error += mn
      indeg[mn_idx] = 0
      q.put(mn_idx)
    while not q.empty():
      i = q.get()
      res.append(x[i])
      indeg[i] = -1
      for j in adj[i]:
        indeg[j] -= 1
        if indeg[j] == 0:
          q.put(j)
  if error > 0:
    print(f"Warning: {error} conflicts found", file=sys.stderr)
  return res

def minmax(a, b):
  return (a, b) if a < b else (b, a)

def render(V: np.ndarray, F: np.ndarray, F_id: np.ndarray, cam_pose: np.ndarray, mag: float, size: float):
  V_t = np.linalg.pinv(cam_pose).dot(np.concatenate([V, np.ones((len(V), 1))], axis=1).T)[:3].T
  V_t = (V_t / mag) * (size / 2)

  def comp(a, b):
    ai = F.T[a]
    bi = F.T[b]
    
    v_ai: tuple[np.ndarray, np.ndarray, np.ndarray] = tuple(V_t[i] for i in ai)
    v_bi: tuple[np.ndarray, np.ndarray, np.ndarray] = tuple(V_t[i] for i in bi)

    if (np.maximum(v_ai[0], np.maximum(v_ai[1], v_ai[2])) < np.minimum(v_bi[0], np.maximum(v_bi[1], v_bi[2]))).all() or \
       (np.maximum(v_bi[0], np.maximum(v_bi[1], v_bi[2])) < np.minimum(v_ai[0], np.maximum(v_ai[1], v_ai[2]))).all():
      return 0

    v_ai_2d = tuple(V_t[i][:2] for i in ai)
    v_bi_2d = tuple(V_t[i][:2] for i in bi)

    v_ai_z = tuple(V_t[i][2] for i in ai)
    v_bi_z = tuple(V_t[i][2] for i in bi)

    intersections = []
    for i in range(3):
      for j in range(3):
        intersection = seg_intersect(v_ai_2d[i], v_ai_2d[(i + 1) % 3], v_bi_2d[j], v_bi_2d[(j + 1) % 3])
        if intersection is not None:
          intersections.append(intersection)
    
    v_b_2d_mean = (v_bi_2d[0] + v_bi_2d[1] + v_bi_2d[2]) / 3
    u, v = barycentric(*v_ai_2d, v_b_2d_mean)
    if in01(u) and in01(v) and in01(1 - u - v):
      intersections.append(v_b_2d_mean)
    v_a_2d_mean = (v_ai_2d[0] + v_ai_2d[1] + v_ai_2d[2]) / 3
    u, v = barycentric(*v_bi_2d, v_a_2d_mean)
    if in01(u) and in01(v) and in01(1 - u - v):
      intersections.append(v_a_2d_mean)
    
    if len(intersections) == 0:
      return 0
    
    mean_intersection = np.array(intersections).mean(axis=0)

    uva = barycentric(*v_ai_2d, mean_intersection)
    uvb = barycentric(*v_bi_2d, mean_intersection)

    # assert(in01exact(uva[0]) and in01exact(uva[1]) and in01exact(1 - uva[0] - uva[1]))
    # assert(in01exact(uvb[0]) and in01exact(uvb[1]) and in01exact(1 - uvb[0] - uvb[1]))

    za = interp(*v_ai_z, uva)
    zb = interp(*v_bi_z, uvb)

    if za < zb - EPS:
      return -1
    elif zb < za - EPS:
      return 1
    else:
      return 0

  # F_i = sorted(list(range(len(F.T))), key=cmp_to_key(comp), reverse=True)
  n_f = len(F.T)
  F_i = topo_sort(list(range(n_f)), comp)
  assert(len(F_i) == n_f)
  F_ord = [-1 for _ in range(n_f)]
  for i in range(n_f):
    F_ord[F_i[i]] = i

  F_id = F_id[0]
  n_sf = F_id.max() + 1
  sf_child = [[] for _ in range(n_sf)]
  for i in range(len(F_id)):
    sf_child[F_id[i]].append(i)

  excluded = set()
  for ii in range(n_sf):
    seen = set()
    for f in sf_child[ii]:
      vs = F.T[f]
      for jj in range(3):
        e = minmax(vs[jj], vs[(jj + 1) % 3])
        if (e in seen):
          excluded.add(e)
        else:
          seen.add(e)

  d = draw.Drawing(size, size, origin='center', displayInline=False)

  for ii in F_i:
    i, j, k = F.T[ii]
    vi = V_t[i][:2]
    vj = V_t[j][:2]
    vk = V_t[k][:2]

    front = np.cross(vj - vi, vk - vi) > 0
    surface = draw.Lines(*vi, *vj, *vk, close=True, fill='#eeee00', fill_opacity='0.5')

    edges = []
    vs = F.T[ii]
    for jj in range(3):
      e = minmax(vs[jj], vs[(jj + 1) % 3])
      if e not in excluded:
        edges.append(draw.Line(*V_t[vs[jj]][:2], *V_t[vs[(jj + 1) % 3]][:2], stroke='black', stroke_width=2))
    
    if front:
      d.append(surface)
      d.extend(edges)
    else:
      d.extend(edges)
      d.append(surface)
  return d