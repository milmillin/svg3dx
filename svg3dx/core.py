from dataclasses import dataclass
import numpy as np
from typing_extensions import Self, NamedTuple, Type, Optional, Sequence
from beartype.typing import Tuple, List, Set
import pyrr
import drawsvg as draw
from sortedcontainers import SortedList, sortedlist
from queue import PriorityQueue
from collections import deque
import shapely
from shapely import Polygon, STRtree, GeometryCollection
import sys
from abc import ABC, abstractmethod

from .typing import NpVec3f, Number, InVec3f, NpMat44f, NpMatX2i, NpMatX3f, NpMatX3i, NpVecXi
from .utils import rgb2hex, normalize


class Camera(NamedTuple):
    width: Number
    height: Number
    position: NpVec3f
    projection: NpMat44f  # (4, 4)
    to_viewport: NpMat44f  # (4, 4)

    def transform_V(self: Self, V: NpMatX3f) -> NpMatX3f:
        """
        V: (n_V, 3)
        returns: (n_V, 3)
        """
        padding = np.ones((len(V), 1))
        xyzw = np.concatenate([V, padding], axis=-1) @ self.projection
        xyz = xyzw[..., :3]
        w = xyzw[..., 3:]
        xyz /= w
        return (np.concatenate([xyz, padding], axis=-1) @ self.to_viewport)[..., :3]

    @classmethod
    def create_perspective(
        cls,
        width: Number,
        height: Number,
        eye: InVec3f = [10, 0, 0],
        target: InVec3f = [0, 0, 0],
        up: InVec3f = [0, 1, 0],
        fovy: Number = 15,
        near: Number = 1,
        far: Number = 10,
    ) -> Self:
        """
        width: render width
        height: render height
        eye: (3,)
        target: (3,)
        up: (3,)
        fovy: field of view in y direction in degrees
        near: distance from the viewer to the near clipping plane (only positive)
        far: distance from the viewer to the far clipping plane (only positive)
        """
        view = pyrr.matrix44.create_look_at(eye, target, up)
        proj = pyrr.matrix44.create_perspective_projection(fovy, width / height, near, far)
        scale = pyrr.matrix44.create_from_scale([width / 2, -height / 2, 1])
        trans = pyrr.matrix44.create_from_translation([width / 2, height / 2, 0])
        return cls(
            width=width,
            height=height,
            position=np.asarray(eye, dtype=np.float64),
            projection=view @ proj,
            to_viewport=scale @ trans,
        )

    @classmethod
    def create_orthogonal(
        cls,
        width: Number,
        height: Number,
        eye: InVec3f = [10, 0, 0],
        target: InVec3f = [0, 0, 0],
        up: InVec3f = [0, 1, 0],
        dist: Number = 10,
        near: Number = 1,
        far: Number = 10,
    ):
        ratio = width / height
        view = pyrr.matrix44.create_look_at(eye, target, up)
        proj = pyrr.matrix44.create_orthogonal_projection(-dist * ratio, dist * ratio, -dist, dist, near, far)
        scale = pyrr.matrix44.create_from_scale([width / 2, -height / 2, 1])
        trans = pyrr.matrix44.create_from_translation([width / 2, height / 2, 0])
        return cls(
            width=width,
            height=height,
            position=np.asarray(eye, dtype=np.float64),
            projection=view @ proj,
            to_viewport=scale @ trans,
        )


class Mesh(NamedTuple):
    V: NpMatX3f  # (n_V, 3) list of vertices
    F: NpMatX3i  # (n_F, 3) list of faces, indexed into V
    E: Optional[NpMatX2i] = None  # (n_E, 2) list of edges, indexed into V
    C: Optional[NpMatX3f] = None  # (n_F, 3) per face color


class Shader(ABC):
    @abstractmethod
    def shade(self, FV: NpMatX3f, FN: NpMatX3f, FC: NpMatX3f, campos: NpVec3f) -> NpMatX3f:
        # FV: (n_F, 3) face centroid
        # FN: (n_F, 3) face normal
        # FC: (n_F, 3) face base color
        # campos: (3,) camera position
        # returns (n_F, 3) face shaded color
        ...


class ConstantShader(Shader):
    def shade(self, FV: NpMatX3f, FN: NpMatX3f, FC: NpMatX3f, campos: NpVec3f) -> NpMatX3f:
        return FC


class DiffuseShader(Shader):
    def shade(self, FV: NpMatX3f, FN: NpMatX3f, FC: NpMatX3f, campos: NpVec3f) -> NpMatX3f:
        L = normalize(campos - FV)  # (n_F, 3)
        return np.absolute((L * FN).sum(axis=-1, keepdims=True)) * FC


def order_faces(mesh: Mesh, camera: Camera) -> NpVecXi:
    V = camera.transform_V(mesh.V)
    return _sort_back_to_front2(V, mesh.F)


def render(
    mesh: Mesh,
    camera: Camera,
    shader: Shader = ConstantShader(),
    opacity: Number = 0.5,
    face_order: Optional[NpVecXi] = None,
) -> draw.Drawing:
    V = camera.transform_V(mesh.V)
    if face_order is None:
        face_order = order_faces(mesh, camera)
    F = mesh.F[face_order]  # (n_F, 3)

    FV = mesh.V[F].sum(axis=-2)  # (n_F, 3)
    FN = normalize(np.cross(mesh.V[F[:, 2]] - mesh.V[F[:, 0]], mesh.V[F[:, 1]] - mesh.V[F[:, 0]]))  # (n_F, 3)
    FC = np.array([[200 / 255, 200 / 255, 0]]).repeat(len(F), axis=0) if mesh.C is None else mesh.C

    C = shader.shade(FV, FN, FC, camera.position)

    d = draw.Drawing(camera.width, camera.height)
    for f, c in zip(F, C):
        Vf = V[f]  # (3, 3)
        d.append(
            draw.Lines(
                Vf[0][0],
                Vf[0][1],
                Vf[1][0],
                Vf[1][1],
                Vf[2][0],
                Vf[2][1],
                close=True,
                opacity=opacity,
                fill=rgb2hex(c),
                # stroke="black",
                # stroke_width="0.5"
            )
        )

    return d


@dataclass
class _LineSegment:
    fid: int
    p1: NpVec3f
    p2: NpVec3f  # assume p1.x <= p2.x
    _t: int  # use for tie-breaking

    def p_at(self, x: float) -> NpVec3f:
        return self.p1 + (self.p2 - self.p1) * (x - self.p1[0]) / (self.p2[0] - self.p1[0])


class _Event(NamedTuple):
    type: int  # 0: remove, 1: add, 2: intersect
    x: float
    y: float
    seg: _LineSegment
    seg2: Optional[_LineSegment]

    def _key(self) -> Tuple[float, float, int]:
        return (self.x, self.y, self.type)

    def __lt__(self, other: Self) -> bool:
        return self._key() < other._key()


def _sort_back_to_front2(V: NpMatX3f, F: NpMatX3i) -> NpVecXi:
    """
    V: (n_V, 3)
    F: (n_F, 3)
    returns: (n_F,)
    """
    # Precompute
    V = V.copy()
    V[:, 2] *= -1
    VF = V[F]  # (n_F, 3, 3)
    ZF = VF[:, :, 2]  # (n_F, 3)
    Z_centroid = ZF.mean(axis=-1)  # (n_F,)
    # Barycentric
    V23 = V[F[:, 1]] - V[F[:, 2]]  # (n_F, 3)
    V13 = V[F[:, 0]] - V[F[:, 2]]
    V3 = V[F[:, 2]]
    denom = V23[:, 1] * V13[:, 0] - V23[:, 0] * V13[:, 1]  # (n_F,)

    def get_z(fid: int, point: NpVec3f) -> float:
        d = point - V3[fid]
        w0 = (V23[fid, 1] * d[0] - V23[fid, 0] * d[1]) / denom[fid]
        w1 = (-V13[fid, 1] * d[0] + V13[fid, 0] * d[1]) / denom[fid]
        return (ZF[fid] * np.array([w0, w1, 1 - w0 - w1])).sum()

    n_F = len(F)
    ps = [Polygon(VF[i]) for i in range(n_F)]
    tree = STRtree(ps)
    rel: Set[Tuple[int, int]] = set()

    for i, p in enumerate(ps):
        candidates = tree.query(p)
        for j in candidates:
            j = int(j)
            if j == i:
                continue
            q = ps[j]
            pq = shapely.intersection(p, q)
            if pq.geom_type == "Polygon":
                pts = np.array(shapely.get_exterior_ring(pq).coords)
                if len(pts) > 0:
                    for pt in pts[:-1]:
                        zp = get_z(i, pt)
                        zq = get_z(j, pt)
                        if zp < zq:
                            rel.add((i, j))
                        elif zp > zq:
                            rel.add((j, i))

    # Topological sort; break ties using Z-centroid
    adj: List[List[int]] = [[] for _ in range(n_F)]
    indeg: List[int] = [0 for _ in range(n_F)]

    for i, j in rel:
        adj[i].append(j)
        indeg[j] += 1

    q: PriorityQueue[Tuple[float, int]] = PriorityQueue()
    iq: list[Tuple[float, int]] = [(Z_centroid[i], i) for i in range(n_F)]
    iq.sort(reverse=True)
    for i in range(n_F):
        if indeg[i] == 0:
            q.put((Z_centroid[i], i))
    res: List[int] = []
    n_errors = 0
    while len(res) < n_F:
        if q.empty():
            while len(iq) > 0:
                top = iq[-1][1]
                if indeg[top] >= 0:
                    break
                iq.pop()
            x = iq.pop()
            n_errors += indeg[x[1]]
            q.put(x)
        while not q.empty():
            _, fid = q.get()
            res.append(fid)
            indeg[fid] = -1
            for j in adj[fid]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.put((Z_centroid[j], j))
    if n_errors > 0:
        print(f"Warning: {n_errors} conflicts found", file=sys.stderr)
    return np.array(res)


def _sort_back_to_front(V: NpMatX3f, F: NpMatX3i) -> NpVecXi:
    """
    V: (n_V, 3)
    F: (n_F, 3)
    returns: (n_F,)
    """

    p1s = V[F].reshape(-1, 3)  # (n_F * 3, 3)
    p2s = V[F[:, [1, 2, 0]]].reshape(-1, 3)  # (n_F * 3, 3)

    evts: PriorityQueue[_Event] = PriorityQueue()

    for i, (p1, p2) in enumerate(zip(p1s, p2s)):
        fid = i // 3
        if p1.x > p2.x:
            p1, p2 = p2, p1
        seg = _LineSegment(fid, p1, p2, -1)
        evts.put(_Event(1, p1[0], p1[1], seg, None))
        evts.put(_Event(0, p2[0], p2[1], seg, None))

    cur_x = float("-inf")

    def seg_key(seg: _LineSegment):
        p = seg.p_at(cur_x)
        return (p[1], seg._t)

    container = SortedList(key=seg_key)
    _t = -1

    def new_t() -> int:
        nonlocal _t
        _t += 1
        return _t

    while not evts.empty():
        evt = evts.get()
        cur_x = evt.x
        if evt.type == 0:
            container.remove(evt.seg)
        elif evt.type == 1:
            evt.seg._t = new_t()
            container.add(evt.seg)
            idx = container.index(evt.seg)

            ...

    faces = V[F]  # (n_F, 3, 3)
    z_centroids = -np.sum(faces[:, :, 2], axis=1)
    for face_index in range(len(z_centroids)):
        z_centroids[face_index] /= len(faces[face_index])
    return np.argsort(z_centroids)
