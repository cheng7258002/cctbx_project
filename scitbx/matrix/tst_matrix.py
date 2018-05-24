"""
Note: this module can be used in isolation (without the rest of scitbx).
All external dependencies (other than plain Python) are optional.

The primary purpose of this module is to provide compact
implementations of essential matrix/vector algorithms with minimal
external dependencies. Optimizations for execution performance are
used only if compactness is not affected. The scitbx/math module
provides faster C++ alternatives to some algorithms included here.
"""

from __future__ import absolute_import, division, print_function

import scitbx.matrix # to access internal variables
from scitbx.matrix import *

def exercise_1():
  try:
    from libtbx import test_utils
  except ImportError:
    print("INFO: libtbx not available: some tests disabled.")
    test_utils = None
    def approx_equal(a, b): return True
    Exception_expected = RuntimeError
  else:
    approx_equal = test_utils.approx_equal
    Exception_expected = test_utils.Exception_expected
  #
  a = zeros(n=0)
  assert a.n == (0,1)
  assert a.elems == ()
  a = zeros(n=1)
  assert a.n == (1,1)
  assert a.elems == (0,)
  a = zeros(n=2)
  assert a.n == (2,1)
  assert a.elems == (0,0)
  a = zeros(n=(0,0))
  assert a.n == (0,0)
  assert a.elems == ()
  a = zeros(n=(1,0))
  assert a.n == (1,0)
  assert a.elems == ()
  a = zeros(n=(2,3))
  assert a.elems == (0,0,0,0,0,0)
  #
  a = mutable_col((0,0,0))
  a[1] = 1
  assert tuple(a) == (0,1,0)
  #
  for n in [(0,0), (1,0), (0,1)]:
    a = rec((),n)
    assert a.mathematica_form() == "{}"
    assert a.matlab_form() == "[]"
  a = rec(range(1,7), (3,2))
  assert len(a) == 6
  assert a[1] == 2
  assert (a*3).mathematica_form() == "{{3, 6}, {9, 12}, {15, 18}}"
  assert (-2*a).mathematica_form() == "{{-2, -4}, {-6, -8}, {-10, -12}}"
  for seq in [(2,-3), [2,-3]]:
    assert (a*seq).elems == (a*col((2,-3))).elems
  b = rec(range(1,7), (2,3))
  assert a.dot(b) == 91
  assert col((3,4)).dot() == 25
  c = a * b
  d = rt((c, (1,2,3)))
  assert (-a).mathematica_form() == "{{-1, -2}, {-3, -4}, {-5, -6}}"
  assert d.r.mathematica_form() == "{{9, 12, 15}, {19, 26, 33}, {29, 40, 51}}"
  assert d.t.mathematica_form() == "{{1}, {2}, {3}}"
  e = d + col((3,5,6))
  assert e.r.mathematica_form() == "{{9, 12, 15}, {19, 26, 33}, {29, 40, 51}}"
  assert e.t.mathematica_form() == "{{4}, {7}, {9}}"
  f = e - col((1,2,3))
  assert f.r.mathematica_form() == "{{9, 12, 15}, {19, 26, 33}, {29, 40, 51}}"
  assert f.t.mathematica_form() == "{{3}, {5}, {6}}"
  e = e + f
  assert e.r.mathematica_form() \
      == "{{18, 24, 30}, {38, 52, 66}, {58, 80, 102}}"
  assert e.t.mathematica_form() == "{{7}, {12}, {15}}"
  f = f - e
  assert f.r.mathematica_form() \
      == "{{-9, -12, -15}, {-19, -26, -33}, {-29, -40, -51}}"
  assert f.t.mathematica_form() == "{{-4}, {-7}, {-9}}"
  e = f.as_float()
  assert e.r.mathematica_form() \
      == "{{-9.0, -12.0, -15.0}, {-19.0, -26.0, -33.0}, {-29.0, -40.0, -51.0}}"
  assert e.t.mathematica_form() == "{{-4.0}, {-7.0}, {-9.0}}"
  a = f.as_augmented_matrix()
  assert a.mathematica_form() == "{{-9, -12, -15, -4}, {-19, -26, -33, -7}," \
                               + " {-29, -40, -51, -9}, {0, 0, 0, 1}}"
  assert a.extract_block(stop=(1,1)).mathematica_form() \
      == "{{-9}}"
  assert a.extract_block(stop=(2,2)).mathematica_form() \
      == "{{-9, -12}, {-19, -26}}"
  assert a.extract_block(stop=(3,3)).mathematica_form() \
      == "{{-9, -12, -15}, {-19, -26, -33}, {-29, -40, -51}}"
  assert a.extract_block(stop=(4,4)).mathematica_form() \
      == a.mathematica_form()
  assert a.extract_block(stop=(4,4),step=(2,2)).mathematica_form() \
      == "{{-9, -15}, {-29, -51}}"
  assert a.extract_block(start=(1,1),stop=(4,4),step=(2,2)).mathematica_form()\
      == "{{-26, -7}, {0, 1}}"
  assert a.extract_block(start=(1,0),stop=(4,3),step=(2,1)).mathematica_form()\
      == "{{-19, -26, -33}, {0, 0, 0}}"
  #
  for ar in xrange(3):
    for bc in xrange(3):
      a = rec([], (ar,0))
      b = rec([], (0,bc))
      c = a * b
      assert c.elems == tuple([0] * (ar*bc))
      assert c.n == (ar,bc)
  #
  ar = range(1,10)
  at = range(1,4)
  br = range(11,20)
  bt = range(4,7)
  g = rt((ar,at)) * rt((br,bt))
  assert g.r.mathematica_form() == \
    "{{90, 96, 102}, {216, 231, 246}, {342, 366, 390}}"
  assert g.t.mathematica_form() == "{{33}, {79}, {125}}"
  grt = g.r.transpose()
  assert grt.mathematica_form() == \
    "{{90, 216, 342}, {96, 231, 366}, {102, 246, 390}}"
  grtt = grt.transpose()
  assert grtt.mathematica_form() == \
    "{{90, 96, 102}, {216, 231, 246}, {342, 366, 390}}"
  gtt = g.t.transpose()
  assert gtt.mathematica_form() == "{{33, 79, 125}}"
  gttt = gtt.transpose()
  assert gttt.mathematica_form() == "{{33}, {79}, {125}}"
  assert sqr([4]).determinant() == 4
  assert sqr([3,2,-7,15]).determinant() == 59
  m = rec(elems=(), n=(0,0))
  mi = m.inverse()
  assert mi.n == (0,0)
  m = sqr([4])
  mi = m.inverse()
  assert mi.mathematica_form() == "{{0.25}}"
  m = sqr((1,5,-3,9))
  mi = m.inverse()
  assert mi.n == (2,2)
  assert approx_equal(mi, (3/8, -5/24, 1/8, 1/24))
  m = sqr((7, 7, -4, 3, 1, -1, 15, 16, -9))
  assert m.determinant() == 1
  mi = m.inverse()
  assert mi.mathematica_form() \
      == "{{7.0, -1.0, -3.0}, {12.0, -3.0, -5.0}, {33.0, -7.0, -14.0}}"
  assert (m*mi).mathematica_form() \
      == "{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}"
  assert (mi*m).mathematica_form() \
      == "{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}"
  s = rt((m, (1,-2,3)))
  si = s.inverse()
  assert si.r.mathematica_form() == mi.mathematica_form()
  assert (s*si).r.mathematica_form() \
      == "{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}"
  assert (si*s).r.mathematica_form() \
      == "{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}"
  assert si.t.mathematica_form().replace("-0.0", "0.0") \
      == "{{0.0}, {-3.0}, {-5.0}}"
  assert (s*si).t.mathematica_form() == "{{0.0}, {0.0}, {0.0}}"
  assert (si*s).t.mathematica_form() == "{{0.0}, {0.0}, {0.0}}"
  #
  m = rt((sqr([
    0.1226004505424059, 0.69470636260753704, -0.70876808568064376,
    0.20921402107901119, -0.71619825067837384, -0.66579993925291725,
    -0.97015391712385002, -0.066656848694206489, -0.23314853980114805]),
    col((0.34, -0.78, 0.43))))
  assert approx_equal(m.r*m.r.transpose(), (1,0,0,0,1,0,0,0,1))
  mi = m.inverse()
  mio = m.inverse_assuming_orthogonal_r()
  assert approx_equal(mio.r, mi.r)
  assert approx_equal(mio.t, mi.t)
  #
  r = rec(elems=(8,-4,3,-2,7,9,-3,2,1), n=(3,3))
  t = rec(elems=(7,-6,3), n=(3,1))
  gr = g * r
  assert gr.r == g.r * r
  assert gr.t == g.t
  gt = g * t
  assert gt == g.r * t + g.t
  gr = g * r.elems
  assert gr.r == g.r * r
  assert gr.t == g.t
  gt = g * t.elems
  assert gt == g.r * t + g.t
  try: g * col([1])
  except ValueError as e:
    assert str(e).startswith("cannot multiply ")
    assert str(e).endswith(": incompatible number of rows or columns")
  else: raise Exception_expected
  try: g * [1]
  except ValueError as e:
    assert str(e).startswith("cannot multiply ")
    assert str(e).endswith(": incompatible number of elements")
  else: raise Exception_expected
  flex = flex_proxy()
  if flex:
    gv = g * flex.vec3_double([(-1,2,3),(2,-3,4)])
    assert isinstance(gv, flex.vec3_double)
    assert approx_equal(gv, [(441, 1063, 1685), (333, 802, 1271)])
  else:
    print("INFO: scitbx.array_family.flex not available.")
  #
  try: from boost import rational
  except ImportError: pass
  else:
    assert approx_equal(col((rational.int(3,4),2,1.5)).as_float(),(0.75,2,1.5))
  #
  assert approx_equal(col((-2,3,-6)).normalize().elems, (-2/7.,3/7.,-6/7.))
  assert col((-1,2,-3)).each_abs().elems == (1,2,3)
  assert approx_equal(col((-1.7,2.8,3.4)).each_mod_short(), (0.3,-0.2,0.4))
  assert col((5,3,4)).min() == 3
  assert col((4,5,3)).max() == 5
  assert col((5,3,4)).min_index() == 1
  assert col((4,5,3)).max_index() == 1
  assert col((4,5,3)).sum() == 12
  assert col((2,3,4)).product() == 2*3*4
  assert sqr((1,2,3,4,5,6,7,8,9)).trace() == 15
  assert diag((1,2,3)).mathematica_form() == \
    "{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"
  assert approx_equal(col((1,0,0)).cos_angle(col((1,1,0)))**2, 0.5)
  assert approx_equal(col((1,0,0)).angle(col((1,1,0))), 0.785398163397)
  assert approx_equal(col((1,0,0)).angle(col((1,1,0)), deg=True), 45)
  assert approx_equal(col((-1,0,0)).angle(col((1,1,0)), deg=True), 45+90)
  assert approx_equal(col((1,0,0)).accute_angle(col((1,1,0)), deg=True), 45)
  assert approx_equal(col((-1,0,0)).accute_angle(col((1,1,0)), deg=True), 45)
  m = sqr([4, 4, -1, 0, -3, -3, -3, -2, -3, 2, -1, 1, -4, 1, 3, 2])
  md = m.determinant()
  assert approx_equal(md, -75)
  #
  r = sqr([-0.9533, 0.2413, -0.1815,
           0.2702, 0.414, -0.8692,
           -0.1346, -0.8777, -0.4599])
  assert r.mathematica_form(
    label="rotation",
    format="%.6g",
    one_row_per_line=True,
    prefix="  ") == """\
  rotation={{-0.9533, 0.2413, -0.1815},
            {0.2702, 0.414, -0.8692},
            {-0.1346, -0.8777, -0.4599}}"""
  r = sqr([])
  assert r.mathematica_form() == "{}"
  assert r.mathematica_form(one_row_per_line=True) == "{}"
  r = sqr([1])
  assert r.mathematica_form() == "{{1}}"
  assert r.mathematica_form(one_row_per_line=True, prefix="&$") == """\
&${{1}}"""
  #
  a = rec(range(1,6+1), (3,2))
  b = rec(range(1,15+1), (3,5))
  assert approx_equal(a.transpose_multiply(b), a.transpose() * b)
  assert approx_equal(a.transpose_multiply(a), a.transpose() * a)
  assert approx_equal(a.transpose_multiply(), a.transpose() * a)
  assert approx_equal(b.transpose_multiply(b), b.transpose() * b)
  assert approx_equal(b.transpose_multiply(), b.transpose() * b)
  #
  a = col([1,2,3])
  b = row([10,20])
  assert a.outer_product(b).as_list_of_lists() == [[10,20],[20,40],[30,60]]
  assert a.outer_product().as_list_of_lists() == [[1,2,3],[2,4,6],[3,6,9]]
  #
  a = sym(sym_mat3=range(6))
  assert a.as_list_of_lists() == [[0, 3, 4], [3, 1, 5], [4, 5, 2]]
  assert approx_equal(a.as_sym_mat3(), range(6))
  assert a.as_mat3() == (0,3,4,3,1,5,4,5,2)
  if flex:
    f = a.as_flex_double_matrix()
    assert f.all() == a.n
    assert approx_equal(f, a, eps=1.e-12)
  #
  for i in xrange(3):
    x = rec([], n=(0,i))
    assert repr(x) == "matrix.rec(elems=(), n=(0,%d))" % i
    assert str(x) == "{}"
    assert x.matlab_form() == "[]"
    assert x.matlab_form(one_row_per_line=True) == "[]"
    x = rec([], n=(i,0))
    assert repr(x) == "matrix.rec(elems=(), n=(%d,0))" % i
    assert str(x) == "{}"
    assert x.matlab_form() == "[]"
    assert x.matlab_form(one_row_per_line=True) == "[]"
  x = rec([2], n=(1,1))
  assert repr(x) == "matrix.rec(elems=(2,), n=(1,1))"
  assert str(x) == "{{2}}"
  assert x.matlab_form() == "[2]"
  assert x.matlab_form(one_row_per_line=True) == "[2]"
  x = col((1,2,3))
  assert repr(x) == "matrix.rec(elems=(1, 2, 3), n=(3,1))"
  assert str(x) == """\
{{1},
 {2},
 {3}}"""
  assert x.matlab_form() == "[1; 2; 3]"
  assert x.matlab_form(one_row_per_line=True) == """\
[1;
 2;
 3]"""
  x = row((3,2,1))
  assert repr(x) == "matrix.rec(elems=(3, 2, 1), n=(1,3))"
  assert str(x) == "{{3, 2, 1}}"
  assert x.matlab_form() == "[3, 2, 1]"
  assert x.matlab_form(one_row_per_line=True) == "[3, 2, 1]"
  x = rec((1,2,3,
           4,5,6,
           7,8,9,
           -1,-2,-3), (4,3))
  assert repr(x) == "matrix.rec(elems=(1, ..., -3), n=(4,3))"
  assert str(x) == """\
{{1, 2, 3},
 {4, 5, 6},
 {7, 8, 9},
 {-1, -2, -3}}"""
  assert x.matlab_form() == "[1, 2, 3; 4, 5, 6; 7, 8, 9; -1, -2, -3]"
  assert x.matlab_form(label="m", one_row_per_line=True, prefix="@") == """\
@m=[1, 2, 3;
@   4, 5, 6;
@   7, 8, 9;
@   -1, -2, -3]"""
  #
  t = (1,2,3,4,5,6)
  g = (3,2)
  a = rec(t, g)
  b = rec(t, g)
  assert a == b
  if flex:
    c = flex.double(t)
    c.reshape(flex.grid(g))
    assert a == c
  #
  a = identity(4)
  for ir in xrange(4):
    for ic in xrange(4):
      assert (ir == ic and a(ir,ic) == 1) or (ir != ic and a(ir,ic) == 0)
  a = inversion(4)
  for ir in xrange(4):
    for ic in xrange(4):
      assert (ir == ic and a(ir,ic) == -1) or (ir != ic and a(ir,ic) == 0)
  #
  x = col((3/2+0.01, 5/4-0.02, 11/8+0.001))
  assert (x*8).as_int()/8 == col((3/2, 5/4, 11/8))
  #
  for x in [(0, 0, 0), (1, -2, 5), (-2, 5, 1), (5, 1, -2) ]:
    x = col(x)
    assert approx_equal(x.dot(x.ortho()), 0)
  #
  x = col((1, -2, 3))
  n = col((-2, 4, 5))
  alpha = 2*math.pi/3
  y = x.rotate_around_origin(n, alpha)
  n = n.normalize()
  x_perp = x - n.dot(x)*n
  y_perp = y - n.dot(y)*n
  assert approx_equal(x_perp.angle(y_perp), alpha)
  x = col((0,1,0))
  y = x.rotate_around_origin(axis=col((1,0,0)), angle=75, deg=True)
  assert approx_equal(y, (0.0, 0.25881904510252074, 0.96592582628906831))
  assert approx_equal(x.angle(y, deg=True), 75)
  a = col((0.33985998937421624, 0.097042540321188753, -0.60916214763712317))
  x = col((0.61837962293383231, -0.46724958233858915, -0.48367879178081852))
  y = x.rotate_around_origin(axis=a, angle=37, deg=True)
  assert approx_equal(abs(x), 0.913597670681)
  assert approx_equal(abs(y), 0.913597670681)
  assert approx_equal(x.angle(y, deg=True), 25.6685689758)
  assert approx_equal(y, (0.2739222799, -0.5364841936, -0.6868857244))
  uq = a.axis_and_angle_as_unit_quaternion(angle=37, deg=True)
  assert approx_equal(uq, (0.94832366, 0.15312122, 0.04372175, -0.27445317))
  r = uq.unit_quaternion_as_r3_rotation_matrix()
  assert approx_equal(r*x, y)
  assert approx_equal(
    a.axis_and_angle_as_r3_rotation_matrix(angle=37, deg=True), r)
  #
  pivot = col((29.278,-48.061,72.641))
  raa = pivot.rt_for_rotation_around_axis_through(
    point=col((28.09,-48.047,71.684)),
    angle=190.811940444, deg=True)
  assert approx_equal(
    raa * col((28.097,-47.559,70.248)),
    (26.639170440424856,-48.299377845438173,72.046888429403481))
  #
  assert col((0,0,1)).vector_to_001_rotation().elems == (1,0,0,0,1,0,0,0,1)
  assert col((0,0,-1)).vector_to_001_rotation().elems == (1,0,0,0,-1,0,0,0,-1)
  assert approx_equal(
    col((5,3,-7)).normalize().vector_to_001_rotation(),
    (-0.3002572205351709, -0.78015433232110254, -0.54882129994845175,
     -0.78015433232110254, 0.53190740060733865, -0.32929277996907103,
     0.54882129994845175, 0.32929277996907103, -0.76834981992783236))
  #
  a = row((1,0,0))
  b = row((0,1,0))
  assert a.cross(b) == row((0,0,1))
  #
  a = col((1.43416642866471794, -2.47841960952275497, -0.7632916804502845))
  b = col((0.34428681113080323, -1.85983494542314587, 0.37702845822372399))
  assert approx_equal(a.cross(b), cross_product_matrix(a) * b)
  #
  f = linearly_dependent_pair_scaling_factor
  assert approx_equal(f(vector_1=[1,2,3], vector_2=[3,6,9]), 3)
  assert approx_equal(f(vector_1=[3,6,9], vector_2=[1,2,3]), 1/3)
  assert f(vector_1=col([0,1,1]), vector_2=[1,1,1]) is None
  assert f(vector_1=[1,1,1], vector_2=col([0,1,1])) is None
  assert f(vector_1=col([0,0,0]), vector_2=col([0,0,0])) is 0
  #
  a = col_list(seq=[(1,2), (2,3)])
  for e in a: assert isinstance(e, col)
  assert approx_equal(a, [(1,2), (2,3)])
  a = row_list(seq=[(1,2), (2,3)])
  for e in a: assert isinstance(e, row)
  assert approx_equal(a, [(1,2), (2,3)])
  #
  def f(a): return a.resolve_partitions().mathematica_form()
  a = rec(elems=[], n=[0,0])
  assert f(a) == "{}"
  a = rec(elems=[], n=[0,1])
  assert f(a) == "{}"
  a = rec(elems=[], n=[1,0])
  assert f(a) == "{}"
  for e in [col([]), row([])]:
    a = rec(elems=[e], n=[1,1])
    assert f(a) == "{}"
    a = rec(elems=[e, e], n=[1,2])
    assert f(a) == "{}"
    a = rec(elems=[e, e], n=[2,1])
    assert f(a) == "{}"
  for e in [col([1]), row([1])]:
    a = rec(elems=[e], n=[1,1])
    assert f(a) == "{{1}}"
  a = rec(elems=[col([1,2]), col([3,4])], n=[1,2])
  assert f(a) == "{{1, 3}, {2, 4}}"
  a = rec(elems=[col([1,2]), col([3,4])], n=[2,1])
  assert f(a) == "{{1}, {2}, {3}, {4}}"
  a = rec(elems=[sqr([1,2,3,4]), sqr([5,6,7,8])], n=[1,2])
  assert f(a) == "{{1, 2, 5, 6}, {3, 4, 7, 8}}"
  a = rec(elems=[sqr([1,2,3,4]), sqr([5,6,7,8])], n=[2,1])
  assert f(a) == "{{1, 2}, {3, 4}, {5, 6}, {7, 8}}"
  a = rec(elems=[rec([1,2,3,4,5,6], n=(2,3)), rec([7,8], n=(2,1))], n=[1,2])
  assert f(a) == "{{1, 2, 3, 7}, {4, 5, 6, 8}}"
  a = rec(elems=[rec([1,2,3,4,5,6], n=(2,3)), rec([7,8,9], n=(1,3))], n=[2,1])
  assert f(a) == "{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}"
  a = rec(
    elems=[
      sqr([11,12,13,14,15,16,17,18,19]),
      sqr([21,22,23,24,25,26,27,28,29]),
      sqr([31,32,33,34,35,36,37,38,39]),
      sqr([41,42,43,44,45,46,47,48,49])],
    n=[2,2])
  assert a.resolve_partitions().mathematica_form(one_row_per_line=True) == """\
{{11, 12, 13, 21, 22, 23},
 {14, 15, 16, 24, 25, 26},
 {17, 18, 19, 27, 28, 29},
 {31, 32, 33, 41, 42, 43},
 {34, 35, 36, 44, 45, 46},
 {37, 38, 39, 47, 48, 49}}"""
  a = rec(
    elems=[
      rec([1,2,3,4,5,6], n=(2,3)),
      rec([7,8,9,10,11,12,13,14], n=(2,4)),
      rec([15,16,17,18,19,20,21,22,23], n=(3,3)),
      rec([24,25,26,27,28,29,30,31,32,33,34,35], n=(3,4)),
      rec([36,37,38], n=(1,3)),
      rec([39,40,41,42], n=(1,4))],
    n=[3,2])
  assert a.resolve_partitions().mathematica_form(one_row_per_line=True) == """\
{{1, 2, 3, 7, 8, 9, 10},
 {4, 5, 6, 11, 12, 13, 14},
 {15, 16, 17, 24, 25, 26, 27},
 {18, 19, 20, 28, 29, 30, 31},
 {21, 22, 23, 32, 33, 34, 35},
 {36, 37, 38, 39, 40, 41, 42}}"""
  #
  def check(m, expected):
    assert approx_equal(determinant_via_lu(m=m), expected)
    assert approx_equal(m.determinant(), expected)
    if (expected != 0):
      mi = inverse_via_lu(m=m)
      assert mi.n == m.n
      assert approx_equal(mi*m, identity(n=m.n[0]))
      assert approx_equal(m*mi, identity(n=m.n[0]))
      mii = inverse_via_lu(m=mi)
      assert approx_equal(mii, m)
  check(sqr([]), 1)
  check(sqr([0]), 0)
  check(sqr([4]), 4)
  check(sqr([0]*4), 0)
  check(sqr([1,2,-3,4]), 10)
  check(sqr([0]*9), 0)
  check(sqr([
    0.1226004505424059, 0.69470636260753704, -0.70876808568064376,
    0.20921402107901119, -0.71619825067837384, -0.66579993925291725,
    -0.97015391712385002, -0.066656848694206489, -0.23314853980114805]), 1)
  check(sqr([0]*16), 0)
  check(sqr([
    -2/15,-17/75,4/75,-19/75,
    7/15,22/75,-14/75,29/75,
    1/3,4/15,-8/15,8/15,
    -1,-1,1,-1]), -1/75)
  #
  r = identity(n=3)
  assert r.is_r3_rotation_matrix()
  uqr = r.r3_rotation_matrix_as_unit_quaternion()
  assert approx_equal(uqr, (1,0,0,0))
  # axis = (1/2**0.5, 1/2**0.5, 0)
  # angle = 2 * math.asin((2/3.)**0.5)
  uq = col((1/3**0.5,1/3**0.5,1/3**0.5,0))
  r = sqr((
     1/3.,2/3., 2/3.,
     2/3.,1/3.,-2/3.,
    -2/3.,2/3.,-1/3.))
  assert approx_equal(uq.unit_quaternion_as_r3_rotation_matrix(), r)
  uqr = r.r3_rotation_matrix_as_unit_quaternion()
  assert approx_equal(uqr, uq)
  #
  for i_trial in xrange(10):
    uq1 = col.random(n=4, a=-1, b=1).normalize()
    uq2 = col.random(n=4, a=-1, b=1).normalize()
    r1 = uq1.unit_quaternion_as_r3_rotation_matrix()
    r2 = uq2.unit_quaternion_as_r3_rotation_matrix()
    uqp12 = uq1.unit_quaternion_product(uq2)
    rp12 = uqp12.unit_quaternion_as_r3_rotation_matrix()
    assert approx_equal(rp12, r1*r2)
    uqp21 = uq2.unit_quaternion_product(uq1)
    rp21 = uqp21.unit_quaternion_as_r3_rotation_matrix()
    assert approx_equal(rp21, r2*r1)
    for uq,r in [(uq1,r1), (uq2,r2), (uqp12,rp12), (uqp21,rp21)]:
      assert r.is_r3_rotation_matrix()
      uqr = r.r3_rotation_matrix_as_unit_quaternion()
      assert approx_equal(uqr.unit_quaternion_as_r3_rotation_matrix(), r)
  #
  r = sqr((
    0.12, 0.69, -0.70,
    0.20, -0.71, -0.66,
    -0.97, -0.06, -0.23))
  assert approx_equal(r.is_r3_rotation_matrix_rms(), 0.0291602469125)
  assert not r.is_r3_rotation_matrix()
  #
  v = col((1.1, -2.2, 2.3))
  assert approx_equal(v % 2, col((1.1, 1.8, 0.3)))
  #
  rational1 = sqr((2,1,1,0,1,0,0,0,1))
  assert str(rational1.inverse().elems)==\
    "(0.5, -0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)"
  if flex:
    assert str(rational1.as_boost_rational().inverse().elems)==\
      "(1/2, -1/2, -1/2, 0, 1, 0, 0, 0, 1)"
  #
  assert scitbx.matrix._dihedral_angle(sites=col_list([(0,0,0)]*4), deg=False) is None
  assert dihedral_angle(sites=col_list([(0,0,0)]*4)) is None
  sites = col_list([
    (-3.193, 1.904, 4.589),
    (-1.955, 1.332, 3.895),
    (-1.005, 2.228, 3.598),
    ( 0.384, 1.888, 3.199)])
  expected = 166.212120415
  assert approx_equal(scitbx.matrix._dihedral_angle(sites=sites, deg=True), expected)
  assert approx_equal(dihedral_angle(sites=sites, deg=True), expected)
  # more dihedral tests in scitbx/math/boost_python/tst_math.py
  #
  if (test_utils is not None):
    from libtbx import group_args
    for deg in [False, True]:
      args = group_args(
        axis_point_1=sites[0],
        axis_point_2=sites[1],
        point=sites[2],
        angle=13,
        deg=True).__dict__
      assert approx_equal(
        rotate_point_around_axis(**args),
        scitbx.matrix.__rotate_point_around_axis(**args))
    args = group_args(
      axis_point_1=sites[0],
      axis_point_2=sites[1],
      point=sites[2]).__dict__
    assert approx_equal(
      project_point_on_axis(**args),
      list(scitbx.matrix.__project_point_on_axis(**args))[0])
  # exercise plane_equation
  point_1=col((1,2,3))
  point_2=col((10,20,30))
  point_3=col((7,53,18))
  a,b,c,d = plane_equation(point_1=point_1, point_2=point_2, point_3=point_3)
  point_in_plane = (point_3 - (point_2-point_1)/2)/2
  assert approx_equal(
    a*point_in_plane[0]+b*point_in_plane[1]+c*point_in_plane[2]+d,0)
  xyz = (0,1,1)
  assert approx_equal(2.828427,
          distance_from_plane(xyz=(1,1,5), points=[(0,0,0), (1,1,1), (-1,1,1)]))
  d = distance_from_plane(xyz=(0,0,1), points=[(0,0,0), (0,0,0), (0,1,0)])
  assert d is None
  d = distance_from_plane(xyz=(0,0,1), points=[(0,0,0), (1,0,0), (0,1,0)])
  assert approx_equal(d, 1)
  #
  numpy = numpy_proxy()
  if numpy:
    m = rec(elems=range(6), n=(2,3))
    n = m.as_numpy_array()
    assert n.tolist() == [[0, 1, 2], [3, 4, 5]]
  else:
    print("INFO: numpy not available.")
  #
  print("OK")

def exercise_2():
  points = \
    [[20.559, 2.613, 29.030],
     [21.030, 3.817, 29.627],
     [21.079, 3.972, 30.986],
     [21.604, 5.302, 31.090],
     [21.813, 5.803, 29.779],
     [21.448, 4.862, 28.912],
     [20.781, 3.239, 32.060],
     [20.978, 3.765, 33.283],
     [21.472, 5.018, 33.416],
     [21.774, 5.761, 32.331],
     [22.288, 7.066, 32.547],
     [21.481, 4.924, 27.938],
     [20.765, 3.241, 34.078],
     [22.403, 7.376, 33.397],
     [22.503, 7.595, 31.835]]
  assert all_in_plane(points=points, tolerance=0.005)
  points = \
    [[20.559, 2.613, 29.030],
     [21.030, 3.817, 29.627],
     [21.079, 3.972, 30.986],
     [21.604, 5.302, 31.090],
     [21.813, 5.803, 29.779],
     [21.448, 4.862, 28.912],
     [20.781, 3.239, 32.060],
     [20.978, 3.765, 33.283],
     [21.472, 0.018, 33.416],
     [21.774, 5.761, 32.331],
     [22.288, 7.066, 32.547],
     [21.481, 4.924, 27.938],
     [20.765, 3.241, 34.078],
     [22.403, 7.376, 33.397],
     [22.503, 7.595, 31.835]]
  assert not all_in_plane(points=points, tolerance=0.005)

if __name__ == "__main__":
  exercise_1()
  exercise_2()
