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

import math
import random

_flex_imported = False
def flex_proxy():
  global _flex_imported
  if (_flex_imported is False):
    try: from scitbx.array_family import flex as _flex_imported
    except ImportError: _flex_imported = None
  return _flex_imported

_numpy_imported = False
def numpy_proxy():
  global _numpy_imported
  if (_numpy_imported is False):
    try: import numpy.linalg
    except ImportError: _numpy_imported = None
    else: _numpy_imported = numpy
  return _numpy_imported

class rec(object):
  """
  Base rectangle object, used to store 2-dimensional data.

  Examples
  --------
  >>> from scitbx.matix import rec
  >>> m = rec((1, 2, 3, 4), (2, 2))
  >>> print m.trace()
  5
  >>> other = rec((1, 1, 1, 1), (2, 2))
  >>> print m.dot(other)
  10
  """

  container_type = tuple

  def __init__(self, elems, n):
    assert len(n) == 2
    if (not isinstance(elems, self.container_type)):
      elems = self.container_type(elems)
    assert len(elems) == n[0] * n[1]
    self.elems = elems
    self.n = tuple(n)

  def n_rows(self):
    return self.n[0]

  def n_columns(self):
    return self.n[1]

  def __neg__(self):
    return rec([-e for e in self.elems], self.n)

  def __add__(self, other):
    assert self.n == other.n
    a = self.elems
    b = other.elems
    return rec([a[i] + b[i] for i in xrange(len(a))], self.n)

  def __sub__(self, other):
    assert self.n == other.n
    a = self.elems
    b = other.elems
    return rec([a[i] - b[i] for i in xrange(len(a))], self.n)

  def __mul__(self, other):
    if (not hasattr(other, "elems")):
      if (not isinstance(other, (list, tuple))):
        return rec([x * other for x in self.elems], self.n)
      other = col(other)
    a = self.elems
    ar = self.n_rows()
    ac = self.n_columns()
    b = other.elems
    if (other.n_rows() != ac):
      raise RuntimeError(
        "Incompatible matrices:\n"
        "  self.n:  %s\n"
        "  other.n: %s" % (str(self.n), str(other.n)))
    bc = other.n_columns()
    if (ac == 0):
      # Roy Featherstone, Springer, New York, 2007, p. 53 footnote
      return rec((0,)*(ar*bc), (ar,bc))
    result = []
    for i in xrange(ar):
      for k in xrange(bc):
        s = 0
        for j in xrange(ac):
          s += a[i * ac + j] * b[j * bc + k]
        result.append(s)
    if (ar == bc):
      return sqr(result)
    return rec(result, (ar, bc))

  def __rmul__(self, other):
    "scalar * matrix"
    if (isinstance(other, rec)): # work around odd Python 2.2 feature
      return other.__mul__(self)
    return self * other

  def transpose_multiply(self, other=None):
    a = self.elems
    ar = self.n_rows()
    ac = self.n_columns()
    if (other is None):
      result = [0] * (ac * ac)
      jac = 0
      for j in xrange(ar):
        ik = 0
        for i in xrange(ac):
          for k in xrange(ac):
            result[ik] += a[jac + i] * a[jac + k]
            ik += 1
        jac += ac
      return sqr(result)
    b = other.elems
    assert other.n_rows() == ar, "Incompatible matrices."
    bc = other.n_columns()
    result = [0] * (ac * bc)
    jac = 0
    jbc = 0
    for j in xrange(ar):
      ik = 0
      for i in xrange(ac):
        for k in xrange(bc):
          result[ik] += a[jac + i] * b[jbc + k]
          ik += 1
      jac += ac
      jbc += bc
    if (ac == bc):
      return sqr(result)
    return rec(result, (ac, bc))

  def __div__(self, other):
    return rec([e/other for e in self.elems], self.n)

  def __truediv__(self, other):
    return rec([e/other for e in self.elems], self.n)

  def __floordiv__(self, other):
    return rec([e//other for e in self.elems], self.n)

  def __mod__(self, other):
    return rec([ e % other for e in self.elems], self.n)

  def __call__(self, ir, ic):
    return self.elems[ir * self.n_columns() + ic]

  def __len__(self):
    return len(self.elems)

  def __getitem__(self, i):
    return self.elems[i]

  def as_float(self):
    return rec([float(e) for e in self.elems], self.n)

  def as_boost_rational(self):
    from boost import rational
    return rec([rational.int(e) for e in self.elems], self.n)

  def as_int(self, rounding=True):
    if rounding:
      return rec([int(round(e)) for e in self.elems], self.n)
    else:
      return rec([int(e) for e in self.elems], self.n)

  def as_numpy_array(self):
    numpy = numpy_proxy()
    assert numpy
    return numpy.array(self.elems).reshape(self.n)

  def each_abs(self):
    return rec([abs(e) for e in self.elems], self.n)

  def each_mod_short(self, period=1):
    half = period / 2.0
    def mod_short(e):
      r = math.fmod(e, period)
      if   (r < -half): r += period
      elif (r >  half): r -= period
      return r
    return rec([mod_short(e) for e in self.elems], self.n)

  def min(self):
    result = None
    for e in self.elems:
      if (result is None or result > e):
        result = e
    return result

  def max(self):
    result = None
    for e in self.elems:
      if (result is None or result < e):
        result = e
    return result

  def min_index(self):
    result = None
    for i in xrange(len(self.elems)):
      if (result is None or self.elems[result] > self.elems[i]):
        result = i
    return result

  def max_index(self):
    result = None
    for i in xrange(len(self.elems)):
      if (result is None or self.elems[result] < self.elems[i]):
        result = i
    return result

  def sum(self):
    result = 0
    for e in self.elems:
      result += e
    return result

  def product(self):
    result = 1
    for e in self.elems:
      result *= e
    return result

  def trace(self):
    assert self.n_rows() == self.n_columns()
    n = self.n_rows()
    result = 0
    for i in xrange(n):
      result += self.elems[i*n+i]
    return result

  def norm_sq(self):
    result = 0
    for e in self.elems:
      result += e*e
    return result

  def round(self, digits):
    return rec([ round(x, digits) for x in self.elems ], self.n)

  def __abs__(self):
    assert self.n_rows() == 1 or self.n_columns() == 1
    return math.sqrt(self.norm_sq())

  length_sq = norm_sq # for compatibility with scitbx/vec3.h
  length = __abs__

  def normalize(self):
    return self / abs(self)

  def dot(self, other=None):
    result = 0
    a = self.elems
    if (other is None):
      for i in xrange(len(a)):
        v = a[i]
        result += v * v
    else:
      assert len(self.elems) == len(other.elems)
      b = other.elems
      for i in xrange(len(a)):
        result += a[i] * b[i]
    return result

  def cross(self, other):
    assert self.n in ((3,1), (1,3))
    assert self.n == other.n
    a = self.elems
    b = other.elems
    return rec((
      a[1] * b[2] - b[1] * a[2],
      a[2] * b[0] - b[2] * a[0],
      a[0] * b[1] - b[0] * a[1]), self.n)

  def is_r3_rotation_matrix_rms(self):
    if (self.n != (3,3)): raise RuntimeError("Not a 3x3 matrix.")
    rtr = self.transpose_multiply()
    return (rtr - identity(n=3)).norm_sq()**0.5

  def is_r3_rotation_matrix(self, rms_tolerance=1e-8):
    return self.is_r3_rotation_matrix_rms() < rms_tolerance and \
      abs(1-self.determinant())<rms_tolerance

  def is_r3_identity_matrix(self):
    return (self.elems == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

  def is_col_zero(self):
    """ Is the translation vector zero """
    return (self.elems == (0.0, 0.0, 0.0))

  def is_zero(self):
    """ Are all elements zero  """
    return len([1 for x in self.elems if x != 0]) == 0

  def is_approx_zero(self, eps):
    """ Are all elements zero  """
    return len([1 for x in self.elems if abs(x)>eps]) == 0

  def unit_quaternion_as_r3_rotation_matrix(self):
    assert self.n in [(1,4), (4,1)]
    q0,q1,q2,q3 = self.elems
    return sqr((
      2*(q0*q0+q1*q1)-1, 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2),
      2*(q1*q2+q0*q3),   2*(q0*q0+q2*q2)-1, 2*(q2*q3-q0*q1),
      2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   2*(q0*q0+q3*q3)-1))

  def r3_rotation_matrix_as_x_y_z_angles(self, deg=False, alternate_solution=False):
    """
    Get the rotation angles around the axis x,y,z for rotation r
    Such that r = Rx*Ry*Rz
    Those angles are the Tait-Bryan angles form of Euler angles

    Note that typically there are two solutions, and this function will return
    only one. In the case that cos(beta) == 0 there are infinite number of
    solutions, the function returns the one where gamma = 0

    Args:
      deg : When False use radians, when True use degrees
      alternate_solution: return the alternate solution for the angles

    Returns:
      angles: containing rotation angles in the form
        (rotx, roty, rotz)
    """
    if (self.n != (3,3)): raise RuntimeError("Not a 3x3 matrix.")
    Rxx,Rxy,Rxz,Ryx,Ryy,Ryz,Rzx,Rzy,Rzz = self.elems
    if Rxz not in [1,-1]:
      beta = math.asin(Rxz)
      if alternate_solution:
        beta = math.pi - beta

      # using atan2 to take into account the possible different angles and signs
      alpha = math.atan2(-Ryz/math.cos(beta),Rzz/math.cos(beta))
      gamma = math.atan2(-Rxy/math.cos(beta),Rxx/math.cos(beta))
    elif Rxz == 1:
      beta = math.pi/2
      alpha = math.atan2(Ryx,Ryy)
      gamma = 0
    elif Rxz == -1:
      beta = -math.pi/2
      alpha = math.atan2(-Ryx,Ryy)
      gamma = 0
    else:
      raise ArithmeticError("Can't calculate rotation angles")

    if deg:
      # Convert to degrees
      return 180*alpha/math.pi, \
             180*beta /math.pi, \
             180*gamma/math.pi
    else:
      return alpha, beta, gamma

  def r3_rotation_matrix_as_unit_quaternion(self):
    # Based on work by:
    #   Shepperd (1978), J. Guidance and Control, 1, 223-224.
    #   Sam Buss, http://math.ucsd.edu/~sbuss/MathCG
    #   Robert Hanson, jmol/Jmol/src/org/jmol/util/Quaternion.java
    if (self.n != (3,3)): raise RuntimeError("Not a 3x3 matrix.")
    m00,m01,m02,m10,m11,m12,m20,m21,m22 = self.elems
    trace = m00 + m11 + m22
    if (trace >= 0.5):
      w = (1 + trace)**0.5
      d = w + w
      w *= 0.5
      x = (m21 - m12) / d
      y = (m02 - m20) / d
      z = (m10 - m01) / d
    else:
      if (m00 > m11):
        if (m00 > m22): mx = 0
        else:           mx = 2
      elif (m11 > m22): mx = 1
      else:             mx = 2
      invalid_cutoff = 0.8 # not critical; true value is closer to 0.83
      invalid_message = "Not a r3_rotation matrix."
      if (mx == 0):
        x_sq = 1 + m00 - m11 - m22
        if (x_sq < invalid_cutoff): raise RuntimeError(invalid_message)
        x = x_sq**0.5
        d = x + x
        x *= 0.5
        w = (m21 - m12) / d
        y = (m10 + m01) / d
        z = (m20 + m02) / d
      elif (mx == 1):
        y_sq = 1 + m11 - m00 - m22
        if (y_sq < invalid_cutoff): raise RuntimeError(invalid_message)
        y = y_sq**0.5
        d = y + y
        y *= 0.5
        w = (m02 - m20) / d
        x = (m10 + m01) / d
        z = (m21 + m12) / d
      else:
        z_sq = 1 + m22 - m00 - m11
        if (z_sq < invalid_cutoff): raise RuntimeError(invalid_message)
        z = z_sq**0.5
        d = z + z
        z *= 0.5
        w = (m10 - m01) / d
        x = (m20 + m02) / d
        y = (m21 + m12) / d
    return col((w, x, y, z))

  def unit_quaternion_product(self, other):
    assert self.n in [(1,4), (4,1)]
    assert other.n in [(1,4), (4,1)]
    q0,q1,q2,q3 = self.elems
    o0,o1,o2,o3 = other.elems
    return col((
      q0*o0 - q1*o1 - q2*o2 - q3*o3,
      q0*o1 + q1*o0 + q2*o3 - q3*o2,
      q0*o2 - q1*o3 + q2*o0 + q3*o1,
      q0*o3 + q1*o2 - q2*o1 + q3*o0))

  def quaternion_inverse(self):
    assert self.n in [(1,4), (4,1)]
    q0,q1,q2,q3 = self.elems
    return col(( q0,-q1,-q2,-q3 ))

  def unit_quaternion_as_axis_and_angle(self, deg=False):
    assert self.n in [(1,4), (4,1)]
    q0,q1,q2,q3 = self.elems
    if q0 > 1. : q0 = 1.
    if q0 < -1. : q0 = -1.
    angle_alpha = 2. * math.acos(q0)
    if deg:  angle_alpha *= 180./math.pi
    axis = col((q1,q2,q3))
    axis_length = axis.length()
    if axis_length > 0:
      axis = axis/axis_length #unit axis
    return angle_alpha,axis

  def axis_and_angle_as_unit_quaternion(self, angle, deg=False):
    assert self.n in ((3,1), (1,3))
    if (deg): angle *= math.pi/180
    h = angle * 0.5
    c, s = math.cos(h), math.sin(h)
    u,v,w = self.normalize().elems
    return col((c, u*s, v*s, w*s))

  def axis_and_angle_as_r3_rotation_matrix(self, angle, deg=False):
    uq = self.axis_and_angle_as_unit_quaternion(angle=angle, deg=deg)
    return uq.unit_quaternion_as_r3_rotation_matrix()

  def axis_and_angle_as_r3_derivative_wrt_angle(self, angle, deg=False):
    assert self.n in ((3,1), (1,3))
    prefactor = 1.
    if (deg): angle *= math.pi/180; prefactor = 180./math.pi
    unit_axis = self.normalize()
    #Use matrix form of Rodrigues' rotation formula (see Wikipedia, e.g.)
    I3 = identity(n=3)
    OP = unit_axis.outer_product(unit_axis)
    CP = cross_product_matrix(unit_axis.elems)
    c, s = math.cos(angle), math.sin(angle)
    return prefactor * ( -I3*s + OP*s + CP*c ) # bug fix 8/27/13 NKS
    # note: the rotation operator itself is prefactor * (I3*c + OP*(1-c) + CP*s)

  def rt_for_rotation_around_axis_through(self, point, angle, deg=False):
    assert self.n in ((3,1), (1,3))
    assert point.n in ((3,1), (1,3))
    r = (point - self).axis_and_angle_as_r3_rotation_matrix(
      angle=angle, deg=deg)
    return rt((r, self-r*self))

  def ortho(self):
    assert self.n in ((3,1), (1,3))
    x, y, z = self.elems
    a, b, c = abs(x), abs(y), abs(z)
    if c <= a and c <= b:
      return col((-y, x, 0))
    if b <= a and b <= c:
      return col((-z, 0, x))
    return col((0, -z, y))

  def rotate_around_origin(self, axis, angle, deg=False):
    assert self.n in ((3,1), (1,3))
    assert axis.n == self.n
    if deg: angle *= math.pi/180
    n = axis.normalize()
    x = self
    c, s = math.cos(angle), math.sin(angle)
    return x*c + n*n.dot(x)*(1-c) + n.cross(x)*s

  def rotate_2d(self, angle, deg=False):
    # implements right-hand rotation of the vector
    assert self.n in ((2,1),) # treats column vector only; easily extended to row
    if deg: angle *= math.pi/180
    c, s = math.cos(angle), math.sin(angle)
    rotmat = sqr((c,-s,s,c))
    return rotmat*self

  def rotate(self, axis, angle, deg=False):
    import warnings
    warnings.warn(
      message=
        "The .rotate() method has been renamed to .rotate_around_origin()"
        " for clarity. Please update the code calling this method.",
      category=DeprecationWarning,
      stacklevel=2)
    return self.rotate_around_origin(axis=axis, angle=angle, deg=deg)

  def vector_to_001_rotation(self,
        sin_angle_is_zero_threshold=1.e-10,
        is_normal_vector_threshold=1.e-10):
    assert self.n in ((3,1), (1,3))
    x,y,c = self.elems
    xxyy = x*x + y*y
    if (abs(xxyy + c*c - 1) > is_normal_vector_threshold):
      raise RuntimeError("self is not a normal vector.")
    s = math.sqrt(xxyy)
    if (s < sin_angle_is_zero_threshold):
      if (c > 0):
        return sqr((1,0,0,0,1,0,0,0,1))
      return sqr((1,0,0,0,-1,0,0,0,-1))
    us = y
    vs = -x
    u = us / s
    v = vs / s
    oc = 1-c
    return sqr((c + u*u*oc, u*v*oc, vs, u*v*oc, c + v*v*oc, -us, -vs, us, c))

  def outer_product(self, other=None):
    if (other is None): other = self
    assert self.n[0] == 1 or self.n[1] == 1
    assert other.n[0] == 1 or other.n[1] == 1
    result = []
    for a in self.elems:
      for b in other.elems:
        result.append(a*b)
    return rec(result, (len(self.elems), len(other.elems)))

  def cos_angle(self, other, value_if_undefined=None):
    self_norm_sq = self.norm_sq()
    if (self_norm_sq == 0): return value_if_undefined
    other_norm_sq = other.norm_sq()
    if (other_norm_sq == 0): return value_if_undefined
    d = self_norm_sq * other_norm_sq
    if (d == 0): return value_if_undefined
    return self.dot(other) / math.sqrt(d)

  def angle(self, other, value_if_undefined=None, deg=False):
    cos_angle = self.cos_angle(other=other)
    if (cos_angle is None): return value_if_undefined
    result = math.acos(max(-1,min(1,cos_angle)))
    if (deg): result *= 180/math.pi
    return result

  def rotation_angle(self, eps=1.e-6):
    """
    Assuming it is a rotation matrix, tr(m) = 1+2*cos(alpha)
    """
    assert self.is_r3_rotation_matrix(rms_tolerance=eps)
    arg = (self.trace()-1.)/2
    if(  arg<0 and abs(arg)>1 and abs(1-abs(arg))<1.e-6): arg=-1.
    elif(arg>0 and abs(arg)>1 and abs(1-abs(arg))<1.e-6): arg= 1.
    return math.acos(arg)*180./math.pi

  def accute_angle(self, other, value_if_undefined=None, deg=False):
    cos_angle = self.cos_angle(other=other)
    if (cos_angle is None): return value_if_undefined
    if (cos_angle < 0): cos_angle *= -1
    result = math.acos(min(1,cos_angle))
    if (deg): result *= 180/math.pi
    return result

  def is_square(self):
    return self.n[0] == self.n[1]

  def determinant(self):
    assert self.is_square()
    m = self.elems
    n = self.n[0]
    if (n == 1):
      return m[0]
    if (n == 2):
      return m[0]*m[3] - m[1]*m[2]
    if (n == 3):
      return   m[0] * (m[4] * m[8] - m[5] * m[7]) \
             - m[1] * (m[3] * m[8] - m[5] * m[6]) \
             + m[2] * (m[3] * m[7] - m[4] * m[6])
    flex = flex_proxy()
    if (flex is not None):
      m = flex.double(m)
      m.resize(flex.grid(self.n))
      return m.matrix_determinant_via_lu()
    return determinant_via_lu(m=self)

  def co_factor_matrix_transposed(self):
    n = self.n
    if (n == (0,0)):
      return rec(elems=(), n=n)
    if (n == (1,1)):
      return rec(elems=(1,), n=n)
    m = self.elems
    if (n == (2,2)):
      return rec(elems=(m[3], -m[1], -m[2], m[0]), n=n)
    if (n == (3,3)):
      return rec(elems=(
         m[4] * m[8] - m[5] * m[7],
        -m[1] * m[8] + m[2] * m[7],
         m[1] * m[5] - m[2] * m[4],
        -m[3] * m[8] + m[5] * m[6],
         m[0] * m[8] - m[2] * m[6],
        -m[0] * m[5] + m[2] * m[3],
         m[3] * m[7] - m[4] * m[6],
        -m[0] * m[7] + m[1] * m[6],
         m[0] * m[4] - m[1] * m[3]), n=n)
    assert self.is_square()
    raise RuntimeError("Not implemented.")

  def inverse(self):
    assert self.is_square()
    n = self.n
    if (n[0] < 4):
      determinant = self.determinant()
      assert determinant != 0
      return self.co_factor_matrix_transposed() / determinant
    flex = flex_proxy()
    if (flex is not None):
      m = flex.double(self.elems)
      m.resize(flex.grid(n))
      m.matrix_inversion_in_place()
      return rec(elems=m, n=n)
    numpy = numpy_proxy()
    if numpy:
      m = numpy.asarray(self.elems)
      m.shape = n
      m = numpy.ravel(numpy.linalg.inv(m))
      return rec(elems=m, n=n)
    return inverse_via_lu(m=self)

  def transpose(self):
    elems = []
    for j in xrange(self.n_columns()):
      for i in xrange(self.n_rows()):
        elems.append(self(i,j))
    return rec(elems, (self.n_columns(), self.n_rows()))

  def _mathematica_or_matlab_form(self,
        outer_open, outer_close,
        inner_open, inner_close, inner_close_follow,
        label,
        one_row_per_line,
        format,
        prefix):
    nr = self.n_rows()
    nc = self.n_columns()
    s = prefix
    indent = prefix
    if (label):
      s += label + "="
      indent += " " * (len(label) + 1)
    s += outer_open
    if (nc != 0):
      for ir in xrange(nr):
        s += inner_open
        for ic in xrange(nc):
          if (format is None):
            s += str(self(ir, ic))
          else:
            s += format % self(ir, ic)
          if (ic+1 != nc): s += ", "
          elif (ir+1 != nr or len(inner_open) != 0): s += inner_close
        if (ir+1 != nr):
          s += inner_close_follow
          if (one_row_per_line):
            s += "\n"
            s += indent
          s += " "
    return s + outer_close

  def mathematica_form(self,
        label="",
        one_row_per_line=False,
        format=None,
        prefix="",
        matrix_form=False):
    result = self._mathematica_or_matlab_form(
      outer_open="{", outer_close="}",
      inner_open="{", inner_close="}", inner_close_follow=",",
      label=label,
      one_row_per_line=one_row_per_line,
      format=format,
      prefix=prefix)
    if matrix_form: result += "//MatrixForm"
    result = result.replace('e', '*^')
    return result

  def matlab_form(self,
        label="",
        one_row_per_line=False,
        format=None,
        prefix=""):
    return self._mathematica_or_matlab_form(
      outer_open="[", outer_close="]",
      inner_open="", inner_close=";", inner_close_follow="",
      label=label,
      one_row_per_line=one_row_per_line,
      format=format,
      prefix=prefix)

  def __repr__(self):
    n0, n1 = self.n
    e = self.elems
    if (len(e) <= 3):
      e = str(e)
    else:
      e = "(%s, ..., %s)" % (str(e[0]), str(e[-1]))
    return "matrix.rec(elems=%s, n=(%d,%d))" % (e, n0, n1)

  def __str__(self):
    return self.mathematica_form(one_row_per_line=True)

  def as_list_of_lists(self):
    result = []
    nr,nc = self.n
    for ir in xrange(nr):
      result.append(list(self.elems[ir*nc:(ir+1)*nc]))
    return result

  def as_sym_mat3(self):
    assert self.n == (3,3)
    m = self.elems
    return (m[0],m[4],m[8],
            (m[1]+m[3])/2.,
            (m[2]+m[6])/2.,
            (m[5]+m[7])/2.)

  def as_mat3(self):
    assert self.n == (3,3)
    return self.elems

  def as_flex_double_matrix(self):
    flex = flex_proxy()
    assert flex is not None
    result = flex.double(self.elems)
    result.reshape(flex.grid(self.n))
    return result

  def as_flex_int_matrix(self):
    flex = flex_proxy()
    assert flex is not None
    result = flex.int(self.elems)
    result.reshape(flex.grid(self.n))
    return result

  def extract_block(self, stop, start=(0,0), step=(1,1)):
    assert 0 <= stop[0] <= self.n[0]
    assert 0 <= stop[1] <= self.n[1]
    i_rows = range(start[0], stop[0], step[0])
    i_colums = range(start[1], stop[1], step[1])
    result = []
    for ir in i_rows:
      for ic in i_colums:
        result.append(self(ir,ic))
    return rec(result, (len(i_rows),len(i_colums)))

  def __eq__(self, other):
    if self is other: return True
    if other is None: return False
    if issubclass(type(other), rec):
      return self.elems == other.elems
    for ir in xrange(self.n_rows()):
      for ic in xrange(self.n_columns()):
        if self(ir,ic) != other[ir,ic]: return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def resolve_partitions(self):
    nr,nc = self.n
    result_nr = 0
    for ir in xrange(nr):
      part_nr = 0
      for ic in xrange(nc):
        part = self(ir,ic)
        assert isinstance(part, rec)
        if (ic == 0): part_nr = part.n[0]
        else: assert part.n[0] == part_nr
      result_nr += part_nr
    result_nc = 0
    for ic in xrange(nc):
      part_nc = 0
      for ir in xrange(nr):
        part = self(ir,ic)
        if (ir == 0): part_nc = part.n[1]
        else: assert part.n[1] == part_nc
      result_nc += part_nc
    result_elems = [0] * (result_nr * result_nc)
    result_ir = 0
    for ir in xrange(nr):
      result_ic = 0
      for ic in xrange(nc):
        part = self(ir,ic)
        part_nr,part_nc = part.n
        i_part = 0
        for part_ir in xrange(part_nr):
          i_result = (result_ir + part_ir) * result_nc + result_ic
          for part_ic in xrange(part_nc):
            result_elems[i_result + part_ic] = part[i_part]
            i_part += 1
        result_ic += part_nc
      assert result_ic == result_nc
      result_ir += part_nr
    assert result_ir == result_nr
    return rec(elems=result_elems, n=(result_nr, result_nc))

class mutable_rec(rec):
  container_type = list

  def __setitem__(self, i, x):
    self.elems[i] = x

class row_mixin(object):

  def __init__(self, elems):
    super(row_mixin, self).__init__(elems, (1, len(elems)))

class row(row_mixin, rec): pass
class mutable_row(row_mixin, mutable_rec): pass

class col_mixin(object):

  def __init__(self, elems):
    super(col_mixin, self).__init__(elems, (len(elems), 1))

  def random(cls, n, a, b):
    uniform = random.uniform
    return cls([ uniform(a,b) for i in xrange(n) ])
  random = classmethod(random)

class col(col_mixin, rec):
    """
    Class type built on top of rec and col_mixin, allows for single-dimensional
    vectors.  This is especially convenient when working with 3D coordinate
    data in Python.

    Examples
    --------
    >>> from scitbx.matrix import col
    >>> vector = col([3, 0, 4])
    >>> print abs(vector)
    5.0
    """
    pass

class mutable_col(col_mixin, mutable_rec): pass

class sqr(rec):

  def __init__(self, elems):
    l = len(elems)
    n = int(l**(.5) + 0.5)
    assert l == n * n
    rec.__init__(self, elems, (n,n))

class diag(rec):

  def __init__(self, diag_elems):
    n = len(diag_elems)
    elems = [0 for i in xrange(n*n)]
    for i in xrange(n):
      elems[i*(n+1)] = diag_elems[i]
    rec.__init__(self, elems, (n,n))

class identity(diag):

  def __init__(self, n):
    super(identity, self).__init__((1,)*n)

class inversion(diag):

  def __init__(self, n):
    super(inversion, self).__init__((-1,)*n)

class sym(rec):

  def __init__(self, elems=None, sym_mat3=None):
    assert elems is None, "Not implemented."
    assert len(sym_mat3) == 6
    m = sym_mat3
    rec.__init__(self, (m[0], m[3], m[4],
                        m[3], m[1], m[5],
                        m[4], m[5], m[2]), (3,3))

def zeros(n, mutable=False):
  if mutable:
    col_t, rec_t = mutable_col, mutable_rec
  else:
    col_t, rec_t = col, rec
  if (isinstance(n, int)):
    return col_t(elems=(0,)*n)
  nr,nc = n
  return rec_t(elems=(0,)*(nr*nc), n=(nr,nc))

def mutable_zeros(n):
  return zeros(n, mutable=True)

def sum(iterable):
  """ The sum of the given sequence of matrices """
  sequence = iter(iterable)
  result = next(sequence)
  for m in sequence:
    result += m
  return result

def cross_product_matrix(vector):
  """\
Matrix associated with vector cross product:
  a.cross(b) is equivalent to cross_product_matrix(a) * b
Useful for simplification of equations. Used frequently in
robotics and classical mechanics literature.
"""
  v0, v1, v2 = vector
  return sqr((
      0, -v2,  v1,
     v2,   0, -v0,
    -v1,  v0,   0))

def linearly_dependent_pair_scaling_factor(vector_1, vector_2):
  assert len(vector_1) == len(vector_2)
  result = None
  for e1,e2 in zip(vector_1, vector_2):
    if (e1 == 0):
      if (e2 != 0): return None
    else:
      if (e2 == 0): return None
      m = e2 / e1
      if (result is None):
        result = m
      elif (result != m):
        return None
  if (result is None):
    return 0
  return result

def _dihedral_angle(sites, deg):
  assert len(sites) == 4
  d_01 = sites[0] - sites[1]
  d_21 = sites[2] - sites[1]
  d_23 = sites[2] - sites[3]
  n_0121 = d_01.cross(d_21)
  n_0121_norm = n_0121.length_sq()
  n_2123 = d_21.cross(d_23)
  n_2123_norm = n_2123.length_sq()
  if (n_0121_norm == 0 or n_2123_norm == 0):
    return None
  cos_angle = max(-1.,min(1.,
    n_0121.dot(n_2123) / (n_0121_norm * n_2123_norm)**0.5))
  result = math.acos(cos_angle)
  if (d_21.dot(n_0121.cross(n_2123)) < 0):
    result *= -1
  if (deg): result *= 180/math.pi
  return result

def dihedral_angle(sites, deg=False):
  flex = flex_proxy()
  if not flex:
    return _dihedral_angle(sites=sites, deg=deg)
  from scitbx.math import dihedral_angle
  return dihedral_angle(sites=sites, deg=deg)

def __rotate_point_around_axis(
      axis_point_1,
      axis_point_2,
      point,
      angle,
      deg=False):
  """About 6 times slower than the implementation below.
     Interestingly, a C++ implementation is still 2 times slower than
     the implementation below, due to the tuple-vec3 conversion overhead.
  """
  pivot = col(axis_point_1)
  axis = col(axis_point_2) - pivot
  r = axis.axis_and_angle_as_r3_rotation_matrix(angle=angle, deg=deg)
  return (r * (col(point) - pivot) + pivot).elems

def plane_equation(point_1, point_2, point_3):
  # plane equation: a*x + b*y * c*z + d = 0
  n = (point_2-point_1).cross(point_3-point_1)
  a,b,c = n
  d = -n.dot(point_1)
  return a,b,c,d

def distance_from_plane(xyz, points):
  """
  http://mathworld.wolfram.com/Point-PlaneDistance.html
  Given three points describing a plane and a fourth point outside the plane,
  return the distance from the fourth point to the plane.
  """
  assert (len(points) == 3)
  a,b,c,d = plane_equation(
    point_1=col(points[0]),
    point_2=col(points[1]),
    point_3=col(points[2]))
  x,y,z = xyz
  den = math.sqrt(a**2+b**2+c**2)
  if(den==0): return None
  return abs(a*x+b*y+c*z+d) / den

def all_in_plane(points, tolerance):
  assert len(points)>3
  for point in points[3:]:
    d = distance_from_plane(xyz=point, points=points[:3])
    if(d>tolerance): return False
  return True

def __project_point_on_axis(axis_point_1,axis_point_2, point):
  """
  Slow version based on flex arrays
  """
  from scitbx.array_family import flex
  a = flex.vec3_double([axis_point_1])
  b = flex.vec3_double([axis_point_2])
  p = flex.vec3_double([point])
  ab = b-a
  ap = p-a
  proj = a + (ap.dot(ab)/ab.dot(ab))*ab
  return proj

def project_point_on_axis(axis_point_1,axis_point_2, point):
  """
  Project a 3D coordinate on a given arbitrary axis.

  :param axis_point_1: tuple representing 3D coordinate at one end of the axis
  :param axis_point_2: tuple representing 3D coordinate at other end of the axis
  :param point: tuple representing 3D coordinate of starting point to rotate
  :param deg: Python boolean (default=False), specifies whether the angle is
    in degrees
  :returns: Python tuple (len=3) of rotated point
  """
  ax, ay, az = axis_point_1
  bx, by, bz = axis_point_2
  px, py, pz = point
  abx, aby, abz = (bx-ax, by-ay, bz-az)
  apx, apy, apz = (px-ax, py-ay, pz-az)
  ap_dot_ab = apx*abx+apy*aby+apz*abz
  ab_dot_ab = abx*abx+aby*aby+abz*abz
  brackets = ap_dot_ab/ab_dot_ab
  proj = (ax+brackets*abx, ay+brackets*aby,az+brackets*abz)
  return proj

def rotate_point_around_axis(
      axis_point_1,
      axis_point_2,
      point,
      angle,
      deg=False):
  """
  Rotate a 3D coordinate about a given arbitrary axis by the specified angle.

  :param axis_point_1: tuple representing 3D coordinate at one end of the axis
  :param axis_point_2: tuple representing 3D coordinate at other end of the axis
  :param point: tuple representing 3D coordinate of starting point to rotate
  :param angle: rotation angle (defaults to radians)
  :param deg: Python boolean (default=False), specifies whether the angle is
    in degrees
  :returns: Python tuple (len=3) of rotated point
  """
  if (deg): angle *= math.pi/180.
  xa,ya,za = axis_point_1
  xb,yb,zb = axis_point_2
  x,y,z = point
  xl,yl,zl = xb-xa,yb-ya,zb-za
  xlsq = xl**2
  ylsq = yl**2
  zlsq = zl**2
  dlsq = xlsq + ylsq + zlsq
  dl = dlsq**0.5
  ca = math.cos(angle)
  dsa = math.sin(angle)/dl
  oca = (1-ca)/dlsq
  xlylo = xl*yl*oca
  xlzlo = xl*zl*oca
  ylzlo = yl*zl*oca
  xma,yma,zma = x-xa,y-ya,z-za
  m1 = xlsq*oca+ca
  m2 = xlylo-zl*dsa
  m3 = xlzlo+yl*dsa
  m4 = xlylo+zl*dsa
  m5 = ylsq*oca+ca
  m6 = ylzlo-xl*dsa
  m7 = xlzlo-yl*dsa
  m8 = ylzlo+xl*dsa
  m9 = zlsq*oca+ca
  x_new = xma*m1 + yma*m2 + zma*m3 + xa
  y_new = xma*m4 + yma*m5 + zma*m6 + ya
  z_new = xma*m7 + yma*m8 + zma*m9 + za
  return (x_new,y_new,z_new)

class rt(object):
  """
  Object for representing associated rotation and translation matrices.  These
  will usually be a 3x3 matrix and a 1x3 matrix, internally represented by
  objects of type :py:class:`scitbx.matrix.sqr` and
  :py:class:`scitbx.matrix.col`.  Transformations may be applied to
  :py:mod:`scitbx.array_family.flex` arrays using the overloaded
  multiplication operator.

  Examples
  --------
  >>> from scitbx.matrix import rt
  >>> symop = rt((-1,0,0,0,1,0,0,0,-1), (0,0.5,0)) # P21 symop
  >>> from scitbx.array_family import flex
  >>> sites_frac = flex.vec3_double([(0.1,0.1,0.1), (0.1,0.2,0.3)])
  >>> sites_symm = symop * sites_frac
  >>> print list(sites_symm)
  [(-0.1, 0.6, -0.1), (-0.1, 0.7, -0.3)]
  """

  def __init__(self, tuple_r_t):
    if (hasattr(tuple_r_t[0], "elems")):
      self.r = sqr(tuple_r_t[0].elems)
    else:
      self.r = sqr(tuple_r_t[0])
    if (hasattr(tuple_r_t[1], "elems")):
      self.t = col(tuple_r_t[1].elems)
    else:
      self.t = col(tuple_r_t[1])
    assert self.r.n_rows() == self.t.n_rows()

  def __add__(self, other):
    if (isinstance(other, rt)):
      return rt((self.r + other.r, self.t + other.t))
    else:
      return rt((self.r, self.t + other))

  def __sub__(self, other):
    if (isinstance(other, rt)):
      return rt((self.r - other.r, self.t - other.t))
    else:
      return rt((self.r, self.t - other))

  def __mul__(self, other):
    if (isinstance(other, rt)):
      return rt((self.r * other.r, self.r * other.t + self.t))
    if (isinstance(other, rec)):
      if (other.n == self.r.n):
        return rt((self.r * other, self.t))
      if (other.n == self.t.n):
        return self.r * other + self.t
      raise ValueError(
        "cannot multiply %s by %s: incompatible number of rows or columns"
          % (repr(self), repr(other)))
    n = len(self.t.elems)
    if (isinstance(other, (list, tuple))):
      if (len(other) == n):
        return self.r * col(other) + self.t
      if (len(other) == n*n):
        return rt((self.r * sqr(other), self.t))
      raise ValueError(
        "cannot multiply %s by %s: incompatible number of elements"
          % (repr(self), repr(other)))
    if (n == 3):
      flex = flex_proxy()
      if flex and isinstance(other, flex.vec3_double):
        return self.r.elems * other + self.t.elems
    raise TypeError("cannot multiply %s by %s" % (repr(self), repr(other)))

  def inverse(self):
    r_inv = self.r.inverse()
    return rt((r_inv, -(r_inv*self.t)))

  def inverse_assuming_orthogonal_r(self):
    r_inv = self.r.transpose()
    return rt((r_inv, -(r_inv*self.t)))

  def as_float(self):
    return rt((self.r.as_float(), self.t.as_float()))

  def as_augmented_matrix(self):
    assert self.r.n_rows() == self.r.n_columns()
    n = self.r.n_rows()
    result = []
    for i_row in xrange(n):
      result.extend(self.r.elems[i_row*n:(i_row+1)*n])
      result.append(self.t[i_row])
    result.extend([0]*n)
    result.append(1)
    return rec(result, (n+1,n+1))

def col_list(seq): return [col(elem) for elem in seq]
def row_list(seq): return [row(elem) for elem in seq]

def lu_decomposition_in_place(a, n, raise_if_singular=True):
  is_singular_message = "lu_decomposition_in_place: singular matrix"
  assert len(a) == n*n
  vv = [0.] * n
  pivot_indices = [0] * (n+1)
  for i in xrange(n):
    big = 0.
    for j in xrange(n):
      dum = a[i*n+j]
      if (dum < 0.): dum = -dum
      if (dum > big): big = dum
    if (big == 0.):
      if (raise_if_singular):
        raise RuntimeError(is_singular_message)
      return None
    vv[i] = 1. / big
  imax = 0
  for j in xrange(n):
    for i in xrange(j):
      sum = a[i*n+j]
      for k in xrange(i): sum -= a[i*n+k] * a[k*n+j]
      a[i*n+j] = sum
    big = 0.
    for i in xrange(j,n):
      sum = a[i*n+j]
      for k in xrange(j): sum -= a[i*n+k] * a[k*n+j]
      a[i*n+j] = sum
      if (sum < 0.): sum = -sum
      dum = vv[i] * sum
      if (dum >= big):
        big = dum
        imax = i
    if (j != imax):
      for k in xrange(n):
        ik, jk = imax*n+k, j*n+k
        a[ik], a[jk] = a[jk], a[ik]
      pivot_indices[n] += 1
      vv[imax] = vv[j] # no swap, we don't need vv[j] any more
    pivot_indices[j] = imax
    if (a[j*n+j] == 0.):
      if (raise_if_singular):
        raise RuntimeError(is_singular_message)
      return None
    if (j+1 < n):
      dum = 1 / a[j*n+j]
      for i in xrange(j+1,n):
        a[i*n+j] *= dum
  return pivot_indices

def lu_back_substitution(a, n, pivot_indices, b, raise_if_singular=True):
  assert len(a) == n*n
  ii = n
  for i in xrange(n):
    pivot_indices_i = pivot_indices[i]
    if (pivot_indices_i >= n):
      if (raise_if_singular):
        raise RuntimeError(
          "lu_back_substitution: pivot_indices[i] out of range")
      return False
    sum = b[pivot_indices_i]
    b[pivot_indices_i] = b[i]
    if (ii != n):
      for j in xrange(ii,i): sum -= a[i*n+j] * b[j]
    elif (sum):
      ii = i
    b[i] = sum
  for i in xrange(n-1,-1,-1):
    sum = b[i]
    for j in xrange(i+1, n): sum -= a[i*n+j] * b[j]
    b[i] = sum / a[i*n+i]
  return True

def inverse_via_lu(m):
  assert m.is_square()
  n = m.n[0]
  if (n == 0): return sqr([])
  a = list(m.elems)
  pivot_indices = lu_decomposition_in_place(a=a, n=n)
  r = [0] * (n*n)
  for j in xrange(n):
    b = [0.] * n
    b[j] = 1.
    lu_back_substitution(a=a, n=n, pivot_indices=pivot_indices, b=b)
    for i in xrange(n):
      r[i*n+j] = b[i]
  return sqr(r)

def determinant_via_lu(m):
  assert m.is_square()
  n = m.n[0]
  if (n == 0): return 1 # to be consistent with other implemenations
  a = list(m.elems)
  pivot_indices = lu_decomposition_in_place(a=a, n=n, raise_if_singular=False)
  if (pivot_indices is None):
    return 0
  result = 1
  for i in xrange(n):
    result *= a[i*n+i]
  if (pivot_indices[-1] % 2):
    result = -result
  return result

if __name__ == "__main__":
  import scitbx.matrix.tst_matrix
  scitbx.matrix.tst_matrix.exercise_1()
  scitbx.matrix.tst_matrix.exercise_2()
