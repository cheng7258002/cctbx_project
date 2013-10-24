from __future__ import division
import boost.python
boost.python.import_ext("rstbx_indexing_api_ext")
from rstbx_indexing_api_ext import *
import rstbx_indexing_api_ext as ext
from rstbx.array_family import flex
from scitbx.matrix import col

class _(boost.python.injector, ext.dps_extended):

  def set_beam_vector(self,beam):  # currently self.beam is treated as the direction from sample to detector, but
                                   # this should be redone so it is sample to source.  S0  = -beam
    self.beam = beam
    self.beam_vector = beam # will be deprecated soon XXX
    self.inv_wave = self.beam.length() # will be deprecated soon XXX
    self.wavelength_set = 1./self.inv_wave

  def set_rotation_axis(self,axis):
    self.rotation_vector = axis
    self.axis = axis # will be deprecated soon XXX
    assert axis.length() == 1.0

  def set_detector(self,input_detector):
    self.detector = input_detector

  def set_detector_position(self,origin,d1,d2): # optional, alternate form
    from dxtbx.model.detector import detector_factory
    self.detector = detector_factory.make_detector(
      stype = "indexing",
      fast_axis = d1,
      slow_axis = d2,
      origin = origin,
      pixel_size = (1.0,1.0),  #not actually using pixels for indexing
      image_size = (100,100),  #not using pixels
      )
  @staticmethod
  def multicase(raw_spot_input,detector,inverse_wave,beam,axis,panelID):
    reciprocal_space_vectors = flex.vec3_double()
    origin = [col(d.get_origin()) for d in detector]
    d1     = [col(d.get_fast_axis()) for d in detector]
    d2     = [col(d.get_slow_axis()) for d in detector]

    # tile surface to laboratory transformation
    for n in xrange(len(raw_spot_input)):
      pid = panelID[n]
      lab_direct = origin[pid] + d1[pid] * raw_spot_input[n][0] + d2[pid] * raw_spot_input[n][1]

    # laboratory direct to reciprocal space xyz transformation
      lab_recip = (lab_direct.normalize() * inverse_wave) - beam

      reciprocal_space_vectors.append ( lab_recip.rotate_around_origin(
        axis=axis, angle=raw_spot_input[n][2], deg=True)
        )
    return reciprocal_space_vectors

  @staticmethod
  def raw_spot_positions_mm_to_reciprocal_space( raw_spot_input, # as vec3_double
      detector, inverse_wave, beam, axis, # beam, axis as scitbx.matrix.col
      panelID=None
      ):
    if panelID is not None:
      return ext.dps_extended.multicase(raw_spot_input,detector,inverse_wave,beam,axis,panelID)

    """Assumptions:
    1) the raw_spot_input is in the same units of measure as the origin vector (mm).
       they are not given in physical length, not pixel units
    2) the raw_spot centers of mass are given with the same corner/center convention
       as the origin vector.  E.g., spotfinder assumes that the mm scale starts in
       the middle of the lower-corner pixel.
    """

    reciprocal_space_vectors = flex.vec3_double()
    origin = col(detector.get_origin())
    d1     = col(detector.get_fast_axis())
    d2     = col(detector.get_slow_axis())

    # tile surface to laboratory transformation
    for n in xrange(len(raw_spot_input)):
      lab_direct = origin + d1 * raw_spot_input[n][0] + d2 * raw_spot_input[n][1]

    # laboratory direct to reciprocal space xyz transformation
      lab_recip = (lab_direct.normalize() * inverse_wave) - beam

      reciprocal_space_vectors.append ( lab_recip.rotate_around_origin(
        axis=axis, angle=raw_spot_input[n][2], deg=True)
        )
    return reciprocal_space_vectors

  def model_likelihood(self,separation_mm):
    TOLERANCE = 0.5
    fraction_properly_predicted = 0.0

    #help(self.detector)
    #print self.detector[0]
    #help(self.detector[0])
    panel = self.detector[0]
    from scitbx import matrix
    Astar = matrix.sqr(self.getOrientation().reciprocal_matrix())

    import math
    xyz = self.getXyzData()

    # step 1.  Deduce fractional HKL values from the XyzData.  start with x = A* h
    #          solve for h:  h = (A*^-1) x
    Astarinv = Astar.inverse()
    Hint = flex.vec3_double()
    for x in xyz:
      H = Astarinv * x
      #print "%7.3f %7.3f %7.3f"%(H[0],H[1],H[2])
      Hint.append((round(H[0],0), round(H[1],0), round(H[2],0)))
    xyz_miller = flex.vec3_double()
    from rstbx.diffraction import rotation_angles
    ra = rotation_angles(limiting_resolution=1.0,orientation = Astar,
                         wavelength = self.wavelength_set, axial_direction = self.rotation_vector)
    for ij,hkl in enumerate(Hint):
      xyz_miller.append( Astar * hkl ) # figure out how to do this efficiently on vector data
      if ra(hkl):
        omegas = ra.get_intersection_angles()
        rotational_diffs = [ abs((-omegas[omegaidx] * 180./math.pi)-self.raw_spot_input[ij][2])
                             for omegaidx in [0,1] ]
        min_diff = min(rotational_diffs)
        min_index = rotational_diffs.index(min_diff)
        omega = omegas[min_index]
        rot_mat = self.rotation_vector.axis_and_angle_as_r3_rotation_matrix(omega)

        Svec = (rot_mat * Astar) * hkl + self.beam_vector
#        print panel.get_ray_intersection(Svec), self.raw_spot_input[ij]
        if self.panelID is not None: panel = self.detector[ self.panelID[ij] ]
        calc = matrix.col(panel.get_ray_intersection(Svec))
        pred = matrix.col(self.raw_spot_input[ij][0:2])
#        print (calc-pred).length(), separation_mm * TOLERANCE
        if ((calc-pred).length() < separation_mm * TOLERANCE):
          fraction_properly_predicted += 1./ self.raw_spot_input.size()
    print "fraction properly predicted",fraction_properly_predicted,"with spot sep (mm)",separation_mm
    return fraction_properly_predicted
