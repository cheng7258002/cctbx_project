from iotbx import crystal_symmetry_from_any
from cctbx import crystal
from cctbx import sgtbx
from cctbx import uctbx
from libtbx.optparse_wrapper import option_parser, OptionError, make_option
import sys

class iotbx_option_parser(option_parser):

  def __init__(self, usage=None, description=None, more_help=None):
    option_parser.__init__(self, usage=usage, description=description)
    self.more_help = more_help
    self.show_defaults_callback = show_defaults_callback()
    self.symmetry_callback = symmetry_callback()
    self.chunk_callback = chunk_callback()

  def format_help (self, formatter=None):
    if formatter is None:
      formatter = self.formatter
    result = []
    if self.usage:
      result.append(self.get_usage() + "\n")
    result.append(self.format_option_help(formatter))
    if self.description:
      result.append("\n")
      result.append(self.format_description(formatter) + "\n")
      result.append("\n")
    if (self.more_help is not None):
      for line in self.more_help:
        result.append(line + "\n")
    return "".join(result)

  def show_help(self, f=None):
    if (f is None): f = sys.stdout
    f.write(self.format_help())

  def enable_show_defaults(self):
    self.add_option(make_option(None, "--show_defaults",
      action="callback",
      type="string",
      callback=self.show_defaults_callback,
      help="Print parameters visible at the given expert level and exit",
      metavar="EXPERT_LEVEL"))
    self.show_defaults_callback.is_enabled = True
    return self

  def enable_unit_cell(self):
    self.add_option(make_option(None, "--unit_cell",
      action="callback",
      type="string",
      callback=self.symmetry_callback,
      help="External unit cell parameters",
      metavar="10,10,20,90,90,120|FILENAME"))
    self.symmetry_callback.is_enabled = True
    return self

  def enable_space_group(self):
    self.add_option(make_option(None, "--space_group",
      action="callback",
      type="string",
      callback=self.symmetry_callback,
      help="External space group symbol",
      metavar="P212121|FILENAME"))
    self.symmetry_callback.is_enabled = True
    return self

  def enable_symmetry(self):
    self.add_option(make_option(None, "--symmetry",
      action="callback",
      type="string",
      callback=self.symmetry_callback,
      help="External file with symmetry information",
      metavar="FILENAME"))
    self.symmetry_callback.is_enabled = True
    return self

  def enable_symmetry_comprehensive(self):
    self.enable_unit_cell()
    self.enable_space_group()
    self.enable_symmetry()
    return self

  def enable_resolution(self, default=None):
    self.add_option(make_option(None, "--resolution",
      action="store",
      default=default,
      type="float",
      dest="resolution",
      help="High resolution limit (minimum d-spacing, d_min)",
      metavar="FLOAT"))
    return self

  def enable_low_resolution(self, default=None):
    self.add_option(make_option(None, "--low_resolution",
      action="store",
      default=default,
      type="float",
      dest="low_resolution",
      help="Low resolution limit (maximum d-spacing, d_max)",
      metavar="FLOAT"))
    return self

  def enable_resolutions(self, default_low=None, default_high=None):
    self.enable_resolution(default=default_high)
    self.enable_low_resolution(default=default_low)
    return self

  def enable_chunk(self):
    self.add_option(make_option(None, "--chunk",
      action="callback",
      type="string",
      callback=self.chunk_callback,
      help="Number of chunks for parallel execution and index for one process",
      metavar="n,i"))
    self.chunk_callback.is_enabled = True
    return self

  def process(self, args=None, nargs=None, min_nargs=None, max_nargs=None):
    if (self.show_defaults_callback.is_enabled
        and args is not None
        and len(args) > 0
        and args[-1] == "--show_defaults"):
      args = args + ["0"]
    assert nargs is None or (min_nargs is None and max_nargs is None)
    try:
      (options, args) = self.parse_args(args)
    except OptionError, e:
      print >> sys.stderr, e
      sys.exit(1)
    if (min_nargs is None): min_nargs = nargs
    if (min_nargs is not None):
      if (len(args) < min_nargs):
        if (len(args) == 0):
          self.show_help()
          sys.exit(1)
        self.error("Not enough arguments (at least %d required, %d given)." % (
          min_nargs, len(args)))
    if (max_nargs is None): max_nargs = nargs
    if (max_nargs is not None):
      if (len(args) > max_nargs):
        self.error("Too many arguments (at most %d allowed, %d given)." % (
          max_nargs, len(args)))
    return processed_options(self, options, args,
      show_defaults_callback=self.show_defaults_callback,
      symmetry_callback=self.symmetry_callback,
      chunk_callback=self.chunk_callback)

class processed_options(object):

  def __init__(self, parser, options, args,
        show_defaults_callback,
        symmetry_callback,
        chunk_callback):
    self.parser = parser
    self.options = options
    self.args = args
    self.expert_level = show_defaults_callback.expert_level
    self.symmetry = symmetry_callback.get()
    self.chunk_n = chunk_callback.n
    self.chunk_i = chunk_callback.i

class show_defaults_callback(object):

  def __init__(self):
    self.is_enabled = False
    self.expert_level = None

  def __call__(self, option, opt, value, parser):
    value = value.strip().lower()
    if (value == "all"):
      self.expert_level = -1
    else:
      try: value = int(value)
      except ValueError:
        raise OptionError('invalid value "%s"\n' % value
          + '  Please specify an integer value or the word "all"', opt)
      self.expert_level = value

class symmetry_callback(object):

  def __init__(self):
    self.is_enabled = False
    self.unit_cell = None
    self.space_group_info = None

  def __call__(self, option, opt, value, parser):
    if (opt == "--unit_cell"):
      unit_cell = None
      try: unit_cell = uctbx.unit_cell(value)
      except: pass
      if (unit_cell is not None):
        self.unit_cell = unit_cell
      else:
        crystal_symmetry = crystal_symmetry_from_any.extract_from(value)
        if (   crystal_symmetry is None
            or crystal_symmetry.unit_cell() is None):
          raise OptionError("cannot read parameters: " + value, opt)
        self.unit_cell = crystal_symmetry.unit_cell()
    elif (opt == "--space_group"):
      space_group_info = None
      space_group_info = sgtbx.space_group_info(symbol=value)
      try: space_group_info = sgtbx.space_group_info(symbol=value)
      except: pass
      if (space_group_info is not None):
        self.space_group_info = space_group_info
      else:
        crystal_symmetry = crystal_symmetry_from_any.extract_from(value)
        if (   crystal_symmetry is None
            or crystal_symmetry.space_group_info() is None):
          raise OptionError("unknown space group: " + value, opt)
        self.space_group_info = crystal_symmetry.space_group_info()
    elif (opt == "--symmetry"):
      crystal_symmetry = crystal_symmetry_from_any.extract_from(value)
      if (   crystal_symmetry is None
          or crystal_symmetry.space_group_info() is None):
        raise OptionError("cannot read symmetry: " + value, opt)
      if (crystal_symmetry.unit_cell() is not None):
        self.unit_cell = crystal_symmetry.unit_cell()
      if (crystal_symmetry.space_group_info() is not None):
        self.space_group_info = crystal_symmetry.space_group_info()
    else:
      raise RuntimeError, "Programming error."

  def get(self):
    return crystal.symmetry(
      unit_cell=self.unit_cell,
      space_group_info=self.space_group_info)

class chunk_callback(object):

  def __init__(self):
    self.is_enabled = False
    self.n = 1
    self.i = 0

  def __call__(self, option, opt, value, parser):
    assert opt == "--chunk"
    try:
      self.n, self.i = [int(i) for i in value.split(",")]
    except:
      raise OptionError(
        "Two comma-separated positive integers required.",
        opt)
    if (self.n < 1):
      raise OptionError(
        "First integer (number of chunks) must be greater than 0 (%d given)."
        % self.n, opt)
    if (self.i < 0):
      raise OptionError(
        "Second integer (index of chunks) must be positive (%d given)."
        % self.i, opt)
    if (self.n < self.i):
      raise OptionError(
        ("First integer (number of chunks, %d given) must be greater"
        + " than second integer (index of chunks, %d given).")%(self.n,self.i),
        opt)
