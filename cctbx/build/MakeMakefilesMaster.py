# $Id$

import sys, os
from string import split, strip, maketrans, translate
transl_table_slash_backslash = maketrans("/", "\\")

class write_makefiles:

  def __init__(self, configuration):
    self.platform = strip(configuration[0])
    assert (self.platform in ("tru64_cxx", "linux_gcc", "irix_CC",
                              "mingw32", "vc60"))
    self.macros = configuration[1:]
    # remove empty lines at beginning
    while (len(strip(self.macros[0])) == 0): del self.macros[0]
    self.dependencies()

  def head(self):
    print r"""# Usage:
#
# Unix:
#
#   make softlinks     Create softlinks to source code and tests
#   make               Compile all sources
#   make clean         Remove all object files
#   make unlink        Remove softlinks
#
# Development option:
#   Rename this Makefile to Makefile.nodepend.
#   Run "make -f Makefile.nodepend depend > Makefile"
#   to automatically generate all dependencies.
#
# Windows:
#
#   make copy          Copy source code and tests
#   make               Compile all sources
#   make clean         Remove all object files
#   make del           Remove source code and tests
"""

    self.all = []
    self.depend = []
    self.clean = []
    for m in self.macros: print m
    print
    print "all: all_deferred"
    print

  def format_objects(self, objects):
    if (self.platform == "vc60"):
      doto = ".obj"
    else:
      doto = ".o"
    s = ""
    for obj in objects:
      s = s + " " + obj + doto
    return s[1:]

  def tail(self):

    if (len(self.all) != 0):
      s = "all_deferred:"
      for t in self.all: s = s + " " + t
      print s
      print

    if (hasattr(self, "make_test")):
      self.make_test()

    doto = self.format_objects(("",))
    print r"""
CPPOPTS=$(STDFIXINC) $(STDOPTS) $(WARNOPTS) $(OPTOPTS) \
        $(CCTBXINC) $(BOOSTINC) $(PYINC)

.SUFFIXES: %s .cpp

.cpp%s:
	$(CPP) $(CPPOPTS) -c $*.cpp
""" % (doto, doto)

    self.make_clean()

    if (self.platform != "vc60"):
      print "depend:"
      if (self.platform == "mingw32"):
        print "\t@type Makefile.nodepend"
      else:
        print "\t@cat Makefile.nodepend"
      for src in self.depend:
        print "\t@$(CPP) $(CPPOPTS) $(MAKEDEP) %s.cpp" % (src,)
      print

  def file_management(self):
    print "softlinks:"
    for srcf in self.files:
      print "\t-ln -s $(CCTBX_UNIX)/" + srcf + " ."
    print
    print "cp:"
    for srcf in self.files:
      print "\t-cp $(CCTBX_UNIX)/" + srcf + " ."
    print
    print "unlink:"
    for srcf in self.files:
      f = split(srcf, "/")[-1]
      print "\t-test -L %s && rm %s" % (f, f)
    print
    print "rm:"
    for srcf in self.files:
      print "\t-rm " + split(srcf, "/")[-1]
    print
    if (self.platform in ("mingw32", "vc60")):
      print "copy:"
      for srcf in self.files:
        f = translate(srcf, transl_table_slash_backslash)
        print "\t-copy $(CCTBX_WIN)\\" + f
      print
      print "del:"
      for srcf in self.files:
        print "\t-del " + split(srcf, "/")[-1]
      print

  def update_depend(self, objects):
    for obj in objects:
      if (not obj in self.depend):
        self.depend.append(obj)

  def make_library(self, name, objects):
    objstr = self.format_objects(objects)
    if (self.platform != "vc60"):
      lib = "lib" + name + ".a"
      print "%s: %s" % (lib, objstr)
      if (self.platform == "mingw32"):
        print "\t-del %s" % (lib,)
      else:
        print "\trm -f %s" % (lib,)
      if   (self.platform == "tru64_cxx"):
        print "\tar r %s %s cxx_repository/*.o" % (lib, objstr)
      elif (self.platform == "irix_CC"):
        print "\t$(CPP) -ar -o %s %s" % (lib, objstr)
      else:
        print "\tar r %s %s" % (lib, objstr)
    else:
      lib = "lib" + name + ".lib"
      print "%s: %s" % (lib, objstr)
      print "\t-del %s" % (lib,)
      print "\t$(LD) -lib /nologo /out:%s %s" % (lib, objstr)
    print
    self.all.append(lib)
    self.update_depend(objects)

  def make_executable(self, name, objects, libs = "$(LDMATH)"):
    objstr = self.format_objects(objects)
    if (not self.platform in ("mingw32", "vc60")):
      nameexe = name
    else:
      nameexe = name + ".exe"
    if (self.platform != "vc60"):
      out = "-o "
    else:
      out = "/out:"
    print "%s: %s" % (nameexe, objstr)
    print "\t$(LD) $(LDEXE) %s %s%s %s" % (objstr, out, nameexe, libs)
    print
    self.all.append(nameexe)
    self.update_depend(objects)
    self.clean.append(nameexe)

  def make_boost_python_module(self, name, objects):
    objstr = self.format_objects(objects)
    if   (self.platform == "mingw32"):
      self.mingw32_pyd(name, objstr)
    elif (self.platform == "vc60"):
      self.vc60_pyd(name, objstr)
    else:
      self.unix_so(name, objstr)
    print
    self.update_depend(objects)

  def unix_so(self, name, objstr):
    nameso = name + ".so"
    print "%s: %s" % (nameso, objstr)
    print "\t$(LD) $(LDDLL) -o %s %s $(BOOST_PYTHONLIB) $(PYLIB) $(LDMATH)" \
          % (nameso, objstr)
    print
    self.all.append(nameso)

  def mingw32_pyd(self, name, objstr):
    namepyd = name + ".pyd"
    namedef = name + ".def"
    print "%s: %s %s" % (namepyd, namedef, objstr)
    print (  "\tdllwrap -s --driver-name g++ --entry _DllMainCRTStartup@12"
           + " --target=i386-mingw32 --dllname %s --def %s"
           + " %s $(BOOST_PYTHONLIB) $(PYLIB)") % (namepyd, namedef, objstr)
    print
    print "%s:" % (namedef,)
    print "\techo EXPORTS > %s" % (namedef,)
    print "\techo \tinit%s >> %s" % (name, namedef)
    print
    self.all.append(namepyd)

  def vc60_pyd(self, name, objstr):
    namepyd = name + ".pyd"
    print "%s: %s" % (namepyd, objstr)
    print (  "\t$(LD) $(LDDLL) /out:%s /export:init%s %s"
           + " $(BOOST_PYTHONLIB) $(PYLIB)") % (namepyd, name, objstr)
    self.all.append(namepyd)

  def make_clean(self):
    print "clean_unix:"
    for f in self.clean:
      print "\trm -f " + f
    print "\trm -f *.o *.a *.so *.pyc"
    print "\trm -f *.obj *.lib *.exp *.idb *.exe *.def *.pyd"
    print "\trm -rf cxx_repository so_locations ii_files"
    print
    print "clean_win:"
    for f in self.clean:
      print "\t-del " + f
    for ext in ("o", "a", "so", "pyc",
                "obj", "lib", "exp", "idb", "exe", "def", "pyd"):
      print "\t-del *." + ext
    print
    if (self.platform in ("mingw32", "vc60")):
      print "clean: clean_win"
    else:
      print "clean: clean_unix"
    print

  def write(self, file):
    old_sys_stdout = sys.stdout
    sys.stdout = file
    try:
      self.head()
      if (hasattr(self, "libraries")):
        for name in self.libraries.keys():
          self.make_library(name, self.libraries[name])
      if (hasattr(self, "executables")):
        for name in self.executables.keys():
          self.make_executable(name, self.executables[name])
      if (hasattr(self, "examples")):
        for name in self.examples.keys():
          self.make_executable(name, self.examples[name],
                               "$(CCTBXLIB) $(LDMATH)")
      if (hasattr(self, "boost_python_modules")):
        for name in self.boost_python_modules.keys():
          self.make_boost_python_module(name, self.boost_python_modules[name])
      self.file_management()
      self.tail()
    finally:
      sys.stdout = old_sys_stdout
