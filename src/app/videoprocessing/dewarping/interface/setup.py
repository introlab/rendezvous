from distutils.core import setup, Extension
import numpy

name = 'fisheye_dewarping'    # name of the module
version = '1.0'        # the module's version number
libpath = '../../../../../lib'

setup(name=name, version=version,
      # distutils detects .i files and compiles them automatically
      ext_modules=[Extension(name='_fisheye_dewarping', # SWIG requires _ as a prefix for the module name
                             sources=['fisheye_dewarping.i', 'src/FisheyeDewarping.cpp'],
                             include_dirs=['src', '../include', libpath + '/glfw-3.2.1/include', libpath + '/glm/include', libpath + '/glad/include', numpy.get_include()],
			           library_dirs=['../lib', libpath + '/glfw-3.2.1/build/src'],
			           libraries=['Dewarper', 'GL', 'GLU', 'glfw3', 'X11', 'Xxf86vm', 'Xrandr', 'pthread', 'Xi', 'dl', 'Xinerama', 'Xcursor'],
                             swig_opts=['-c++'],
			           extra_compile_args=['-std=c++14'])
])
