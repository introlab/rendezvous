from distutils.core import setup, Extension
import numpy

name = 'virtual_camera_device'    # name of the module
version = '1.0'        # the module's version number

setup(name=name, version=version,
      # distutils detects .i files and compiles them automatically
      ext_modules=[Extension(name='_virtual_camera_device', # SWIG requires _ as a prefix for the module name
                             sources=['virtual_camera_device.i', 'src/VirtualCameraDevice.cpp'],
                             include_dirs=['src', '../include', numpy.get_include()],
                             library_dirs=['../../../../../../../lib/libv4l2cpp'],
                             libraries=['v4l2wrapper'],
                             swig_opts=['-c++'],
                             extra_compile_args=['-std=c++14'])
])
