import  os
import sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    # first check if the CUDA_HOME env variable is in use
    if 'CUDA_HOME' in os.environ:
        home = os.environ['CUDA_HOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDA_HOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib': pjoin(home, 'lib')}

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# check for swig
if find_in_path('swig', os.environ['PATH']):
    subprocess.check_call('swig -python -c++ -o stream_wrap.cpp stream.i', shell=True)
else:
    raise EnvironmentError('the swig executable was not found in your PATH')


def customize_compiler_for_nvcc(self):    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# look at these HACKS, the .cu files are passed as arguments and removed after
cuda_sourcse = []
for i in range(2, len(sys.argv)):
    cuda_sourcse.append('../' + sys.argv[i])

sys.argv = [sys.argv[0], sys.argv[1]]

sources = ['stream_wrap.cpp', 'src/Stream.cpp']
sources.extend(cuda_sourcse)

darknet_home = os.environ['DARKNET_HOME']

# the actual swig compilation definition
ext = Extension('_stream',
                sources=sources,
                library_dirs=[CUDA['lib'], '../lib', darknet_home],
                libraries=['cudart', 'Stream', 'pthread', 'darknet'],
                runtime_library_dirs=[CUDA['lib']],
                # this syntax is specific to this build system
                # we're only going to use certain compiler args with nvcc and not with gcc
                # the implementation of this trick is in customize_compiler() below
                extra_compile_args={'gcc': ['-std=c++14'],
                                    'nvcc': ['-arch=sm_50', '--compiler-options', "'-fPIC'", '-std=c++14']},
                include_dirs = [numpy_include, CUDA['include'], 'src', '../src', pjoin(darknet_home, 'include')])

setup(name='stream',
      version='1.0',
      ext_modules = [ext],
      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext})
