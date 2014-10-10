# This file is generated by /Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/scipy_64/setup.py
# It contains system_info results at the time of building this package.
__all__ = ["get_info","show"]

blas_opt_info={'libraries': ['ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/lib'], 'language': 'c', 'include_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/include'], 'define_macros': [('ATLAS_INFO', '"\\"3.10.1\\""')]}
lapack_opt_info={'libraries': ['lapack', 'ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/lib'], 'language': 'f77', 'include_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/include'], 'define_macros': [('ATLAS_INFO', '"\\"3.10.1\\""')]}
atlas_blas_threads_info={'libraries': ['ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/lib'], 'language': 'c', 'include_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/include'], 'define_macros': [('ATLAS_INFO', '"\\"3.10.1\\""')]}
lapack_mkl_info={}
blas_mkl_info={}
mkl_info={}
atlas_threads_info={'libraries': ['lapack', 'ptf77blas', 'ptcblas', 'atlas'], 'library_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/lib'], 'language': 'f77', 'include_dirs': ['/Users/travis/build/matthew-brett/numpy-atlas-binaries/build/src/atlas_64/include'], 'define_macros': [('ATLAS_INFO', '"\\"3.10.1\\""')]}

def get_info(name):
    g = globals()
    return g.get(name, g.get(name + "_info", {}))

def show():
    for name,info_dict in globals().items():
        if name[0] == "_" or type(info_dict) is not type({}): continue
        print(name + ":")
        if not info_dict:
            print("  NOT AVAILABLE")
        for k,v in info_dict.items():
            v = str(v)
            if k == "sources" and len(v) > 200:
                v = v[:60] + " ...\n... " + v[-60:]
            print("    %s = %s" % (k,v))
    