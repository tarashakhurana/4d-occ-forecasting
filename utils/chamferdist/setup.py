import os
import runpy
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension


def get_extensions():
    include_dirs = ["chamferdist"]
    main_source = os.path.join("chamferdist", "ext.cpp")
    sources = [os.path.join("chamferdist", "knn_cpu.cpp")]
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": ["-std=c++14"]}
    define_macros = []

    ext_modules = [
        extension(
            "chamferdist._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Retrieve __version__ from the package.
__version__ = runpy.run_path("chamferdist/version.py")["__version__"]

if os.getenv("NO_NINJA", "0") == "1":

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)


else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension

package_name = "chamferdist"
long_description = "A pytorch module to compute Chamfer distance \
                    between two point sets (pointclouds)."

setup(
    name="chamferdist",
    version=__version__,
    description="Pytorch Chamfer distance",
    packages=find_packages(),
    package_data={'chamferdist': ['*.h']},
    long_description=long_description,
    install_requires=[],
    extras_require={
        "dev": ["black", "flake8", "isort"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
