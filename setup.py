import os
import sys
from setuptools import setup, find_packages
from skbuild import setup as skbuild_setup
from skbuild.cmaker import CMaker
from skbuild.exceptions import SKBuildError

# 检查 CUDA 是否可用
def is_cuda_available():
    return True

# 获取 CUDA 路径
def get_cuda_path():
    cuda_path = os.getenv("CUDA_HOME", None)
    if cuda_path is None:
        cuda_path = "/usr/local/cuda"  # 默认 CUDA 路径
    print("CUDA PATH: ", cuda_path)
    return cuda_path

SUPPORTED_CUDA_VERSIONS = ["11.2.2"]#["10.1", "11.1", "11.4", "11.7","11.8", "12.0"]

# 定义 CMake 配置
# def get_cmake_args(cuda_version):
    
#     cuda_path = os.getenv("CUDA_HOME", None)
#     if cuda_path is None:
#         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
#         cuda_path = os.getenv("CONDA_PREFIX", "/usr/local/cuda")  # 默认使用 Conda 环境路径

#     cmake_args = [
#         f"-DPython_ROOT_DIR={os.path.dirname(sys.executable)}",
#         f"-DPYTHON_EXECUTABLE={sys.executable}",
#         f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_path}",
#         "-DCUDA_NVCC_FLAGS=--default-stream per-thread;-O3",
#     ]
#     return cmake_args

def get_cmake_args(cuda_version):
    cuda_path = os.getenv("CUDA_HOME", None)
    if cuda_path is None:
        cuda_path = "/usr/local/cuda"  # 默认 CUDA 路径

    cmake_args = [
        f"-DPython_ROOT_DIR={os.path.dirname(sys.executable)}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_path}",  # 使用 CUDA_HOME 设置 CUDA 路径
        "-DCUDA_NVCC_FLAGS=--default-stream per-thread;-O3",
    ]
    return cmake_args

# 动态生成版本号
version = "0.1.2-8"
local_version = os.getenv("LOCAL_VERSION", "1")  # 从环境变量获取本地版本号，默认为 "1"
full_version = f"{version}-{local_version}"

# 使用 scikit-build 进行构建
for cuda_version in SUPPORTED_CUDA_VERSIONS:
    try:
        skbuild_setup(
            name='HyperGP',#f'HyperGP_cuda_{cuda_version.replace(".", "_")}',
            version = version, 
            description="A high performance heterogeneous parallel GP framework",
            author = 'Zhitong Ma',                   # Type in your name
            author_email = 'cszhitongma@mail.scut.edu.cn',      # Type in your E-Mail
            url = 'https://github.com/MZT-srcount/HyperGP',   # Provide either the link to your github or to your website
            license='BSD-3-Clause', 
            packages=find_packages(),
            cmake_args=get_cmake_args(cuda_version),
            cmake_source_dir="./HyperGP/",  # CMakeLists.txt 所在的目录
            cmake_minimum_required_version="3.2",
            keywords = ['Genetic Programming', 'GPU Acceleration', 'Open-Source'],   # Keywords that define your package best
            install_requires=[
                'numpy',
                'dill',
                'matplotlib',
                'psutil',
                'tqdm'
            ],
            python_requires=">=3.9, <=3.13", 
            package_data={
                'HyperGP': ['*.so'],  # 包含 .so 文件
            },
            include_package_data=True,  # 确保包含非 Python 文件
        )
    except SKBuildError as e:
        print(f"An error occurred while building for CUDA {cuda_version}:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)