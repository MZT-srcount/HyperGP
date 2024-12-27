import setuptools  # 导入setuptools打包工具

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
  name = 'HyperGP',         # How you named your package folder (MyLib)
  packages = ['HyperGP'],   # Chose the same as "name"
  version = '0.1.01',      # Start with a small number and increase it with every change you make
  license='BSD-3-Clause',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'skip',   # Give a short description about your library
  author = 'Zhitong Ma',                   # Type in your name
  author_email = 'cszhitongma@mail.scut.edu.cn',      # Type in your E-Mail
  url = 'https://github.com/MZT-srcount/HyperGP',   # Provide either the link to your github or to your website
  # download_url = 'https://github.com/MZT-srcount/HyperGP.git',    # I explain this later on
  keywords = ['Genetic Programming', 'GPU Acceleration', 'Open-Source'],   # Keywords that define your package best
#   install_requires=[
#       ],
  classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: BSD License',
        "Operating System :: Unix",
    ],
)
