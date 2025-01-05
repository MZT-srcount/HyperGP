import os, glob
wheel_dir = os.path.join("./dist/", '*.whl')
print("wheel_dir: ", wheel_dir)
for wheel_file in glob.glob(wheel_dir):
    new_name = wheel_file.replace('HyperGP-0.1.1', 'HyperGP-0.1.1-19')
    os.rename(wheel_file, new_name)
    