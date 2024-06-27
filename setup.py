from setuptools import setup, find_packages

setup(
    name='simpub',
    version='0.1',
    install_requires=["zmq", "trimesh", "pillow", "numpy", "scipy"],
    include_package_data=True,
    packages = ['simpub', 'simpub.parser', 'simpub.sim', 'simpub.res']
)
