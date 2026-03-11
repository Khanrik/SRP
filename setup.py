from setuptools import setup, find_packages
import pathlib

author = [("Henrik Hoangkhanh Huynh", "henrik.huynh@yahoo.dk"),("Cecilia Cvitanich Fisher", "cecicvitf@gmail.com")]

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(name='SRP',
        version='1.0',
        description='Super Resolution Project for Copernicus Data',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author=author,
        url='https://github.com/Khanrik/SRP',
        packages=['srp','srp.*'],
        install_requires=[
            'xarray',
            'xarray-spatial',
            'rasterio',
            'requests',
            'matplotlib',
            'numpy',
            'scikit-learn',
            'scipy',
            'torch',
            'tqdm',
            'pyproj',
            'python-dotenv',
            'rioxarray',
            'datashader',
            'torchvision',
            'Pillow',
            'pystac-client',
            'planetary-computer'
        ],
        )
