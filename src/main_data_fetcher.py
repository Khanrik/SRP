from dataforsyningen import Dataforsyningen
from copernicus import Copernicus
from helpers import BoundingBoxDegree

def main():
    # original testing area
    # midtjylland = BoundingBoxDegree(lon_min=9.0, lat_min=55.000277777777775, lon_max=9.999583333333334, lat_max=57.0)

    # bounding boxes found with this website https://bboxfinder.com/
    west_jutland = BoundingBoxDegree(lon_min=8.047, lat_min=54.908, lon_max=9.673, lat_max=57.169)
    east_jutland = BoundingBoxDegree(lon_min=9.674, lat_min=55.618, lon_max=10.953, lat_max=57.774)
    funen = BoundingBoxDegree(lon_min=9.689, lat_min=54.711, lon_max=10.975, lat_max=55.627)
    zealand = BoundingBoxDegree(lon_min=11.019, lat_min=54.619, lon_max=12.612, lat_max=56.130)
    # ethiopia = what part???

    lr_target_resolution = (128, 128)
    copernicus_west_jutland = Copernicus(aoi=west_jutland)
    copernicus_east_jutland = Copernicus(aoi=east_jutland)
    copernicus_funen = Copernicus(aoi=funen)
    copernicus_zealand = Copernicus(aoi=zealand)

if __name__ == "__main__":
    main()