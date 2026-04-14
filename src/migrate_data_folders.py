from pathlib import Path
import shutil

def main():
    """Migrates data from with_fyn folder to dataforsyningen and copernicus folders divided by region"""
    data_dir = Path(__file__).parent.parent / "data" 

    # removes duplicate data from no_fyn
    no_fyn = data_dir / "no_fyn"
    shutil.rmtree(no_fyn)

    # move data from with_fyn
    with_fyn = data_dir / "with_fyn"
    for file in with_fyn.glob("*/LR/*.tif"):
        if "train" in file.parts:
            dest = data_dir / "copernicus" / "jutland" / file.name
        elif "val" in file.parts:
            dest = data_dir / "copernicus" / "funen" / file.name
        else:
            dest = data_dir / "copernicus" / "zealand" / file.name

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(file, dest)
    
    for file in with_fyn.glob("*/HR/*.tif"):
        if "train" in file.parts:
            dest = data_dir / "dataforsyningen" / "jutland" / file.name
        elif "val" in file.parts:
            dest = data_dir / "dataforsyningen" / "funen" / file.name
        else:
            dest = data_dir / "dataforsyningen" / "zealand" / file.name

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(file, dest)

    # move merged files
    for file in with_fyn.glob("*.tif"):
        shutil.move(file, data_dir / file.name)

    # remove empty with_fyn folder
    shutil.rmtree(with_fyn)
    
if __name__ == "__main__":
    main()