# FeatureShift

Python code to automate the alignment of vector data to target, high-resolution images. Details regarding the impetus for this project
can be found in the publication [Automated Registration of Vector Data to Overhead Imagery](https://ieeexplore.ieee.org/document/9554510).

# Getting Started
## Clone the Repository
```
git clone git@github.com:ChuckDoug/featureshift.git
```

## Build the Docker Image
```
cd featureshift
docker build -t featureshift .
```

## Run the Docker Image as a Container

```
docker run -v /my/data/images/:/mnt/images -v /my/data/vectors/:/mnt/vectors featureshift -i /mnt/vectors/input.shp -t /mnt/images/image_1.tif -o /mnt/vectors/output.shp -x 5
```

### Parameters
**-i (required)**: *Path to input Featureclass (GDB) or shapefile (.shp).* <br>
**-t (required)**: *List of target image/s (.jpg, .jp2, .tif). Image/s must be in a projected coordinate system.* <br>
**-o (required)**: *Path to output shapefile (.shp).* <br>
**-x (optional)**: *Integer specifying the search distance in units associated with the target image/s. Default is 5 units.*  <br>
**-lv (optional)**: *Integer specifying the number of nieghbors used for local validation. Value must be greater than or equal to 5.* <br>
**-gv (optional)**: *Boolean flag to invoke global validation.* <br> 

*If both -lv and -gv are used, global validation will supersede local validation. If neither -lv or -gv are used, no validation will be implemented. <br>
