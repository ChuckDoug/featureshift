# FeatureShift

Python script to automate the alignment of vector data to target, high-resolution images.

`python shifty.py -i /path/to/vectors.shp -d /directory/of/images/ -o /path/to/output.shp -x search_distance`

**-i (required)**: *Path to input Featureclass (GDB) or shapefile (.shp).* <br>
**-d (required)**: *Single level directory containing target image/s (.jpg, .jp2, .tif). Image/s must be in a projected coordinate system.* <br>
**-o (required)**: *Path to output shapefile (.shp).* <br>
**-x (optional)**: *Integer specifying the search distance in units associated with the target image/s. Default is 5 units.*  <br>
**-lv (optional)**: *Integer specifying the number of nieghbors used for local validation. Value must be greater than or equal to 5.* <br>
**-gv (optional)**: *Boolean flag to invoke global validation.* <br> 

*If both -lv and -gv are used, global validation will supersede local validation. If neither -lv or -gv are used, no validation will be implemented. <br>

# Getting Started
## Clone the Repository
`git clone git@code.ornl.gov:a4a/featureshift.git`

## Build the Docker Image
`cd featureshift` <br>
`docker build -t featureshift .`

## Run the Docker Image as a Container
`docker run -v /my/data/images/:/mnt/images -v /my/data/vectors/:/mnt/vectors featureshift -i /mnt/vectors/input.shp -d /mnt/images/ -o /mnt/vectors/output.shp -x 5`
