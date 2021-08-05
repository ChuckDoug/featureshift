from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
import pandas as pd
from scipy import spatial
import numpy as np
import cv2 as cv
import argparse
import tempfile
import shutil
import math
import os
import time
import sys


def parse_args():
    """ Function for command line compatibility """

    usage = "./shifty.py"

    parser = argparse.ArgumentParser(usage)

    parser.add_argument("-i", "--input",
                        action  = "store",
                        dest    = "inputFeat",
                        help    = "Path to input Featureclass (GDB) or shapefile (.shp).",
                        type    = str,
                        required = True)

    parser.add_argument("-t", "--target",
                        action  = "store",
                        dest    = "targets",
                        help    = "List of target image/s (.jpg, .jp2, .tif). Image/s must be in a projected coordinate system.",
                        type    = str,
                        nargs = '+',
                        required = True)

    parser.add_argument("-o", "--output",
                        action  = "store",
                        dest    = "outputFeat",
                        help    = "Path to output shapefile (.shp).",
                        type    = str,
                        required = True)

    parser.add_argument("-x", "--distance",
                        action  = "store",
                        dest    = "distance",
                        type    = int,
                        required = False,
                        help    = "Integer specifying the search distance in units associated with the target image/s. Default is 5 units.",
                        default = 5)

    parser.add_argument("-lv", "--local-validation",
                        action  = "store",
                        dest    = "localVal",
                        type    = int,
                        required = False,
                        help    = "Integer specifying the number of nieghbors used for local validation. Value must be greater than or equal to 5.",
                        default = None)

    parser.add_argument("-gv", "--global-validation",
                        action  = "store_true",
                        dest    = "globalVal",
                        required = False,
                        help    = "Boolean flag to invoke global validation.",
                        default = False)

    args = parser.parse_args()

    return args


class Image(object):
    def __init__(self, path_to_image):
        self.source = path_to_image
        self._get_properties()
        #self.get_statistics()
    
    def _get_properties(self):
        """ Private method to validate raster and retrieve properties """
        
        ds = gdal.Open(self.source)
        
        if ds:
            
            self.cellX = ds.GetGeoTransform()[1]
            self.cellY = ds.GetGeoTransform()[5]
            self.cols = ds.RasterXSize
            self.rows = ds.RasterYSize
            self.extent = (ds.GetGeoTransform()[0], 
                           ds.GetGeoTransform()[0] + self.cols * self.cellX,
                           ds.GetGeoTransform()[3] + self.rows * self.cellY, 
                           ds.GetGeoTransform()[3])
            self.spatialRef = ds.GetProjection()
            self.bands = ds.RasterCount
            self.dataType = ds.GetRasterBand(1).DataType        
            self._cell_area()
            
            ds = None
            
        else:
            raise OSError( "Unable to open: {0}".format(self.source) )
        
    
    def _cell_area(self):
        """ Private method to calculate area of each cell """
        
        if osr.SpatialReference(self.spatialRef).IsGeographic:
            if os.path.exists(os.path.splitext(self.source)[0] + ".IMD"):
                with open(os.path.splitext(self.source)[0] + ".IMD", 'r') as handle:
                    for row in handle:
                        if row.lstrip().startswith("meanCollectedGSD"):
                            self.cellArea = eval(row.split("=")[-1].lstrip().rstrip(";\n")) ** 2
                            break
            else:
                self.cellArea = None
        else:
            self.cellArea = self.cellX * self.cellY 
            
        
    def _ms_to_is(self, point):
        """ Private method to convert between map space and image space"""
        
        col = int((point[0] - self.extent[0]) / self.cellX)
        row = int((point[1] - self.extent[3]) / self.cellY)

        if 0 <= col <= self.cols:
            if 0 <= row <= self.rows:
                return col, row
            else:
                raise ValueError( "Point exceeds 'y' dimension of image." )
        else:
            raise ValueError( "Point exceeds 'x' dimension of image." )
        
    
    def extent_as_geom(self, extent):
        """ Method to return the extent of a raster as a geometry object.
            The extent parameter is a tuple representing (xmin, xmax, ymin, ymax) """
    
        ring = ogr.Geometry(ogr.wkbLinearRing)
        
        ring.AddPoint(extent[0], extent[3])
        ring.AddPoint(extent[1], extent[3])
        ring.AddPoint(extent[1], extent[2])
        ring.AddPoint(extent[0], extent[2])
        ring.AddPoint(extent[0], extent[3])
            
        geom = ogr.Geometry(ogr.wkbPolygon)
        geom.AddGeometry(ring)
        
        return geom
    
    
    def virtual_tile(self, windowx, windowy, origin=None):
        """ Method to create virtual subsets (offsets) a raster """    
    
        # Create container to hold arguments
        self.tiles = []     
        
        # Set counters, rows, and columns (varies with user input)
        if origin:
            offx, offy = self._ms_to_is(origin)
            
        else:
            offx = 0
            offy = 0
        
        for y in np.arange(offy, self.rows, windowy):
            for x in np.arange(offx, self.cols, windowx):
                
                if x + windowx > self.cols:
                    winx = self.cols - x
                else:
                    winx = windowx
                if y + windowy > self.rows:
                    winy = self.rows - y
                else:
                    winy = windowy
                    
                self.tiles.append((x, y, winx, winy))        
        
                                    
    def create_grid(self, windowx, windowy, output_shape, origin=None):
        """ Method to create a shapefile template of a tiled raster """
        
        if not os.path.exists(output_shape):
        
            self.virtual_tile(windowx, windowy, origin)
            
            # Set up the shapefile driver
            drv = ogr.GetDriverByName("ESRI Shapefile")
            
            # Create the data source
            ds = drv.CreateDataSource(output_shape)
            
            # Create the spatial reference
            srs = osr.SpatialReference(self.spatialRef)
            
            # Create layer
            lyr = ds.CreateLayer("tiles", srs, ogr.wkbPolygon)
            
            # Add standard fields
            fn = ogr.FieldDefn("Tile", ogr.OFTInteger)
            lyr.CreateField(fn)
            fn = ogr.FieldDefn("Row", ogr.OFTInteger)
            lyr.CreateField(fn)    
            fn = ogr.FieldDefn("Col", ogr.OFTInteger)
            lyr.CreateField(fn)
            fn = ogr.FieldDefn("OffSet", ogr.OFTString)
            lyr.CreateField(fn)
            fn = ogr.FieldDefn("WinSize", ogr.OFTString)
            lyr.CreateField(fn)
    
            # Iterate over polygon objects and add to the layer
            c = 0
            
            for tile in self.tiles:
                feat = ogr.Feature(lyr.GetLayerDefn())   
                
                # Set the feature geometry 
                e = (self.extent[0] + (tile[0] * self.cellX),
                     self.extent[0] + ((tile[0] + tile[2]) * self.cellX),
                     self.extent[3] + (tile[1] * self.cellY),
                     self.extent[3] + ((tile[1] + tile[3]) * self.cellY))
                
                feat.SetGeometry(self.extent_as_geom(e))
                
                # Set the attributes 
                feat.SetField("Tile", c)
                feat.SetField("Row", tile[1])
                feat.SetField("Col", tile[0])
                feat.SetField("OffSet", str((tile[0], tile[1])))
                feat.SetField("WinSize", str((tile[2], tile[3])))
                
                # Create the feature in the layer (shapefile)
                lyr.CreateFeature(feat)
                
                # Destroy the feature to free resources
                feat.Destroy()
                
                c += 1
            
            # Destroy the data source to free resources
            ds.Destroy()
        
        else:
            raise OSError( "{0} already exists!".format(output_shape) ) 
        

class Tile(Image):
    def __init__(self, path_to_image, offsetx, offsety, windowx, windowy):
        self.source = path_to_image
        self._get_properties()
        self.offX = offsetx 
        self.offY = offsety 
        self.winX = windowx
        self.winY = windowy
        self._validate_extent()
        self._tile_extent()
        self._geotransform()
    
    
    def _validate_extent(self):
        """ Private method to ensure the requested tile is within the parent image """
        
        if self.offX < 0 or self.offY < 0:
            raise NotImplementedError( "Negative indexing is not supported!" )
        
        if self.offX + self.winX > self.cols:
            raise ValueError( "Tile exceeds image in 'x' dimension!" )
            
        if self.offY + self.winY > self.rows:
            raise ValueError( "Tile exceeds image in 'y' dimension!" )
        
    
    def _tile_extent(self):
        """ Private method to generate extent for a tile from a parent image """
            
        self.minX = self.extent[0] + (self.offX * self.cellX)
        self.maxY = self.extent[3] + (self.offY * self.cellY)
        self.maxX = self.minX + (self.winX * self.cellX)
        self.minY = self.maxY + (self.winY * self.cellY)
    
        self.tile_extent = (self.minX, self.maxX, 
                            self.minY, self.maxY)
    
    
    def _geotransform(self):
        """ Private method to attribute geotransform for a tile of a parent image """
        
        self.geotransform = (self.minX, self.cellX, 0, 
                             self.maxY, 0, self.cellY)
    
    
    def _adjust_edges(self, size=3):
        """ Private method to solve edge effects """
        
        if size % 2 == 1:
            
            adj = size/2

            if self.offX > 0:
                if self.offX + self.winX < self.cols - 1 - adj:
                    self._adjOffX = self.offX - adj
                    self._adjWinX = self.winX + (2*adj)
                    self._adjStartX = adj
                    self._adjStopX = -adj
                else:
                    self._adjOffX = self.offx - adj
                    self._adjWinX = self.winX + adj
                    self._adjStartX = adj
                    self._adjStopX = None                
            else:
                self._adjOffX = self.offX
                if self.winX == self.cols:
                    self._adjWinX = self.winX
                    self._adjStartX = None
                    self._adjStopX = None         
                else:
                    self._adjWinX = self.winX + adj
                    self._adjStartX = None
                    self._adjStopX = -adj         
                    
            if self.offY > 0:
                if self.offY + self.winY < self.rows - 1 - adj:
                    self._adjOffY = self.offY - adj
                    self._adjWinY = self.winY + (2*adj)
                    self._adjStartY = adj
                    self._adjStopY = -adj              
                else:
                    self._adjOffY = self.offY - adj
                    self._adjWinY = self.winY + adj
                    self._adjStartY = adj
                    self._adjStopY = None                
            else:
                self._adjOffY = self.offY
                if self.winY == self.rows:
                    self._adjWinY = self.winY
                    self._adjStartY = None
                    self._adjStopY = None    
                else:
                    self._adjWinY = self.winY + adj
                    self._adjStartY = None
                    self._adjStopY = -adj                    
        else:
            raise ValueError( "Matrix size must be of odd dimensions..." )

    
    def read_tile(self, adjust=None, band=None):
        """ Method to return array of pixel values """
        
        ds = gdal.Open(self.source)
        
        if band:
            
            b = ds.GetRasterBand(band)
            
            if adjust:
                self._adjust_edges(size = adjust)
                ar = b.ReadAsArray(self._adjOffX, self._adjOffY, 
                                   self._adjWinX, self._adjWinY)
                
            else:
                ar = b.ReadAsArray(self.offX, self.offY, 
                                   self.winX, self.winY)
                
        else:
                    
            if adjust:
                self._adjust_edges(size = adjust)
                ar = ds.ReadAsArray(self._adjOffX, self._adjOffY, 
                                    self._adjWinX, self._adjWinY)
            else:
                ar = ds.ReadAsArray(self.offX, self.offY, 
                                    self.winX, self.winY)            
        b = None
        ds = None
        
        return ar
    
        
    def save(self, array=None, output=None):
        """ Method to save array using the tile parameters """
        
        if output:
        
            if os.path.isdir(output):
                output = os.path.join(output, self.name)
                
            if array is not None:
                ar = array
                if len(ar.shape) > 2:
                    bands = ar.shape[0]
                else: 
                    bands = 1
            else:
                ar = self.read_tile()
                bands = self.bands
            
            dtype = gdal_array.NumericTypeCodeToGDALTypeCode(ar.dtype)
            
            driver = gdal.GetDriverByName("GTiff")
            
            # Create output and specify raster properties
            dst = driver.Create(output, self.winX, self.winY, 
                                bands, dtype, options=['COMPRESS=LZW'])
            dst.SetProjection(self.spatialRef)
            dst.SetGeoTransform(self.geotransform)
            
            if bands == 1:
                band = dst.GetRasterBand(1)
                band.WriteArray(ar)
            else:
                for i in range(bands):
                    band = dst.GetRasterBand(i + 1)
                    band.WriteArray(ar[i])
               
            # Free memory
            ar = None
            dst.FlushCache()
            dst = None
        
        else:
            raise ValueError( "No output specified..." )
            
            
class Shift(object):
    def __init__(self, input_features, target_images, output_features, search_distance=5, local_validation=None, global_validation=False):
        self.images = target_images
        self.in_feat = input_features
        self.out_feat = output_features
        self.search_distance = search_distance
        self.local_val = local_validation
        self.global_val = global_validation
        self._validate_args()
        self.field = "_SID_"


    def _validate_args(self):
        """ Private method to validate user specified inputs """
        
        self._validate_input()
        
        self._validate_output()
        
        self._validate_targets()

        self._validate_validation()
        
        self.workspace = tempfile.mkdtemp()
        
        
    def _validate_input(self):
        """ Private method to validate input features """
        
        if ".gdb" in self.in_feat:
            srcds = ogr.Open(os.path.dirname(self.in_feat))
            
            if srcds:
                srclyr = srcds.GetLayerByName(os.path.basename(self.in_feat))
                
                if srclyr:
                    srclyr = None
                    srcds = None
                    
                else:
                    raise OSError( "Unable to open layer: {0}".format(os.path.basename(self.in_feat)) )            
            
            else:
                raise OSError( "Unable to open GDB: {0}".format(os.path.dirname(self.in_feat)) )            
            
        else:
            srcds = ogr.Open(self.in_feat)
            
            if srcds:
                srcds = None
            
            else:
                raise OSError( "Unable to open file: {0}".format(self.in_feat) )        


    def _validate_targets(self):
        """ Private method to validate target images """

        for i in self.images:
            if i.lower().endswith((".tif", ".jpg", ".jp2")):
                pass
            else:
                raise ValueError( "Incompatible file type for {}".format(i) )
        
        
    def _validate_output(self):
        """ Private method to validate output features """
        
        if not os.path.exists(self.out_feat):
    
            if self.out_feat.endswith(".shp"):
                pass
    
            else:
                raise ValueError( "Output shapefile {0} is not a valid shapefile format!".format(self.out_feat) )
    
            if not os.path.exists(os.path.dirname(self.out_feat)):
                os.makedirs(os.path.dirname(self.out_feat))
                
        else:
            raise OSError( "Output shapefile {0} already exists!".format(self.out_feat) )
        

    def _validate_validation(self):
        """ Private method to validate validation technique """

        if self.global_val and self.local_val:

            self.local_val = None

        if self.local_val:
            
            if self.local_val < 5:
                raise ValueError( "Number of neighbors for local validation must be greater than 5!" )


    def _outliers(self, array):
        """ Private method to identify bounds for outliers (+- 1 standard deviation) """
        
        m = np.mean(array)
        std = np.std(array)
        
        lo = m - std
        uo = m + std
        
        return m, std, lo, uo
    
        
    def _quartiles(self, array):
        """ Private method to calculate median, interquartile range, outlier threshold """

        array = sorted(array)

        m = np.median(array)

        if len(array) % 2 == 0:
            ind = len(array)//2
        else:
            ind = (len(array)//2) + 1

        q1 = np.median(array[:len(array)//2])        
        q3 = np.median(array[len(array)//2:])

        iqr = q3 - q1

        uo = q3 + (1.5 * iqr)
        lo = q1 - (1.5 * iqr)        

        return m, q1, q3, iqr, lo, uo
    
    
    def _rasterize(self, infile, cellsize, outfile, extent=None):
        """ Method to rasterize a layer. If the outfile
        is "MEM", a raster dataset will be returned.  Otherwise
        the raster is written to disk.
        """
        
        if not os.path.exists(outfile):
    
            if ".gdb" in infile:
                drv = ogr.GetDriverByName("OpenFileGDB")
                srcDs = drv.Open(os.path.dirname(infile))
                
            elif infile.endswith(".shp"):
                drv = ogr.GetDriverByName("ESRI SHAPEFILE")
                srcDs = drv.Open(infile)
        
            else:
                srcDs = None
                
            if srcDs:
    
                srcLyr = srcDs.GetLayer(os.path.basename(infile).split(".")[0])
    
                srcSr = srcLyr.GetSpatialRef().ExportToWkt() 
                
                if extent:
               
                    srcLyr.SetSpatialFilterRect(extent[0], extent[2], 
                                                extent[1], extent[3])
                    
                rows = int(math.ceil((extent[3] - extent[2])/cellsize))
                cols = int(math.ceil((extent[1] - extent[0])/cellsize))
                
                if outfile.upper() == "MEM":
                    rasDrv = gdal.GetDriverByName('MEM')
                    opts = []
                    
                else:
                    rasDrv = gdal.GetDriverByName('GTiff')
                    opts = ['COMPRESS=LZW']
                
                trgDs = rasDrv.Create(outfile, cols, rows, 
                                      1, gdal.GDT_Byte)
                trgDs.SetProjection(srcSr)
                trgDs.SetGeoTransform((extent[0], cellsize, 0, 
                                       extent[3], 0, -cellsize))    
                gdal.RasterizeLayer(trgDs, [1], srcLyr, None, None, [255])
                
                trgDs.FlushCache()
                
                if outfile.upper() == "MEM":
                    return trgDs.ReadAsArray()
                else:
                    trgDs = None
            srcDs = None
        
        
    def _extent_as_geom(self, xmin, xmax, ymin, ymax):
        """ Private method to return the extent of a raster as a geometry object """
    
        ring = ogr.Geometry(ogr.wkbLinearRing)
        
        ring.AddPoint(xmin, ymax)
        ring.AddPoint(xmax, ymax)
        ring.AddPoint(xmax, ymin)
        ring.AddPoint(xmin, ymin)
        ring.AddPoint(xmin, ymax)
            
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        
        return poly


    def _create_index(self, list_of_images, output_index):
        """ Private method to create spatial index of rasters """
        
        if not os.path.exists(output_index):
                    
            if list_of_images:
            
                drv = ogr.GetDriverByName("ESRI Shapefile")
                ds = drv.CreateDataSource(output_index)
                
                ids = gdal.Open(list_of_images[0])
                srs = osr.SpatialReference(ids.GetProjectionRef())            
                ids = None
            
                lyr = ds.CreateLayer("index", srs, ogr.wkbPolygon)
            
                fn = ogr.FieldDefn("tile", ogr.OFTInteger)
                lyr.CreateField(fn)  
                
                c = 0
                
                for image in list_of_images:
                    ids = gdal.Open(image)
                
                    gt = ids.GetGeoTransform()
            
                    xsize, ysize = ids.RasterXSize, ids.RasterYSize
            
                    xmin = gt[0]
                    xmax = gt[0] + (xsize * gt[1])
                    ymin = gt[3] + (ysize * gt[5])
                    ymax = gt[3]
            
                    ids = None
                    
                    geom = self._extent_as_geom(xmin, xmax, ymin, ymax)
                    
                    feat = ogr.Feature(lyr.GetLayerDefn())   
                    feat.SetGeometry(geom)
                    feat.SetField("tile", c)
                    lyr.CreateFeature(feat)
                    feat.Destroy()
                    
                    c += 1
            
                ds.Destroy()             
            
            else:
                raise OSError( "No images found!" )
            
        else:
            raise OSError( "{0} already exists!".format(output_index) )
            
    
    def _filter_transform(self, input_features, mask, output_features):
        """ Private method to subset and transform features based on extent of rasters """
        
        if not os.path.exists(output_features):
            
            mds = ogr.Open(mask)
            
            if mds:
                mlyr = mds.GetLayer()
            
            else:
                raise OSError( "Unable to open file: {0}".format(mask) )
            
            if ".gdb" in input_features:
                srcds = ogr.Open(os.path.dirname(input_features))
                
                if srcds:
                    srclyr = srcds.GetLayerByName(os.path.basename(input_features))
                    
                    if srclyr:
                        pass
                    
                    else:
                        raise OSError( "Unable to open layer: {0}".format(os.path.basename(input_features)) )            
                else:
                    raise OSError( "Unable to open GDB: {0}".format(os.path.dirname(input_features)) )            
            else:
                srcds = ogr.Open(input_features)
                
                if srcds:
                    srclyr = srcds.GetLayer()
                
                else:
                    raise OSError( "Unable to open file: {0}".format(input_features) )            
            
            msr = mlyr.GetSpatialRef()
            srcsr = srclyr.GetSpatialRef()            

            drv = ogr.GetDriverByName("ESRI Shapefile")
            trgds = drv.CreateDataSource(output_features)        
            
            trglyr = trgds.CreateLayer("subset", msr, ogr.wkbPolygon)
            
            srclyrdef = srclyr.GetLayerDefn()
            
            for i in range(srclyrdef.GetFieldCount()):
                trglyr.CreateField(srclyrdef.GetFieldDefn(i))
                
            newfield = ogr.FieldDefn(self.field, ogr.OFTInteger)
            trglyr.CreateField(newfield)

            trglyrdef = trglyr.GetLayerDefn()
            
            poly = ogr.Geometry(ogr.wkbPolygon)
        
            for feat in mlyr:
                poly  = poly.Union(feat.GetGeometryRef())
            feat = None
                    
            if msr.IsSame(srcsr):
                
                srclyr.SetSpatialFilter(poly)
                
                count = 0

                for feat in srclyr:
                    ofeat = ogr.Feature(trglyrdef)       
                    for i in range(0, srclyrdef.GetFieldCount()):
                        ofeat.SetField(trglyrdef.GetFieldDefn(i).GetName(), 
                                       feat.GetField(i))                
                    ofeat.SetField(self.field, count)
                    ofeat.SetGeometry(feat.GetGeometryRef().Clone())            
                    trglyr.CreateFeature(ofeat)
                    ofeat.Destroy()

                    count += 1
                    
            else:
                trg_to_src = osr.CoordinateTransformation(msr, srcsr)
                src_to_trg = osr.CoordinateTransformation(srcsr, msr)
                
                poly.Transform(trg_to_src)
            
                srclyr.SetSpatialFilter(poly)
                
                count = 0

                for feat in srclyr:
                    ofeat = ogr.Feature(trglyrdef)       
                    for i in range(0, srclyrdef.GetFieldCount()):
                        ofeat.SetField(trglyrdef.GetFieldDefn(i).GetNameRef(), 
                                       feat.GetField(i))
                    ofeat.SetField(self.field, count)
                    geom = feat.GetGeometryRef()
                    geom.Transform(src_to_trg)
                    ofeat.SetGeometry(geom)
                    trglyr.CreateFeature(ofeat)
                    ofeat.Destroy()  
                    
                    count += 1

            trgds.Destroy()
            
            feat = None
            mds = None
            srcds = None
            trgds = None
                
        else:
            raise OSError( "{0} already exists!". format(output_features) )
    
    
    def _poly_to_lines(self, input_features, output_features):
        """ Private method to convert polygons to  polylines """
        
        if not os.path.exists(output_features):
                
            srcds = ogr.Open(input_features)
            
            if srcds:
                srclyr = srcds.GetLayer()
            
            else:
                raise OSError( "Unable to open file: {0}".format(input_features) )         
            
            srclyr = srcds.GetLayer() 
            
            drv = ogr.GetDriverByName("ESRI Shapefile")
            trgds = drv.CreateDataSource(output_features)        
            
            trglyr = trgds.CreateLayer("mbg", srclyr.GetSpatialRef(), ogr.wkbLineString)
            
            srclyrdef = srclyr.GetLayerDefn()
            
            for i in range(srclyrdef.GetFieldCount()):
                trglyr.CreateField(srclyrdef.GetFieldDefn(i))
                
            trglyrdef = trglyr.GetLayerDefn()
            
            for feat in srclyr:
                ofeat = ogr.Feature(trglyrdef)       
                geom = feat.GetGeometryRef().GetLinearGeometry()
                for i in range(0, trglyrdef.GetFieldCount()):
                    ofeat.SetField(trglyrdef.GetFieldDefn(i).GetNameRef(), 
                                   feat.GetField(i))                
                ofeat.SetGeometry(geom)            
                trglyr.CreateFeature(ofeat)
                ofeat.Destroy()            
                
            trgds.Destroy()
            
            feat = None
            srcds = None
            trgds = None
    
        else:
            raise OSError( "{0} already exists!". format(output_features) )
    
    
    def _buffer_bbox(self, input_features, buffer_size, output_features):
        """ Private method to create minimum bounding box for features """
        
        if not os.path.exists(output_features):
            
            srcds = ogr.Open(input_features)
            
            if srcds:
                srclyr = srcds.GetLayer()
            
            else:
                raise OSError( "Unable to open file: {0}".format(input_features) )
            
            drv = ogr.GetDriverByName("ESRI Shapefile")
            trgds = drv.CreateDataSource(output_features)        
            
            trglyr = trgds.CreateLayer("mbg", srclyr.GetSpatialRef(), ogr.wkbPolygon)
            
            srclyrdef = srclyr.GetLayerDefn()
            
            for i in range(srclyrdef.GetFieldCount()):
                trglyr.CreateField(srclyrdef.GetFieldDefn(i))         
            
            trglyrdef = trglyr.GetLayerDefn()
                        
            for feat in srclyr:
                ofeat = ogr.Feature(trglyrdef)       
                for i in range(0, trglyrdef.GetFieldCount()):
                    ofeat.SetField(trglyrdef.GetFieldDefn(i).GetNameRef(), 
                                   feat.GetField(i))                 
                geom = feat.GetGeometryRef()
                env = geom.GetEnvelope()
                geom = self._extent_as_geom(*env)
                geom = geom.Buffer(buffer_size)
                env = geom.GetEnvelope()
                geom = self._extent_as_geom(*env)
                ofeat.SetGeometry(geom)            
                trglyr.CreateFeature(ofeat)
                ofeat.Destroy()            
                
            trgds.Destroy()
            
            feat = None
            srcds = None
                
        else:
            raise OSError( "{0} already exists!". format(output_features) )


    def prep_data(self):
        """ Private method to prepare data prior to automated shifting """

        ds = gdal.Open(self.images[0])
        self.bands = ds.RasterCount
        sr = osr.SpatialReference()
        sr.ImportFromWkt(ds.GetProjectionRef())
        
        if not sr.IsProjected():
            ds = None
            raise ValueError("Input image/s are not projected!")

        else:
            ds = None

        if len(self.images) == 1:
            self.vrt = self.images[0]
        
        else:
            self.vrt = os.path.join(self.workspace, "temp.vrt")
            
            if not os.path.exists(self.vrt):
                gdal.BuildVRT(self.vrt, self.images)
        
        self.index = os.path.join(self.workspace, "_index.shp")
        
        if not os.path.exists(self.index):
            self._create_index(self.images, self.index)
        
        self.subset = os.path.join(self.workspace, "_subset.shp")
        
        if not os.path.exists(self.subset):
            self._filter_transform(self.in_feat, self.index, self.subset)
            
        self.lines = os.path.join(self.workspace, "_lines.shp")
                            
        if not os.path.exists(self.lines):
            self._poly_to_lines(self.subset, self.lines)
    
    
    def sample(self, field, value, output_directory):
        """ Public method to create and save sample output for a given building id """
        
        if not os.path.exists(self.subset):
            raise OSError( "Data prepartion must be run prior to sample creation" )
            
        else:
            srcds = ogr.Open(self.subset)
            
            if srcds:
                srclyr = srcds.GetLayer()
        
            else:
                raise OSError( "Unable to open file: {0}".format(self.subset) )
        
        iob = Image(self.vrt)
        
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
            
        srclyr.SetAttributeFilter("{0} = {1}".format(field, value))
        
        if srclyr.GetFeatureCount() > 0:

            for feat in srclyr:
                        
                geom = feat.GetGeometryRef()
                bldextent = geom.GetEnvelope()
                bldextent = self._extent_as_geom(bldextent[0], bldextent[1], bldextent[2], bldextent[3])
                bldextent = bldextent.Buffer(self.search_distance)
                bldextent = bldextent.GetEnvelope()

                bld = os.path.join(output_directory, "bld_{0}.tif".format(value))
                
                if not os.path.exists(bld):
                    adjx = bldextent[0] - ((bldextent[0] - iob.extent[0]) % iob.cellX)
                    adjy = bldextent[3] + ((iob.extent[3] - bldextent[3]) % abs(iob.cellX))
                    self._rasterize(self.lines, iob.cellX, bld, (adjx, bldextent[1], bldextent[2], adjy))
                
                geom = self._extent_as_geom(bldextent[0], bldextent[1], bldextent[2], bldextent[3])
                geom = geom.Buffer(self.search_distance)
                buffextent = geom.GetEnvelope()
                
                xoff, yoff = iob._ms_to_is((buffextent[0], buffextent[3]))
                winx = int(math.ceil((buffextent[1]-buffextent[0])/iob.cellX))
                winy = int(math.ceil((buffextent[2]-buffextent[3])/iob.cellY))    
            
                tob = Tile(self.vrt, xoff, yoff, winx, winy)
                
                origi = tob.read_tile()

                orig = os.path.join(output_directory, "orig_{0}.tif".format(value))
                
                if not os.path.exists(orig):
                    tob.save(origi, orig)
                    
                if self.bands > 1:
                    grey = os.path.join(output_directory, "grey_{0}.tif".format(value))

                    greyi = np.zeros((origi.shape[1], origi.shape[2]))
                        
                    vals = [.299, .587, .114]
                        
                    for i, v in enumerate(vals):
                        greyi += origi[i,:,:] * v

                    greyi = np.rint((greyi/2047) * 255).astype(np.uint8)

                else:
                    greyi = origi

                if not os.path.exists(grey):
                    tob.save(greyi, grey)
                    
                bilat = os.path.join(output_directory, "bilat_{0}.tif".format(value))
                
                if not os.path.exists(bilat):
                    bar = cv.bilateralFilter(greyi, 9, 75, 75)
                    tob.save(bar, bilat)
                 
                sharp = os.path.join(output_directory, "sharp_{0}.tif".format(value))

                if not os.path.exists(sharp):
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    greyi = cv.filter2D(greyi, -1, kernel)
                    tob.save(greyi, sharp)
                       
                grad = os.path.join(output_directory, "grad_{0}.tif".format(value))

                if not os.path.exists(grad):
                    x = cv.Sobel(greyi, cv.CV_64F, 1, 0)
                    y = cv.Sobel(greyi, cv.CV_64F, 0, 1)
        
                    mag, angle = cv.cartToPolar(x, y, angleInDegrees=True)    
                    tob.save(angle, grad)

                canny = os.path.join(output_directory, "canny_{0}.tif".format(value))
                
                if not os.path.exists(canny):
                    #m = np.median(bar)
                    m = np.median(greyi)
                    sigma = 0.25
                    lower = int(max(0, (1.0 - sigma) * m))
                    upper = int(min(255, (1.0 + sigma) * m))
                    #car = cv.Canny(bar, lower, upper)
                    car = cv.Canny(greyi, lower, upper)
                    tob.save(car, canny)
                                     
                feat = None
                
            srcds = None
            
        else:
            srcds = None
            raise ValueError( "No feature with {0} for field {1}".format(value, field) )
            
            
    def _calc_shift(self):
        """ Private method to calculate displacement of features based on image gradients """
                            
        data = []
        
        # Open vector features
        srcds = ogr.Open(self.subset)
        srclyr = srcds.GetLayer()
        
        # Create image object
        iob = Image(self.vrt)
        for feat in srclyr:
                    
            origg = feat.GetGeometryRef().Clone()
            
            # Get envelope of feature
            fenv = origg.GetEnvelope()
            fenv = self._extent_as_geom(fenv[0], fenv[1], fenv[2], fenv[3])
            fenv = fenv.Buffer(50)
            fenv = fenv.GetEnvelope()
            
            # Calculate buffered envelope of feature
            buffg = self._extent_as_geom(fenv[0], fenv[1], 
                                         fenv[2], fenv[3])
            buffg = buffg.Buffer(self.search_distance)
            buffenv = buffg.GetEnvelope()
            
            # Transform feature coords to image coords
            try:
                xoff, yoff = iob._ms_to_is((buffenv[0], 
                                            buffenv[3]))
                
            except ValueError:
                xoff = None
                yoff = None
            
            
            if xoff and yoff:
                
                # Convert polylines to raster
                adjx = fenv[0] - ((fenv[0] - iob.extent[0]) % iob.cellX)
                adjy = fenv[3] + ((iob.extent[3] - fenv[3]) % abs(iob.cellX))
                temp = self._rasterize(self.lines, iob.cellX, "MEM", 
                                       (adjx, fenv[1], fenv[2], adjy))
                
                # Calculate window size 
                winx = int(math.ceil((buffenv[1]-buffenv[0])/iob.cellX))
                winy = int(math.ceil((buffenv[2]-buffenv[3])/iob.cellY))    
        
                # Create tile object
                try:
                    tob = Tile(self.vrt, xoff, yoff, winx, winy)
                    
                except ValueError:
                    tob = None
                    
                if tob:
        
                    # Original image
                    origi = tob.read_tile()
                    
                    if self.bands > 1:
                    
                        # Convert to original image to greyscale
                        greyi = np.zeros((origi.shape[1], origi.shape[2]))
                        
                        vals = [.299, .587, .114]
                        
                        for i, v in enumerate(vals):
                            greyi += origi[i,:,:] * v
                            
                        greyi = np.rint(greyi, dtype=np.float32)

                    else:
                        greyi = origi.astype(np.float32)
                    
                    if np.median(greyi) >= 0:
                    
                        # Apply bilateral filter if less than 20cm
                        if iob.cellX <= 0.2:
                            greyi = cv.bilateralFilter(greyi, 9, 75, 75)
                    
                        # Sharpening filter (from testing)
                        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        #greyi = cv.filter2D(greyi, -1, kernel)
                        
                        # Calculate gradient magnitude
                        x = cv.Sobel(greyi, cv.CV_64F, 1, 0)
                        y = cv.Sobel(greyi, cv.CV_64F, 0, 1)
        
                        mag, angle = cv.cartToPolar(x, y, angleInDegrees=True)

                        # Apply Canny edge detection (from testing)
                        #m = np.median(greyi)
                        #m = np.median(bilati)
                        #sigma = 0.25
                        #lower = int(max(0, (1.0 - sigma) * m))
                        #upper = int(min(255, (1.0 + sigma) * m))
                        #canni = cv.Canny(bilati, lower, upper)
                        #canni = cv.Canny(greyi, lower, upper)
                        
                        # Apply template matching
                        res = cv.matchTemplate(mag.astype(np.float32), temp.astype(np.float32), cv.TM_CCORR_NORMED)
                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                        
                        # Calculate displacement
                        dispx = (tob.minX + (max_loc[0] * iob.cellX)) - fenv[0]
                        dispy = (tob.maxY + (max_loc[1] * iob.cellY)) - fenv[3]
                       
                        # Add data to list
                        data.append([feat.GetField(self.field),
                                     origg.Centroid().GetX(),
                                     origg.Centroid().GetY(),
                                     max_val,
                                     dispx,
                                     dispy])
                        
                        # Free memory
                        feat = None             
                        res = None
                        canni = None
                        origi = None
                        greyi = None
                        bilati = None
                        temp = None
        
        srcds = None
        
        # Create dataframe
        self.df = pd.DataFrame(data, columns=["ID", 
                                              "X", 
                                              "Y",
                                              "Score",
                                              "DispX", 
                                              "DispY"])
             
        self.df["NNX"] = [-9999.0] * len(self.df.index)
        self.df["NNY"] = [-9999.0] * len(self.df.index)
        
        
    def _validate_shift(self):
        """ Private method to compare estimated shift against neighboring estimates """
        
        if self.global_val:

            # Calculate mean, std, and upper/lower for x/y displacement
            mx, stdx, lox, uox = self._outliers(self.df['DispX'])
            my, stdy, loy, uoy = self._outliers(self.df['DispY'])

            for row in self.df.index:
            
                # Get estimated displacement
                cx = self.df.at[row, 'DispX']
                cy = self.df.at[row, 'DispY']

                # Compare x displacement against neighbors
                if cx < lox or cx > uox:
                    self.df.at[row, "NNX"] = mx
            
                # Compare y displacement against neighbors
                if cy < loy or cy > uoy:
                    self.df.at[row, "NNY"] = my
        
        if self.local_val:

            # Calculate distance matrix
            dmat = spatial.distance.pdist(self.df.values[:, 1:3])
            mat = spatial.distance.squareform(dmat)
        
            # Calculate mean, std, and upper/lower for x/y displacement
            mx, stdx, lox, uox = self._outliers(self.df['DispX'])
            my, stdy, loy, uoy = self._outliers(self.df['DispY'])

            for row in self.df.index:
            
                # Get estimated displacement
                cx = self.df.at[row, 'DispX']
                cy = self.df.at[row, 'DispY']
              
                # Find n nearest neighbors
                z = sorted(zip(self.df['DispX'], 
                               self.df['DispY'], 
                               mat[row]), 
                           key=lambda x : x[2])
            
                # Get displacement vectors of nieghbors
                nx = np.array(sorted([b[0] for b in z[1:self.local_val+1]]))
                ny = np.array(sorted([b[1] for b in z[1:self.local_val+1]]))
        
                # Calculate mean, std, and upper/lower for x/y displacement
                mx, stdx, lox, uox = self._outliers(nx)
                my, stdy, loy, uoy = self._outliers(ny)
            
                # Compare x displacement against neighbors
                if cx < lox or cx > uox:
                    self.df.at[row, "NNX"] = mx
            
                # Compare y displacement against neighbors
                if cy < loy or cy > uoy:
                    self.df.at[row, "NNY"] = my
        
    
    def _apply_shift(self):
        """ Private method to shift features based on image gradients """
        
        if not os.path.exists(self.out_feat):
                                    
            srcds = ogr.Open(self.subset)
            srclyr = srcds.GetLayer()
            
            drv = ogr.GetDriverByName("ESRI Shapefile")
            trgds = drv.CreateDataSource(self.out_feat)        
            
            trglyr = trgds.CreateLayer("shift", srclyr.GetSpatialRef(), ogr.wkbPolygon)
            
            # Copy original fields
            srclyrdef = srclyr.GetLayerDefn()
            
            for i in range(srclyrdef.GetFieldCount()):
                trglyr.CreateField(srclyrdef.GetFieldDefn(i))
            
            # Add fields associated with shifting
            fn = ogr.FieldDefn("Score", ogr.OFTReal)
            trglyr.CreateField(fn)
            fn = ogr.FieldDefn("XDisp", ogr.OFTReal)
            trglyr.CreateField(fn)        
            fn = ogr.FieldDefn("YDisp", ogr.OFTReal)
            trglyr.CreateField(fn)            
            fn = ogr.FieldDefn("NNX", ogr.OFTReal)
            trglyr.CreateField(fn)   
            fn = ogr.FieldDefn("NNY", ogr.OFTReal)
            trglyr.CreateField(fn)               
            
            trglyrdef = trglyr.GetLayerDefn()
                        
            for feat in srclyr:
                        
                geom = feat.GetGeometryRef().Clone()
                fid = feat.GetField(self.field)
                
                # Filter dataframe on current feature id
                row = self.df.loc[self.df["ID"] == fid]
                
                # Check if feature is in dataframe
                if not row.empty:
                    
                    # Convert row to dict to prevent series
                    row = dict(zip(row.columns, row.values[0]))

                    # Create output feature
                    ofeat = ogr.Feature(trglyrdef)
                
                    # Copy over existing attributes
                    for i in range(srclyrdef.GetFieldCount()):
                        ofeat.SetField(srclyrdef.GetFieldDefn(i).GetNameRef(), 
                                       feat.GetField(i))
                    
                    # Update fields associated with shifting
                    ofeat.SetField("Score", row['Score'])
                    ofeat.SetField("XDisp", row['DispX'])
                    ofeat.SetField("YDisp", row['DispY'])
                    ofeat.SetField("NNX", row['NNX'])
                    ofeat.SetField("NNY", row['NNY'])
                    
                    # Check if neighborhood displacement is set 
                    if row['NNX'] == -9999:
                        dispx = row['DispX']
                    else:
                        dispx = row['NNX']
                        
                    if row['NNY'] == -9999:
                        dispy = row['DispY']
                    else:
                        dispy = row['NNY']                
                    
                    for ring in range(geom.GetGeometryCount()):
                        pts = geom.GetGeometryRef(ring)
                    
                        # Apply shift to vertices
                        for pt in range(pts.GetPointCount()):
                            trans = [a + b for a, b in zip((dispx,dispy), pts.GetPoint(pt))]
                            pts.SetPoint(pt, trans[0], trans[1])
                    
                    ofeat.SetGeometry(geom)
                    trglyr.CreateFeature(ofeat)
                    
                    ofeat.Destroy()            
 
            trgds = None
            srcds = None
        
        else:
            raise OSError( "{0} already exists!". format(self.out_feat) )
        
            
    def shift(self):
        """ Main method to shift features """
        
        if not os.path.exists(self.out_feat):
            
            s = time.clock()
            
            self.prep_data()
            
            self._calc_shift()
            
            self._validate_shift()
            
            self._apply_shift()
            
            shutil.rmtree(self.workspace)
            
            e = time.clock()
            
            print( "Shifting completed in {0} seconds.".format(e-s) )
        
        else:
            raise OSError( "{0} already exists!". format(self.out_feat) )         
                
        
if __name__ == "__main__":
    
    args = parse_args()
    s = Shift(args.inputFeat, args.targets, args.outputFeat, args.distance, args.localVal, args.globalVal)    
    s.prep_data()
    s.shift()
