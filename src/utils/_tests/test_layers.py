import unittest
from ..layers import DataLayer, TopographyLayer


class TestDataLayer(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Set up tests for reading in real topography
        '''

    def test__get_nearest_tile(self) -> None:
        '''
        Test that the call to _get_nearest_tile() runs properly.
        This method will calculate the closest 5 degrees (N, W) that
            contain the latitude and longitude specified
        '''
        resolution = 30

        # 2 Tiles
        center = (33.4, 116.004)
        height, width = 1600, 1600
        data_layer = DataLayer(center, height, width, resolution)
        test_output_dems = ((33, 117), (33, 115))
        self.assertEqual(data_layer.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(data_layer.five_deg_north_max, test_output_dems[1][0])
        self.assertEqual(data_layer.five_deg_west_min, test_output_dems[1][1])
        self.assertEqual(data_layer.five_deg_west_max, test_output_dems[0][1])

        # 4 Tiles
        center = (35.001, 117.001)
        height, width = 1600, 1600
        data_layer = DataLayer(center, height, width, resolution)
        test_output_dems = ((34, 118), (34, 117), (35, 117), (35, 118))
        self.assertEqual(data_layer.five_deg_north_min, test_output_dems[0][0])
        self.assertEqual(data_layer.five_deg_north_max, test_output_dems[2][0])
        self.assertEqual(data_layer.five_deg_west_min, test_output_dems[1][1])
        self.assertEqual(data_layer.five_deg_west_max, test_output_dems[0][1])

    def test__stack_tiles(self) -> None:
        '''
        Test that the call to _stack_tiles() runs properly.
        This method correctly stitches together tiles on (easternly, southernly, square)
        '''

        # 2 Tiles
        resolution = 30
        center = (35.001, 115.6)
        height, width = 1600, 1600
        data_layer = DataLayer(center, height, width, resolution)
        test_output_dems = {'north': ((34, 116), (35, 116))}
        self.assertEqual(test_output_dems, data_layer.tiles)

        # 4 Tiles
        resolution = 90
        center = (34.99, 115.001)
        height, width = 3200, 3200
        data_layer = DataLayer(center, height, width, resolution)
        test_output_dems = {'square': ((30, 120), (30, 115), (40, 115), (40, 120))}
        self.assertEqual(test_output_dems, data_layer.tiles)

    def test__generate_lat_long(self) -> None:
        '''
        Test that the call to _genrate_lat_long() runs properly.
        This method first creates an array of the latitude and longitude coords of all
            DEMs contained withing the specified lat/long region when MERITLayer
            is initialized. It will correctly calculate the corners of the array
            and fill the array with lat/long pairs using 5degs/6000 pixels
            or ~3 arc-seconds at every point.
        It will then find the correct bounds of the specified lat/long region to
            pull elevation data.
        '''
        resolution = 30

        # # 2 Tiles easternly
        center = (36.4, 118.01)
        height, width = 3200, 3200
        data_layer = DataLayer(center, height, width, resolution)
        self.assertEqual(data_layer.elev_array.shape, (7224, 3612, 2))

        # 2 Tiles northernly
        center = (34.001, 115.6)
        height, width = 3200, 3200
        data_layer = DataLayer(center, height, width, resolution)
        self.assertEqual(data_layer.elev_array.shape, (3612, 7224, 2))

    def test__get_lat_long_bbox(self) -> None:
        '''
        Test that the call to _get_lat_long_bbox() runs properly.
        This method will update the corners of the array of DEM tiles loaded

        '''
        resolution = 30

        # 2 Tiles northernly
        center = (34.001, 115.6)
        height, width = 3200, 3200
        data_layer = DataLayer(center, height, width, resolution)
        output = [(33, 116), (33, 115), (35, 115), (35, 116)]
        self.assertEqual(output, data_layer.corners)

    def test_save_contour_map(self) -> None:
        '''
        Test that the call to _save_contour_map() runs propoerly.
        '''
        resolution = 30
        # Single Tile
        center = (33.5, 116.8)
        height, width = 1600, 1600
        data_layer = DataLayer(center, height, width, resolution)
        topo_layer = TopographyLayer(data_layer)
        data_layer._save_contour_map(topo_layer.data)


class TestTopographyLayer(unittest.TestCase):
    def setUp(self) -> None:
        '''

        '''

    def test__make_contour_and_data(self) -> None:
        '''
        Test that the call to _generate_contours() runs propoerly.
        This method returns the data array containing the elevations within the
            specified bounding box region of the given latitudes and longitudes.

        NOTE: This method should always return a square
        '''
        resolution = 30
        # 2 Tiles (easternly)
        center = (33.4, 115.04)
        height, width = 3200, 3200
        data_layer = DataLayer(center, height, width, resolution)
        topographyGen = TopographyLayer(data_layer)
        self.assertEqual(topographyGen.data.shape[0], topographyGen.data.shape[1])

    def test__get_dems(self) -> None:
        '''
        Test that the call to _get_dems() runs properly.
        This method will generate a list of the DEMs in the fireline /nfs/
        '''

        resolution = 30
        # Single Tile
        center = (35.2, 115.6)
        height, width = 1600, 1600
        data_layer = DataLayer(center, height, width, resolution)
        topographyGen = TopographyLayer(data_layer)
        self.assertEqual(1, len(topographyGen.tif_filenames))

        # 2 Tiles
        center = (38.4, 115.0)
        height, width = 1600, 1600
        data_layer = DataLayer(center, height, width, resolution)
        topographyGen = TopographyLayer(data_layer)
        self.assertEqual(2, len(topographyGen.tif_filenames))

        # 4 Tiles
        center = (34.001, 116.008)
        height, width = 3200, 3200
        data_layer = DataLayer(center, height, width, resolution)
        topographyGen = TopographyLayer(data_layer)
        self.assertEqual(4, len(topographyGen.tif_filenames))
