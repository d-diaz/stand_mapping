import os
import numpy as np
import rasterio
from rasterio import windows
import torch
from torch.utils.data import Dataset
from scipy.stats import randint


class SemanticDataset(Dataset):
    def __init__(self, root, dataframe, raw_chip_size,
                 transform=None, target_transform=None,
                 use_layers=None, random_state=None,
                 boundary_class=False):
        """Initialize a SemanticDataset for semantic segmentation.

        Parameters
        ----------
        root : str
          path to root of data
        dataframe : Pandas DataFrame
          dataframe containing attributes of samples to load
        raw_chip_size : int
          height and width of area to read from each input and target layer
        transform, target_transform : callable, optional
          a function/transform that takes in the input or target and returns
          a transformed version.
        use_layers : dict, optional
          key, value pairs where each key is a type of input layer and value
          is whether or not to include this layer in the stack of input layers;
          each layer requested must have a {layer}_PATH column in the
          dataframe. If use_layers is not provided, by default only NAIP 4-band
          imagery will be loaded.
        boundary_class : bool, optional
          if True, an additional semantic class will be added to the semantic
          target which indicates whether or not a pixel is a boundary between
          land cover instances.
        random_state : int, optional
          ...
        """
        super().__init__()
        self.root = root
        self.df = dataframe.copy()
        self.raw_chip_size = raw_chip_size
        self.transform = transform
        self.target_transform = target_transform
        self.path_cols = [col for col in dataframe.columns if '_PATH' in col]
        self.layer_types = [col.split('_PATH')[0].lower() for col in
                            self.path_cols]
        self.boundary_class = boundary_class
        self.random_state = random_state

        if use_layers is None:
            self.use_layers = {layer_type: {'use': False, 'col': path_col} for
                               layer_type, path_col in
                               zip(self.layer_types, self.path_cols)}
            self.use_layers['naip']['use'] = True
        else:
            self.use_layers = {key: {'use': value} for
                               key, value in use_layers.items()}
            for layer_type in self.use_layers:
                if (layer_type.upper() + '_PATH') not in self.path_cols:
                    raise ValueError(f'Unrecognized layer type: {layer_type}')
                else:
                    self.use_layers[layer_type]['col'] = layer_type.upper() + \
                        '_PATH'

    def __getitem__(self, index):
        """Fetch a sample from the dataset.

        Parameters
        ----------
        index : int
          index of the sample in the dataframe to retrieve data for

        Returns
        -------
        input : FloatTensor
          input image as a FloatTensor
        sem : LongTensor
          semantic segmentation target
        nolabel : BoolTensor
          indicates areas that do not have a land cover label assigned (either
          because the areas was unmapped or because the annotator did not
          assign a land cover type).
        """
        window = None
        inputs = []
        for layer_type in self.use_layers:
            if self.use_layers[layer_type]['use']:
                col = self.use_layers[layer_type]['col']
                path = os.path.join(self.root, self.df.iloc[index][col])
                with rasterio.open(path) as src:
                    if window is None:
                        height, width = src.shape
                        col_off = randint.rvs(0, width - self.raw_chip_size,
                                              random_state=self.random_state)
                        row_off = randint.rvs(0, height - self.raw_chip_size,
                                              random_state=self.random_state)
                        window = windows.Window(col_off, row_off,
                                                self.raw_chip_size,
                                                self.raw_chip_size)
                    img = src.read(window=window)
                    inputs.append(img)

        input = np.vstack(inputs)
        input = torch.FloatTensor(input)

        sem_path = os.path.join(self.root,
                                self.df.iloc[index]['SEMANTIC_PATH'])
        with rasterio.open(sem_path) as src:
            sem = src.read(window=window)

        sem = torch.LongTensor(sem)
        if self.boundary_class:
            bnd_path = os.path.join(self.root,
                                    self.df.iloc[index]['BOUNDARY_PATH'])
            with rasterio.open(bnd_path) as src:
                bnd = src.read(1, window=window)
            sem[:, bnd == 1] = 6
        # 0 means no cover type assigned, 255 means area wasn't mapped
        nolabel = torch.BoolTensor(sem == 0) + torch.BoolTensor(sem == 255)
        sem[nolabel] = 0  # set all nodata values to 0

        target = sem - 1  # shift semantic labels for 0-indexing (water = 0)

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            nolabel = self.target_transform(nolabel)

        return input, target, nolabel

    def __len__(self):
        return len(self.df)


class SemanticAndWatershedDataset(SemanticDataset):
    def __init__(self, root, dataframe, raw_chip_size,
                 transform=None, target_transform=None,
                 use_layers=None, random_state=None,
                 boundary_class=False, clip_watershed=-100):
        """Initialize a Dataset for semantic segmentation and watershed energy
        modeling. Semantic layer includes land cover types plus an optional
        cover type for boundaries between land cover objects/instances. The
        watershed energy layer indicates the distance of a pixel from the
        nearest boundary.

        Parameters
        ----------
        root : str
          path to root of data
        dataframe : Pandas DataFrame
          dataframe containing attributes of samples to load
        raw_chip_size : int
          height and width of area to read from each input and target layer
        transform, target_transform : callable, optional
          a function/transform that takes in the input or target and returns
          a transformed version.
        use_layers : dict, optional
          key, value pairs where each key is a type of input layer and value
          is whether or not to include this layer in the stack of input layers;
          each layer requested must have a {layer}_PATH column in the
          dataframe. If use_layers is not provided, by default only NAIP 4-band
          imagery will be loaded.
        boundary_class : bool, optional
          if True, an additional semantic class will be added to the semantic
          target which indicates whether or not a pixel is a boundary between
          land cover instances.
        clip_watershed : numeric, optional
          value to clip watershed energy target to. Watershed energy indicates
          -1 times the distance to the nearest boundary. The default value of
          -100 means that all pixels further than 100 meters will be treated as
          if they were only 100 meters away.
        """
        super().__init__(
            root, dataframe, raw_chip_size,
            transform=transform, target_transform=target_transform,
            use_layers=use_layers, random_state=random_state,
            boundary_class=boundary_class)

        self.clip_watershed = clip_watershed

    def __getitem__(self, index):
        """Fetch a sample from the dataset.

        Parameters
        ----------
        index : int
          index of the sample in the dataframe to retrieve data for

        Returns
        -------
        input : FloatTensor
          input image as a FloatTensor
        (sem, watershed) : 2-tuple of LongTensors
          semantic segmentation target and watershed energy target
        (nolabel, nodata) : 2-tuple of BoolTensors
          nolabel indicates areas that have been delineated but did not have a
          cover type assigned; nodata indicates areas that have not been mapped
          at all.
        """
        window = None
        inputs = []
        for layer_type in self.use_layers:
            if self.use_layers[layer_type]['use']:
                col = self.use_layers[layer_type]['col']
                path = os.path.join(self.root, self.df.iloc[index][col])
                with rasterio.open(path) as src:
                    if window is None:
                        height, width = src.shape
                        col_off = randint.rvs(0, width - self.raw_chip_size,
                                              random_state=self.random_state)
                        row_off = randint.rvs(0, height - self.raw_chip_size,
                                              random_state=self.random_state)
                        window = windows.Window(col_off, row_off,
                                                self.raw_chip_size,
                                                self.raw_chip_size)
                    img = src.read(window=window)
                    inputs.append(img)

        input = np.vstack(inputs)
        input = torch.FloatTensor(input)

        sem_path = os.path.join(self.root,
                                self.df.iloc[index]['SEMANTIC_PATH'])
        with rasterio.open(sem_path) as src:
            sem = src.read(window=window)

        sem = torch.LongTensor(sem)
        if self.boundary_class:
            bnd_path = os.path.join(self.root,
                                    self.df.iloc[index]['BOUNDARY_PATH'])
            with rasterio.open(bnd_path) as src:
                bnd = src.read(1, window=window)
            sem[:, bnd == 1] = 6
        # 0 means no cover type assigned, 255 means area wasn't mapped
        nolabel = torch.BoolTensor(sem == 0) + torch.BoolTensor(sem == 255)
        nodata = torch.BoolTensor(sem == 255)
        sem[nolabel] = 0  # set all nodata values to 0

        sem = sem - 1  # shift semantic labels for 0-indexing (water = 0)

        watershed_path = os.path.join(self.root,
                                      self.df.iloc[index]['WATERSHED_PATH'])
        with rasterio.open(watershed_path) as src:
            watershed = np.expand_dims(src.read(1, window=window), 0)
            if self.clip_watershed is not None:
                watershed = watershed.clip(self.clip_watershed, 0)
        watershed = torch.LongTensor(watershed)

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            sem = self.target_transform(sem)
            watershed = self.target_transform(watershed)
            nolabel = self.target_transform(nolabel)
            nodata = self.target_transform(nodata)

        return input, (sem, watershed), (nolabel, nodata)

    def __len__(self):
        return len(self.df)

class WatershedDataset(SemanticDataset):
    def __init__(self, root, dataframe, raw_chip_size,
                 transform=None, target_transform=None,
                 use_layers=None, random_state=None,
                 clip_watershed=-100):
        """Initialize a Dataset for watershed energy modeling.

        The watershed energy layer indicates the distance of a pixel from the
        nearest boundary.

        Parameters
        ----------
        root : str
          path to root of data
        dataframe : Pandas DataFrame
          dataframe containing attributes of samples to load
        raw_chip_size : int
          height and width of area to read from each input and target layer
        transform, target_transform : callable, optional
          a function/transform that takes in the input or target and returns
          a transformed version.
        use_layers : dict, optional
          key, value pairs where each key is a type of input layer and value
          is whether or not to include this layer in the stack of input layers;
          each layer requested must have a {layer}_PATH column in the
          dataframe. If use_layers is not provided, by default only NAIP 4-band
          imagery will be loaded.
        clip_watershed : numeric, optional
          value to clip watershed energy target to. Watershed energy indicates
          -1 times the distance to the nearest boundary. The default value of
          -100 means that all pixels further than 100 meters will be treated as
          if they were only 100 meters away.
        """
        super().__init__(
            root, dataframe, raw_chip_size,
            transform=transform, target_transform=target_transform,
            use_layers=use_layers, random_state=random_state)

        self.clip_watershed = clip_watershed

    def __getitem__(self, index):
        """Fetch a sample from the dataset.

        Parameters
        ----------
        index : int
          index of the sample in the dataframe to retrieve data for

        Returns
        -------
        input : FloatTensor
          input image as a FloatTensor
        watershed : LongTensor
          watershed energy target
        nodata : BoolTensor
          nodata indicates areas that have not been mapped at all
        """
        window = None
        inputs = []
        for layer_type in self.use_layers:
            if self.use_layers[layer_type]['use']:
                col = self.use_layers[layer_type]['col']
                path = os.path.join(self.root, self.df.iloc[index][col])
                with rasterio.open(path) as src:
                    if window is None:
                        height, width = src.shape
                        col_off = randint.rvs(0, width - self.raw_chip_size,
                                              random_state=self.random_state)
                        row_off = randint.rvs(0, height - self.raw_chip_size,
                                              random_state=self.random_state)
                        window = windows.Window(col_off, row_off,
                                                self.raw_chip_size,
                                                self.raw_chip_size)
                    img = src.read(window=window)
                    inputs.append(img)

        input = np.vstack(inputs)
        input = torch.FloatTensor(input)

        sem_path = os.path.join(self.root,
                                self.df.iloc[index]['SEMANTIC_PATH'])
        with rasterio.open(sem_path) as src:
            sem = src.read(window=window)

        nodata = torch.BoolTensor(sem == 255)

        watershed_path = os.path.join(self.root,
                                      self.df.iloc[index]['WATERSHED_PATH'])
        with rasterio.open(watershed_path) as src:
            watershed = np.expand_dims(src.read(1, window=window), 0)
            if self.clip_watershed is not None:
                watershed = watershed.clip(self.clip_watershed, 0)
        watershed = torch.LongTensor(watershed)

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            watershed = self.target_transform(watershed)
            nodata = self.target_transform(nodata)

        return input, watershed, nodata

    def __len__(self):
        return len(self.df)

class SemanticAndInstanceDataset(SemanticDataset):
    def __init__(self, root, dataframe, raw_chip_size,
                 transform=None, target_transform=None,
                 use_layers=None, random_state=None,
                 boundary_class=False, thing_classes=None):
        """Initialize a Dataset for semantic segmentation and watershed energy
        modeling. Semantic layer includes land cover types plus an optional
        cover type for boundaries between land cover objects/instances. The
        instance layer contains a channel for each `thing_class` requested
        with each instance indicated by a unique integer.

        Parameters
        ----------
        root : str
          path to root of data
        dataframe : Pandas DataFrame
          dataframe containing attributes of samples to load
        raw_chip_size : int
          height and width of area to read from each input and target layer
        transform, target_transform : callable, optional
          a function/transform that takes in the input or target and returns
          a transformed version.
        use_layers : dict, optional
          key, value pairs where each key is a type of input layer and value
          is whether or not to include this layer in the stack of input layers;
          each layer requested must have a {layer}_PATH column in the
          dataframe. If use_layers is not provided, by default only NAIP 4-band
          imagery will be loaded.
        boundary_class : bool, optional
          if True, an additional semantic class will be added to the semantic
          target which indicates whether or not a pixel is a boundary between
          land cover instances.
        thing_classes : dict, optional
          a dictionary with key, value pairs where keys are land cover types
          and value is a boolean indicating whether or not instances of that
          land cover should be returned as targets. For example, an argument of
          `{'forest': True}` would only return forest instances. All cover
          types not mentioned (water, forest, field, barren, developed) will
          be omitted from the instance target layer. By default, only forest
          instances are treated as things.
        """
        super().__init__(
            root, dataframe, raw_chip_size,
            transform=transform, target_transform=target_transform,
            use_layers=use_layers, random_state=random_state,
            boundary_class=boundary_class)

        self.thing_classes = {
            'water': False,
            'forest': True,
            'field': False,
            'barren': False,
            'developed': False,
        }

        if thing_classes is not None:
            self.thing_classes.update(thing_classes)

    def __getitem__(self, index):
        """Fetch a sample from the dataset.

        Parameters
        ----------
        index : int
          index of the sample in the dataframe to retrieve data for

        Returns
        -------
        input : FloatTensor
          input image as a FloatTensor
        (sem, inst) : 2-tuple of LongTensors
          semantic segmentation target and instance target
        (nolabel, nodata) : 2-tuple of BoolTensors
          nolabel indicates areas that have been delineated but did not have a
          cover type assigned; nodata indicates areas that have not been mapped
          at all.
        """
        window = None
        inputs = []
        for layer_type in self.use_layers:
            if self.use_layers[layer_type]['use']:
                col = self.use_layers[layer_type]['col']
                path = os.path.join(self.root, self.df.iloc[index][col])
                with rasterio.open(path) as src:
                    if window is None:
                        height, width = src.shape
                        col_off = randint.rvs(0, width - self.raw_chip_size,
                                              random_state=self.random_state)
                        row_off = randint.rvs(0, height - self.raw_chip_size,
                                              random_state=self.random_state)
                        window = windows.Window(col_off, row_off,
                                                self.raw_chip_size,
                                                self.raw_chip_size)
                    img = src.read(window=window)
                    inputs.append(img)

        input = np.vstack(inputs)
        input = torch.FloatTensor(input)

        sem_path = os.path.join(self.root,
                                self.df.iloc[index]['SEMANTIC_PATH'])
        with rasterio.open(sem_path) as src:
            sem = src.read(window=window)

        sem = torch.LongTensor(sem)
        if self.boundary_class:
            bnd_path = os.path.join(self.root,
                                    self.df.iloc[index]['BOUNDARY_PATH'])
            with rasterio.open(bnd_path) as src:
                bnd = src.read(1, window=window)
            sem[:, bnd == 1] = 6
        # 0 means no cover type assigned, 255 means area wasn't mapped
        nolabel = torch.BoolTensor(sem == 0) + torch.BoolTensor(sem == 255)
        sem[nolabel] = 0  # set all nodata values to 0

        sem = sem - 1  # shift semantic labels for 0-indexing (water = 0)

        instance_path = os.path.join(self.root,
                                     self.df.iloc[index]['INSTANCE_PATH'])
        with rasterio.open(instance_path) as src:
            things = []
            COVER_TYPES = ['water', 'forest', 'field', 'barren', 'developed']
            for i, cover_type in enumerate(COVER_TYPES):
                if self.thing_classes[cover_type]:
                    thing = src.read(i+2, window=window).astype(np.int16)
                    things.append(thing)
            inst = np.stack(things)
        inst = torch.LongTensor(inst)

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            sem = self.target_transform(sem)
            inst = self.target_transform(inst)
            nolabel = self.target_transform(nolabel)

        return input, (sem, inst), (nolabel, nolabel)

    def __len__(self):
        return len(self.df)
