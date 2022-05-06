import dataclasses
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml  # type: ignore
from yaml.parser import ParserError  # type: ignore

from ..world.elevation_functions import flat, gaussian, perlin
from ..world.fuel_array_functions import chaparral_fn
from ..world.wind_mechanics.wind_controller import WindController
from .layers import (
    FuelLayer,
    FunctionalFuelLayer,
    FunctionalTopographyLayer,
    LatLongBox,
    OperationalFuelLayer,
    OperationalTopographyLayer,
    TopographyLayer,
)
from .log import create_logger
from .units import mph_to_ftpm, scale_ms_to_ftpm, str_to_minutes

log = create_logger(__name__)


class ConfigError(Exception):
    """
    Exception class for Config class
    """

    pass


@dataclasses.dataclass
class AreaConfig:
    screen_size: int
    pixel_scale: float

    def __post_init__(self) -> None:
        self.screen_size = int(self.screen_size)
        self.pixel_scale = float(self.pixel_scale)


@dataclasses.dataclass
class DisplayConfig:
    fire_size: int
    control_line_size: int

    def __post_init__(self) -> None:
        self.fire_size = int(self.fire_size)
        self.control_line_size = int(self.control_line_size)


@dataclasses.dataclass
class SimulationConfig:
    def __init__(
        self, update_rate: str, runtime: str, headless: bool, record: bool
    ) -> None:
        self.update_rate = float(update_rate)
        self.runtime = str_to_minutes(runtime)
        self.headless = headless
        self.record = record


@dataclasses.dataclass
class MitigationConfig:
    ros_attenuation: bool

    def __post_init__(self) -> None:
        self.ros_attenuation = bool(self.ros_attenuation)


@dataclasses.dataclass
class OperationalConfig:
    seed: Optional[int]
    latitude: float
    longitude: float
    height: float
    width: float
    resolution: float  # TODO: Make enum for resolution?

    def __post_init__(self) -> None:
        self.latitude = float(self.latitude)
        self.longitude = float(self.longitude)
        self.height = float(self.height)
        self.width = float(self.width)
        self.resolution = float(self.resolution)


@dataclasses.dataclass
class FunctionalConfig:
    """
    Class that tracks functional layer names and keyword arguments.
    """

    name: str
    kwargs: Dict[str, Any]


@dataclasses.dataclass
class TerrainConfig:
    """
    Class that tracks the terrain topography and fuel layers.
    The fuel and terrain function fields are optional. They are used for
    functional layers and ignored for operational layers.
    """

    topography_type: str
    topography_layer: TopographyLayer
    fuel_type: str
    fuel_layer: FuelLayer
    topography_function: Optional[FunctionalConfig] = None
    fuel_function: Optional[FunctionalConfig] = None


@dataclasses.dataclass
class FireConfig:
    def __init__(self, fire_initial_position: str, max_fire_duration: str):
        fire_pos = fire_initial_position[1:-1].split(",")
        self.fire_initial_position = (int(fire_pos[0]), int(fire_pos[1]))
        self.max_fire_duration = int(max_fire_duration)


@dataclasses.dataclass
class EnvironmentConfig:
    moisture: float

    def __post_init__(self) -> None:
        self.moisture = float(self.moisture)


@dataclasses.dataclass
class WindConfig:
    speed: np.ndarray
    direction: np.ndarray
    speed_function: Optional[FunctionalConfig] = None
    direction_function: Optional[FunctionalConfig] = None


@dataclasses.dataclass
class Config:
    def __init__(self, path: Union[str, Path], cfd_precompute: bool = False) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.yaml_data = self._load_yaml()

        # Save the original screen size in case the simulation changes from
        # operational to functional
        self.original_screen_size = self.yaml_data["area"]["screen_size"]

        self.lat_long_box = self._make_lat_long_box()

        self.area = self._load_area()
        self.display = self._load_display()
        self.simulation = self._load_simulation()
        self.mitigation = self._load_mitigation()
        self.operational = self._load_operational()
        self.terrain = self._load_terrain()
        self.fire = self._load_fire()
        self.environment = self._load_environment()
        self.wind = self._load_wind()

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Loads the YAML file specified in self.path and returns the data as a dictionary.

        Returns:
            The YAML data as a dictionary
        """
        try:
            with open(self.path, "r") as f:
                try:
                    yaml_data = yaml.safe_load(f)
                except ParserError:
                    message = f"Error parsing YAML file at {self.path}"
                    log.error(message)
                    raise ConfigError(message)
        except FileNotFoundError:
            message = f"Error opening YAML file at {self.path}. Does it exist?"
            log.error(message)
            raise ConfigError(message)
        return yaml_data

    def _make_lat_long_box(self) -> Union[LatLongBox, None]:
        """
        Optionally create the LatLongBox used by the data layers if any
        of the data layers are of type `operational`.

        Returns:
            None if none of the data layers are `operational`
            LatLongBox with the config coordinates and shape if any of
            the data layers are `operational`
        """
        if (
            self.yaml_data["terrain"]["topography"]["type"] == "operational"
            or self.yaml_data["terrain"]["fuel"]["type"] == "operational"
        ):
            lat = self.yaml_data["operational"]["latitude"]
            lon = self.yaml_data["operational"]["longitude"]
            height = self.yaml_data["operational"]["height"]
            width = self.yaml_data["operational"]["width"]
            resolution = self.yaml_data["operational"]["resolution"]
            return LatLongBox((lat, lon), height, width, resolution)
        else:
            return None

    def _load_area(self) -> AreaConfig:
        """
        Load the AreaConfig from the YAML data.

        returns:
            The YAML data converted to an AreaConfig dataclass
        """
        # No processing needed for the AreaConfig
        if self.lat_long_box is not None:
            # Overwite the screen_size since operational data will determine
            # its own screen_size based on lat/long input
            # Height and width are the same since we assume square input
            height = self.lat_long_box.tr[0][0] - self.lat_long_box.bl[0][0]
            self.yaml_data["area"]["screen_size"] = height
        return AreaConfig(**self.yaml_data["area"])

    def _load_display(self) -> DisplayConfig:
        """
        Load the DisplayConfig from the YAML data.

        returns:
            The YAML data converted to a DisplayConfig dataclass
        """
        # No processing needed for the DisplayConfig
        return DisplayConfig(**self.yaml_data["display"])

    def _load_simulation(self) -> SimulationConfig:
        """
        Load the SimulationConfig from the YAML data.

        returns:
            The YAML data converted to a SimulationConfig dataclass
        """
        # No processing needed for the SimulationConfig
        return SimulationConfig(**self.yaml_data["simulation"])

    def _load_mitigation(self) -> MitigationConfig:
        """
        Load the MitigationConfig from the YAML data.

        returns:
            The YAML data converted to a MitigationConfig dataclass
        """
        # No processing needed for the MitigationConfig
        return MitigationConfig(**self.yaml_data["mitigation"])

    def _load_operational(self) -> OperationalConfig:
        """
        Load the OperationalConfig from the YAML data.

        returns:
            The YAML data converted to an OperationalConfig dataclass
        """
        # No processing needed for the OperationalConfig
        return OperationalConfig(**self.yaml_data["operational"])

    def _load_terrain(self) -> TerrainConfig:
        """
        Load the TerrainConfig from the YAML data.

        returns:
            The YAML data converted to a TerrainConfig dataclass
        """
        topo_type = self.yaml_data["terrain"]["topography"]["type"]
        fuel_type = self.yaml_data["terrain"]["fuel"]["type"]

        topo_type, topo_layer, topo_name, topo_kwargs = self._create_topography_layer(
            init=True
        )
        if topo_name is not None and topo_kwargs is not None:
            topo_fn = FunctionalConfig(topo_name, topo_kwargs)
        else:
            topo_fn = None

        fuel_type, fuel_layer, fuel_name, fuel_kwargs = self._create_fuel_layer(init=True)
        if fuel_name is not None and fuel_kwargs is not None:
            fuel_fn = FunctionalConfig(fuel_name, fuel_kwargs)
        else:
            fuel_fn = None

        return TerrainConfig(
            topo_type, topo_layer, fuel_type, fuel_layer, topo_fn, fuel_fn
        )

    def _create_topography_layer(
        self, init: bool = False, seed: Optional[int] = None
    ) -> Tuple[str, TopographyLayer, Optional[str], Optional[Dict[str, Any]]]:
        """
        Create a TopographyLayer given the config parameters.
        This is used for initalization and after resetting the layer seeds.

        Arguments:
            seed: A randomization seed used by

        Returns:
            A tuple containing:
                A string representing the `type` of the layer (`operational`,
                    `functional`, etc.)
                A FunctionalTopographyLayer that utilizes the fuction specified by
                    fn_name and the keyword arguments in kwargs
                The function name if a functional layer is used. Otherwise None
                The keyword arguments for the function if a functinoal layer is used.
                    Otherwise None
        """
        topo_layer: TopographyLayer
        topo_type = self.yaml_data["terrain"]["topography"]["type"]
        if topo_type == "operational":
            if self.lat_long_box is not None:
                topo_layer = OperationalTopographyLayer(self.lat_long_box)
            else:
                raise ConfigError(
                    "The topography layer type is `operational`, "
                    "but self.lat_long_box is None"
                )
            fn_name = None
            kwargs = None
        elif topo_type == "functional":
            fn_name = self.yaml_data["terrain"]["topography"]["functional"]["function"]
            try:
                kwargs = self.yaml_data["terrain"]["topography"]["functional"][fn_name]
            # No kwargs found (flat is an example of this)
            except KeyError:
                kwargs = {}
            # Reset the seed if this isn't the inital creation
            if "seed" in kwargs and not init:
                kwargs["seed"] = seed
            if fn_name == "perlin":
                fn = perlin(**kwargs)
            elif fn_name == "gaussian":
                fn = gaussian(**kwargs)
            elif fn_name == "flat":
                fn = flat()
            else:
                raise ConfigError(
                    f"The specified topography function ({fn_name}) " "is not valid."
                )
            topo_layer = FunctionalTopographyLayer(
                self.yaml_data["area"]["screen_size"],
                self.yaml_data["area"]["screen_size"],
                fn,
                fn_name,
            )
        else:
            raise ConfigError(
                f"The specified topography type ({topo_type}) " "is not supported"
            )

        return topo_type, topo_layer, fn_name, kwargs

    def _create_fuel_layer(
        self, init: bool = False, seed: Optional[int] = None
    ) -> Tuple[str, FuelLayer, Optional[str], Optional[Dict[str, Any]]]:
        """
        Create a FuelLayer given the config parameters.
        This is used for initalization and after resetting the layer seeds.

        Returns:
            A FunctionalFuelLayer that utilizes the fuction specified by
            fn_name and the keyword arguments in kwargs
        """
        fuel_layer: FuelLayer
        fuel_type = self.yaml_data["terrain"]["fuel"]["type"]
        if fuel_type == "operational":
            if self.lat_long_box is not None:
                fuel_layer = OperationalFuelLayer(self.lat_long_box)
            else:
                raise ConfigError(
                    "The topography layer type is `operational`, "
                    "but self.lat_long_box is None"
                )
            fn_name = None
            kwargs = None
        elif fuel_type == "functional":
            fn_name = self.yaml_data["terrain"]["fuel"]["functional"]["function"]
            try:
                kwargs = self.yaml_data["terrain"]["fuel"]["functional"][fn_name]
            # No kwargs found (some functions don't need input arguments)
            except KeyError:
                kwargs = {}
            # Reset the seed if this isn't the inital creation
            if "seed" in kwargs and not init:
                kwargs["seed"] = seed
            if fn_name == "chaparral":
                fn = chaparral_fn(**kwargs)
            else:
                raise ConfigError(
                    f"The specified fuel function ({fn_name}) " "is not valid."
                )
            fuel_layer = FunctionalFuelLayer(
                self.yaml_data["area"]["screen_size"],
                self.yaml_data["area"]["screen_size"],
                fn,
                fn_name,
            )
        else:
            raise ConfigError(
                f"The specified fuel type ({fuel_type}) " "is not supported"
            )

        return fuel_type, fuel_layer, fn_name, kwargs

    def _load_fire(self) -> FireConfig:
        """
        Load the FireConfig from the YAML data.

        Returns:
            The YAML data converted to a FireConfig dataclass
        """
        # No processing needed for the FireConfig
        return FireConfig(**self.yaml_data["fire"])

    def _load_environment(self) -> EnvironmentConfig:
        """
        Load the EnvironmentConfig from the YAML data.

        Returns:
            The YAML data converted to a EnvironmentConfig dataclass
        """
        # No processing needed for the EnvironmentConfig
        return EnvironmentConfig(**self.yaml_data["environment"])

    def _load_wind(self) -> WindConfig:
        """
        Load the WindConfig from the YAML data.

        Returns:
            The YAML data converted to a WindConfig dataclass
        """
        # Only support simple for now
        # TODO: Figure out how Perlin and CFD create wind
        fn_name = self.yaml_data["wind"]["function"]
        if fn_name == "simple":
            arr_shape = (
                self.yaml_data["area"]["screen_size"],
                self.yaml_data["area"]["screen_size"],
            )
            speed = self.yaml_data["wind"]["simple"]["speed"]
            direction = self.yaml_data["wind"]["simple"]["direction"]
            speed_arr = np.full(arr_shape, speed)
            direction_arr = np.full(arr_shape, direction)
            speed_kwargs = None
            dir_kwargs = None
        elif fn_name == "cfd":
            # Check if wind files have been generated
            cfd_generated = os.path.isfile(
                "generated_wind_directions.npy"
            ) and os.path.isfile("generated_wind_magnitudes.npy")
            if cfd_generated is False:
                log.error("Missing pregenerated cfd npy files, switching to perlin")
                self.wind_function = "perlin"
            else:
                speed_arr = np.load("generated_wind_magnitudes.npy")
                direction_arr = np.load("generated_wind_directions.npy")
                speed_arr = scale_ms_to_ftpm(speed_arr)
            speed_kwargs = self.yaml_data["wind"]["cfd"]
            dir_kwargs = self.yaml_data["wind"]["cfd"]
        elif fn_name == "perlin":
            wind_map = WindController()
            speed_kwargs = deepcopy(self.yaml_data["wind"]["perlin"]["speed"])
            range_min = mph_to_ftpm(
                self.yaml_data["wind"]["perlin"]["speed"]["range_min"]
            )
            range_max = mph_to_ftpm(
                self.yaml_data["wind"]["perlin"]["speed"]["range_max"]
            )
            speed_kwargs["range_min"] = range_min
            speed_kwargs["range_max"] = range_max
            wind_map.init_wind_speed_generator(
                **speed_kwargs, screen_size=self.yaml_data["area"]["screen_size"]
            )

            direction_kwargs = self.yaml_data["wind"]["perlin"]["direction"]
            wind_map.init_wind_direction_generator(
                **direction_kwargs, screen_size=self.yaml_data["area"]["screen_size"]
            )
            if wind_map.map_wind_speed is not None:
                speed_arr = wind_map.map_wind_speed
            else:
                raise ConfigError(
                    "The Perlin WindController is specified in the config, "
                    "but returned None for the wind speed"
                )
            if wind_map.map_wind_direction is not None:
                direction_arr = wind_map.map_wind_direction
            else:
                raise ConfigError(
                    "The Perlin WindController is specified in the config, "
                    "but returned None for the wind direction"
                )
            direction_arr = wind_map.map_wind_direction
            speed_kwargs = self.yaml_data["wind"]["perlin"]["speed"]
            dir_kwargs = self.yaml_data["wind"]["perlin"]["direction"]
        else:
            raise ConfigError(f"Wind type {fn_name} is not supported")

        if fn_name is not None and speed_kwargs is not None:
            speed_fn = FunctionalConfig(fn_name, speed_kwargs)
        else:
            speed_fn = None
        if fn_name is not None and dir_kwargs is not None:
            direction_fn = FunctionalConfig(fn_name, dir_kwargs)
        else:
            direction_fn = None

        # Convert to float to get correct type
        speed_arr = speed_arr.astype(np.float64)
        direction_arr = direction_arr.astype(np.float64)

        return WindConfig(speed_arr, direction_arr, speed_fn, direction_fn)

    def reset_terrain(
        self,
        topography_seed: Optional[int] = None,
        topography_type: Optional[str] = None,
        fuel_seed: Optional[int] = None,
        fuel_type: Optional[str] = None,
        location: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Reset the terrain functional generation seeds if using functional data,
        or reset the terrain lat/long location if using operational data.

        Arguments:
            topography_seed: The seed used to randomize functional topography generation
            fuel_seed: The seed used to randomize functional fuel generation
            location: A new center-point for the operational topography and fuel data
        """
        # We want to update the YAML terrain data so that the call to _load_terrain()
        # re-create the layers with the updated parameters

        # Do the location first, as the creation of the LatLongBox depends on it
        if location is not None:
            # Since all operational layers use the LatLongBox, we can update
            # the yaml data and the LatLongBox at the class level
            lat, long = location
            self.yaml_data["operational"]["latitude"] = lat
            self.yaml_data["operational"]["longitude"] = long
            self.lat_long_box = self._make_lat_long_box()

        # Can only reset functional topography seeds, since operational is updated
        # via the `location` argument
        if topography_seed is not None:
            # Working with functional data
            if self.terrain.topography_function is not None:
                topo_fn_name = self.terrain.topography_function.name
                self.yaml_data["terrain"]["topography"]["functional"][topo_fn_name][
                    "seed"
                ] = topography_seed
        # Can only reset functional fuel seeds, since operational is updated
        # via the `location` argument
        if fuel_seed is not None:
            # Working with functional data
            if self.terrain.fuel_function is not None:
                fuel_fn_name = self.terrain.fuel_function.name
                self.yaml_data["terrain"]["fuel"]["functional"][fuel_fn_name][
                    "seed"
                ] = fuel_seed

        # Need to check if any data layer types are changing, since the
        # screen_size could be affected
        if topography_type is not None and fuel_type is not None:
            # Special case when going from all operational to all functional, so
            # we need to revert back to the original screen_size from the config file
            if topography_type == "operational" and fuel_type == "operational":
                if (
                    self.terrain.topography_type == "functional"
                    and self.terrain.fuel_type == "functional"
                ):
                    self.yaml_data["screen_size"] = self.original_screen_size
        if topography_type is not None:
            # Update the yaml data
            self.yaml_data["terrain"]["topography"]["type"] = topography_type
        if fuel_type is not None:
            # Update the yaml data
            self.yaml_data["terrain"]["fuel"]["type"] = fuel_type

        # Remake the LatLongBox
        self.lat_long_box = self._make_lat_long_box()
        # Remake the AreaConfig since operational/functional could have changed
        self.area = self._load_area()
        # Remake the terrain
        self.terrain = self._load_terrain()

    def reset_wind(
        self, speed_seed: Optional[int] = None, direction_seed: Optional[int] = None
    ) -> None:
        """
        Reset the wind speed and direction seeds.

        Arguments:
            speed_seed: The seed used to randomize wind speed generation
            direction_seed: The seed used to randomize wind direction generation
        """
        # We want to update the YAML wind data so that the call to _load_wind()
        # re-create the WindConfig with the updated parameters
        if speed_seed is not None:
            # Working with functional data
            if self.wind.speed_function is not None:
                speed_fn_name = self.wind.speed_function.name
                if "seed" in self.yaml_data["wind"][speed_fn_name]["speed"]:
                    self.yaml_data["wind"][speed_fn_name]["speed"]["seed"] = speed_seed
                else:
                    log.warn(
                        "Attempted to reset speed seed for wind fucntion "
                        f"{speed_fn_name}, but no seed parameter exists in the config"
                    )

        if direction_seed is not None:
            if self.wind.direction_function is not None:
                direction_fn_name = self.wind.direction_function.name
                if "seed" in self.yaml_data["wind"][direction_fn_name]["direction"]:
                    self.yaml_data["wind"][direction_fn_name]["direction"][
                        "seed"
                    ] = direction_seed
                else:
                    log.warn(
                        "Attempted to reset direction seed for wind fucntion "
                        f"{direction_fn_name}, but no seed parameter exists in the "
                        "config"
                    )

        self.wind = self._load_wind()

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the current config to the specified path.

        Arguments:
            path: The path and filename of the output YAML file
        """
        with open(path, "w") as f:
            yaml.dump(self.yaml_data, f)
