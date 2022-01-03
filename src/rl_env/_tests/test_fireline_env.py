import os
import unittest
from unittest import mock

import numpy as np

from ...game.game import Game
from ...utils.config import Config
from ...game.sprites import Terrain
from ...enums import BurnStatus, GameStatus
from ...rl_env.fireline_env import FireLineEnv, RLEnv
from ...game.managers.fire import RothermelFireManager
from ...game.managers.mitigation import FireLineManager
from ...world.parameters import Environment, FuelParticle


class RLEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Initialize the FireLineEnv class and instantiate a fireline action.
        '''
        self.config = Config('./src/rl_env/_tests/test_config.yml')
        self.fireline_env = FireLineEnv(self.config)
        self.rl_env = RLEnv(self.fireline_env)
        self.action = 1
        self.current_agent_loc = (1, 1)

    def test_init(self) -> None:
        '''
        Test setting the seed for the terrain map.

        '''
        seed = 1212
        fireline_env_seed = FireLineEnv(self.config, seed)
        fireline_env_seed = fireline_env_seed.config.terrain_map
        fireline_env_no_seed = FireLineEnv(self.config)
        fireline_env_no_seed = fireline_env_no_seed.config.terrain_map

        # assert these envs are different
        for i, j in zip(fireline_env_seed, fireline_env_no_seed):
            for fuel_i, fuel_j in zip(i, j):
                self.assertNotEqual(
                    fuel_i.w_0,
                    fuel_j.w_0,
                    msg='Different seeds should produce different terrain '
                    'maps.')

        # assert equal Fuel Maps
        fireline_env_same_seed = FireLineEnv(self.config, seed)
        fireline_env_same_seed = fireline_env_same_seed.config.terrain_map
        self.assertEqual(fireline_env_seed,
                         fireline_env_same_seed,
                         msg='Same seeds should produce the same terrain '
                         'maps.')

    def test_step(self) -> None:
        '''
        Test that the call to `step()` runs through properly.

        `step()` calls a handful of sub-functions, which also get tested.

        TODO: This will change with updates to the state format
        '''
        self.rl_env.reset()
        state, reward, done, _ = self.rl_env.step(self.action)

        agent_pos = np.where(state[0] == 1)

        self.assertEqual(
            agent_pos, (0, 1),
            msg=(f'The returned state of agent position of the game is {agent_pos}, '
                 f'but it should be (1, 0)'))

        self.assertEqual(
            reward,
            0,
            msg=(f'The returned state of reward of the game is {reward}, but it '
                 f'should be -1'))

        self.assertEqual(done,
                         False,
                         msg=(f'The returned state of the game is {done}, but it '
                              f'should be False'))

    def test_reset(self) -> None:
        '''
        Test that the call to `reset()` runs through properly.

        `reset()` calls a handful of sub-functions, which also get tested.

        Assert agent position is returned to upper left corner (1,0) of game.

        state_space:

        0:      'position'
        1:      'terrain: w_0'
        2:      'elevation'
        3:      'mitigation'
        '''
        state = self.rl_env.reset()

        agent_pos = np.where(state[0] == 1)
        fuel_arrays = state[1]
        fireline = state[-1].max()
        elevation = state[2]

        w_0_array = np.array([
            self.fireline_env.terrain.fuel_arrs[i][j].fuel.w_0
            for j in range(self.fireline_env.config.area.screen_size)
            for i in range(self.fireline_env.config.area.screen_size)
        ]).reshape(self.fireline_env.config.area.screen_size,
                   self.fireline_env.config.area.screen_size)

        self.assertEqual(
            agent_pos, (0, 0),
            msg=(f'The returned state of the agent position is {agent_pos}, but it '
                 f'should be [0, 0]'))

        self.assertTrue(
            (fuel_arrays == w_0_array).all(),
            msg=(f'The returned state of the fuel arrays is {fuel_arrays}, but it '
                 f'should be 1'))

        self.assertEqual(fireline,
                         0,
                         msg=(f'The returned state of the fireline is {fireline}, but it '
                              f'should be 0'))
        elevation_zero_min = self.fireline_env.terrain.elevations - \
                             self.fireline_env.terrain.elevations.min()
        valid_elevation = elevation_zero_min / (elevation_zero_min.max() + 1e-6)
        self.assertTrue(
            (elevation == valid_elevation).all(),
            msg=('The returned state of the terrain elevation map is not the same '
                 'as the initialized terrain elevation map'))

    def test_update_current_agent_loc(self) -> None:
        '''
        Test that the call to `_update_current_agent_loc()` runs through properly.

        The agent position should only be at a single location per `step()`.
        '''

        self.rl_env.reset()
        self.rl_env.current_agent_loc = self.current_agent_loc
        x = self.current_agent_loc[0]
        y = self.current_agent_loc[1]
        new_agent_loc = (x, y + 1)
        self.rl_env._update_current_agent_loc()

        self.assertEqual(
            self.rl_env.current_agent_loc,
            new_agent_loc,
            msg=f'The returned agent location is {self.current_agent_loc}, but it '
            f'should be {new_agent_loc}.')


@mock.patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy'})
class FireLineEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Initialize the `FireLineEnv` class and instantiate a fireline action.
        '''
        self.config = Config('./src/rl_env/_tests/test_config.yml')
        self.fireline_env = FireLineEnv(self.config)
        self.mitigation = True
        self.fire_spread = False

        self.game = Game(self.config.area.screen_size)
        self.fuel_particle = FuelParticle()
        self.fuel_arrs = [[
            self.config.terrain.fuel_array_function(x, y)
            for x in range(self.config.area.terrain_size)
        ] for y in range(self.config.area.terrain_size)]
        self.terrain = Terrain(self.fuel_arrs, self.config.terrain.elevation_function,
                               self.config.area.terrain_size,
                               self.config.area.screen_size)
        self.environment = Environment(self.config.environment.moisture,
                                       self.config.environment.wind_speed,
                                       self.config.environment.wind_direction)

        # initialize all mitigation strategies
        self.fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain)
        self.fireline_sprites = self.fireline_manager.sprites
        self.fireline_sprites_reset = self.fireline_manager.sprites.copy()
        self.fire_manager = RothermelFireManager(
            self.config.fire.fire_initial_position, self.config.display.fire_size,
            self.config.fire.max_fire_duration, self.config.area.pixel_scale,
            self.config.simulation.update_rate, self.fuel_particle, self.terrain,
            self.environment)
        self.fire_sprites = self.fire_manager.sprites

    def test_render(self) -> None:
        '''
        Test that the call to `_render()` runs through properly.

        This should be pass as long as the calls to `fireline_manager.update()` and
        `fire_map.update()` pass tests.

        Assert the points get updated in the `fireline_sprites` group.

        '''
        current_agent_loc = (1, 1)

        # Test rendering 'inline' (as agent traverses)
        self.fireline_env._render(1, current_agent_loc, inline=True)
        # assert the points are placed
        self.assertEqual(self.fireline_env.fireline_manager.sprites[0].pos,
                         current_agent_loc,
                         msg=(f'The position of the sprite is '
                              f'{self.fireline_env.fireline_manager.sprites[0].pos} '
                              f', but it should be {current_agent_loc}'))

        # Test Full Mitigation (after agent traversal)
        self.fireline_sprites = self.fireline_sprites_reset
        mitigation = np.full((self.config.area.screen_size, self.config.area.screen_size),
                             1)
        self.fireline_env._render(
            mitigation, (self.config.area.screen_size, self.config.area.screen_size))
        # assert the points are placed
        self.assertEqual(len(self.fireline_env.fireline_manager.sprites),
                         self.config.area.screen_size**2 + 1,
                         msg=(f'The number of sprites updated is '
                              f'{len(self.fireline_env.fireline_manager.sprites)} '
                              f', but it should be {self.config.area.screen_size**2+1}'))

        # Test Full Mitigation (after agent traversal) and fire spread

        # assert the points are placed and fire can spread
        self.fireline_sprites = self.fireline_sprites_reset

        mitigation = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        # start the fire where we have a control line
        mitigation[self.config.fire.fire_initial_position[0] - 1:] = 1
        self.fireline_env._render(
            mitigation, (self.config.area.screen_size, self.config.area.screen_size),
            mitigation_only=False,
            mitigation_and_fire_spread=True)

        self.assertEqual(
            self.fireline_env.fire_status,
            GameStatus.QUIT,
            msg=f'The returned state of the Game is {self.fireline_env.game_status} '
            ' but, should be GameStatus.QUIT.')

    def test_update_sprite_points(self) -> None:
        '''
        Test that the call to `_update_sprites()` runs through properly.

        Since self.action is instantiated as `1`, we need to verify that a fireline sprite
        is created and added to the `fireline_manager`.
        '''

        # assert points get updated 'inline' as agent traverses
        current_agent_loc = (1, 1)
        self.mitigation = 1
        points = set([current_agent_loc])
        self.fireline_env._update_sprite_points(self.mitigation,
                                                current_agent_loc,
                                                inline=True)
        self.assertEqual(self.fireline_env.points,
                         points,
                         msg=f'The sprite was updated at {self.fireline_env.points}, '
                         f'but it should have been at {current_agent_loc}')

        # assert points get updated after agent traverses entire game
        current_agent_loc = (self.config.area.screen_size, self.config.area.screen_size)
        self.mitigation = np.full(
            (self.config.area.screen_size, self.config.area.screen_size), 1)
        points = [(i, j) for j in range(self.config.area.screen_size)
                  for i in range(self.config.area.screen_size)]
        points = set(points)
        self.fireline_env._update_sprite_points(self.mitigation,
                                                current_agent_loc,
                                                inline=False)
        self.assertEqual(
            self.fireline_env.points,
            points,
            msg=f'The number of sprites updated was {len(self.fireline_env.points)} '
            f', but it should have been {len(points)} sprites.')

    def test_reset_state(self) -> None:
        '''
        Test that the call to `_convert_data_to_gym()` runs through properly.

        This function returns the state as an array.

        TODO: Waiting for update to the state space
        '''
        pass

    def test_run(self) -> None:
        '''
        Test that the call to `_run` runs the simulation properly.

        This function returns the burned firemap with or w/o mitigation.

        This function will reset the `fire_map` to all `UNBURNED` pixels at each call to
        the method.

        This should pass as long as the calls to `fireline_manager.update()`
        and `fire_map.update()` pass tests.
        '''
        mitigation = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        mitigation[1, 0] = 1
        position = np.zeros((self.config.area.screen_size, self.config.area.screen_size))
        position[self.config.area.screen_size - 1, self.config.area.screen_size - 1] = 1

        fire_map = np.full((self.config.area.screen_size, self.config.area.screen_size),
                           BurnStatus.BURNED)

        self.fire_map = self.fireline_env._run(mitigation, position, False)
        # assert the fire map is all BURNED
        self.assertEqual(
            self.fire_map.max(),
            fire_map.max(),
            msg=f'The fire map has a maximum BurnStatus of {self.fire_map.max()} '
            f', but it should be {fire_map.max()}')

        # assert fire map has BURNED and FIRELINE pixels
        fire_map[1, 0] = 3
        self.fire_map = self.fireline_env._run(mitigation, position, True)
        self.assertEqual(len(np.where(self.fire_map == 3)),
                         len(np.where(fire_map == 1)),
                         msg=f'The fire map has a mitigation sprite of length '
                         f'{len(np.where(self.fire_map == 3))}, but it should be '
                         f'{len(np.where(fire_map == 1))}')

    def test_compare_states(self) -> None:
        '''
        Test that the call to `_compare_states` runs the comparison of state spaces
        properly.

        This function returns the overall reward.
        '''
        screen_size = self.fireline_env.config.area.screen_size
        # create array of BURNED pixels
        fire_map = np.full((screen_size, screen_size), 2)
        # create array of agent mitigation + fire spread (BURNED pixels)
        fire_map_with_agent = np.full((screen_size, screen_size), 3)

        unmodified_reward = -1 * (screen_size * screen_size)
        modified_reward = -1 * (screen_size * screen_size)
        test_reward = (modified_reward - unmodified_reward) / \
                        (screen_size * screen_size)
        reward = self.fireline_env._compare_states(fire_map, fire_map_with_agent)

        # assert rewards are the same
        self.assertEqual(reward,
                         test_reward,
                         msg=(f'The returned reward of the game is {reward}, but it '
                              f'should be {test_reward}'))
