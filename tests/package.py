"""
Package
-------
This will be used in CI to make sure that we can import all of the packages/modules in the
installed package. If it fails, there is likely a `src` imported somewhere instead of a
relative import.
"""
import sys
import traceback
from importlib import import_module
from modulefinder import ModuleFinder

import rothsim
from rothsim import game, sim, utils, world
from rothsim.game import managers

# Find all of the submodules
finder = ModuleFinder()
sub_modules = ["rothsim." + m for m in finder.find_all_submodules(rothsim)]
sub_modules += ["rothsim.game." + m for m in finder.find_all_submodules(game)]
sub_modules += [
    "rothsim.game.managers." + m for m in finder.find_all_submodules(managers)
]
sub_modules += ["rothsim.utils." + m for m in finder.find_all_submodules(utils)]
sub_modules += ["rothsim.world." + m for m in finder.find_all_submodules(world)]
sub_modules += ["rothsim.sim." + m for m in finder.find_all_submodules(sim)]

# Import all of the found submodules
for module in sub_modules:
    try:
        import_module(module)
    except ModuleNotFoundError:
        traceback.print_exc()
        print(
            '\nERROR: "src" was likely used as an import. Ensure your imports are '
            "relative in all modules and packages."
        )
        sys.exit(1)

print("\nSuccessfully imported modules!")
