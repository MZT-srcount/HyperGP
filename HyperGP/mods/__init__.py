from .mod_base import ModBase
from .multiprocess_parallel import MultiProcess


class AvailableMods:
    parallel: MultiProcess = MultiProcess

class __Mods:
    parallel: MultiProcess = MultiProcess()

