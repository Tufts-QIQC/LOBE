class LadderOperator:

    def __init__(self, particle_type, mode, creation):
        """ """
        self.particle_type = (
            particle_type  # (0 -> fermionic), (1 -> antifermionic) (2 -> bosonic)
        )
        self.mode = mode  # index of the mode the operator acts on
        self.creation = (
            creation  # boolean dictating creation op (True) or annihilation op (False)
        )
