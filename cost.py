
class Cost:
    def __init__(self, alg_choice, tiling_choice, recommend_retiling, retiling_diff, cost):
        self.algorithms = alg_choice
        self.tilings = tiling_choice
        self.retiling = recommend_retiling
        self.retiling_diff = retiling_diff
        self.cost_val = cost

    def __str__(self):
        return str(self.algorithms)+'\n' + \
               str(self.tilings) + '\n' + \
               str(self.retiling) + '\n' + \
               str(self.retiling_diff) + '\n' + \
               str(self.cost_val)