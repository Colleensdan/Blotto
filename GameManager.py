__author__ = 'camzzz'

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("no matplotlib installed")
    quit()

from Blotto.general_utils import sign, cumulative_sum


class GameManager(object):
    """
    This class battles two strategies, you dont need to know how it works but to set it up for testing for the main competition:
    gm = GameManager(strategy_A, strategy_B, 8, 10000)
    gm.run()
    
    use gm.plot_results() to see a nice plot *requires matplotlib
    use gm.declare_winner() to see how you did!
    """
    def __init__(self, stategy_A, strategy_B, num_fields, num_runs, castle_weights=[1,2,3,4,5,6,7,8,9,10], total_score=False):
        self.strategy_A = stategy_A
        self.strategy_B = strategy_B

        self.num_fields = num_fields
        self.num_runs = num_runs

        #Use the total score over many games, or number of individual games won, regardless of score
        self.total_score = total_score

        #results is a list of scores, positive is towards A, negative to B
        self.results = []

        self.castle_weights = castle_weights

    def run(self):
        #initialise strategies
        self.strategy_A.initialise(num_fields=self.num_fields,
                                   num_runs=self.num_runs)
        self.strategy_B.initialise(num_fields=self.num_fields,
                                   num_runs=self.num_runs)

        for i in range(self.num_runs):
            self.iteration(i)

        return self.results

    def iteration(self, i):

        # returns the soldiers and their distribution
        soldiers_A = self.strategy_A.soldiers_request(i)
        soldiers_B = self.strategy_B.soldiers_request(i)

        #Check what was placed is valid
        check_A = self.check_solders(soldiers_A)
        check_B = self.check_solders(soldiers_B)

        if check_A  == 0 or check_B  == 0:
            #TODO if uploading to github - raise error which derives from BaseException
            print("invalid soldier placement")
            quit()


        #Resolve Battle
        score = self.resolve_battle(soldiers_A, soldiers_B, check_A, check_B)

        #Infom strategies
        self.strategy_A.post_results(score, soldiers_B, check_B)
        self.strategy_B.post_results(-score, soldiers_A, check_A)

        self.results.append(score)

    def check_solders(self, soldiers):
        check = 1
        try:
            # 1 if arr of same length as number of battlefields
            check *= len(soldiers) == self.num_fields
            # 1 if all ints
            check *= all(int(s) == s for s in soldiers)
            # 1 if all positive
            check *= all(a >= 0 for a in soldiers)
            # 1 if no. soldiers is less than or equal to 100
            check *= sum(soldiers) <= 100.0
        except (TypeError, IndexError) as _:
            check = 0

        return check

    def resolve_battle(self, soldiers_A, soldiers_B, check_A, check_B, consecA=[], consecB=[]):
        # if b invalid
        if check_A and not check_B:
            return self.num_fields
        if check_B and not check_A:
            return -self.num_fields
        if not check_B and not check_A:
            return 0

        scoreA =0
        scoreB=0

        for i in range(self.num_fields):

            # test for the consecutive rule with deleting arrays and tally score within the iterable

            if soldiers_A[i] > soldiers_B[i]:
                # tally scores
                scoreA += self.castle_weights[i]

                # tally consecutive soldiers
                consecA.append(1)

                # check for three then edit score
                if len(consecA) == 3:
                    for battlefield in range(i,self.num_fields+1):
                        scoreA += self.castle_weights[battlefield-1]

                    return scoreA-scoreB
            else:
                consecA =[]

            if soldiers_A[i] < soldiers_B[i]:
                scoreB += self.castle_weights[i]
                consecB.append(1)

                # check for three then edit score
                if len(consecB) == 3:
                    for battlefield in range(i, self.num_fields+1):
                        scoreB += self.castle_weights[battlefield-1]
                    return scoreA-scoreB
            else:
                consecB = []

        return scoreA-scoreB

    def plot_results(self):
        data = list(cumulative_sum(self.results))
        plt.plot(data)
        plt.ylabel('A - B points')
        plt.show(block=False)

    # def get_max_score(self):
    #     if self.total_score:
    #         return self.num_runs * self.num_fields
    #     else:
    #         return self.num_runs

    def get_max_score(self):
        # is not this because of the consecutive rule
        return self.num_runs * ((self.num_fields*(self.num_fields+1))/2)  # triangular numbers



    def declare_winner(self):
        s = sum(self.results)
        return 100 * s / self.get_max_score()

    def declare_winner_old(self):
        s = sum(self.results)
        if s > 0:
            print(("%s is the winner by %s points!" % (self.strategy_A.name, s)))
            print(("Max possible points: %s" % self.get_max_score()))
            print(("Percentage of highest score: %.2f%%" % (100 * s / self.get_max_score())))

        elif s < 0:
            print(("%s is the winner by %s points!" % (self.strategy_B.name, -s)))
            print(("Max possible points: %s" % self.get_max_score()))
            print(("Percentage of highest score: %.2f%%" % (100 * -s / self.get_max_score())))

        else:
            print("The game was a tie!")
        plt.show()

        return 100 * s / self.get_max_score()
    def get_score(self):
        return sum(self.results)

    def check_strategy_A_win(self):
        if sum(self.results) > 0:
            return True
        else:
            return False

    def check_strategy_B_win(self):
        if sum(self.results) < 0:
            return True
        else:
            return False

