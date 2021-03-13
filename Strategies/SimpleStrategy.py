__author__ = "camzz"

from Blotto.BaseStrategy import BaseStrategy

class SimpleStrategy(BaseStrategy):
    """
    Copy this strategy and rename it to create your own!
    Check out CopierStrategy for an example
    """
    def __init__(self, name, my_argument):
        """
        Create your strategy my writing MyStrategy(name, my_argument)
        
        Writing this method is entirely optional, if you dont want any extra arguments, leave it out
        If you leave it out, create your strategy my writing MyStrategy(name)
        
        Choose a cool name for extra points*!
        """
        super(SimpleStrategy, self).__init__(name)
        
        #Store any variables you need later here
        self.my_argument = my_argument


    def soldiers_request(self, iteration):
        """
        Here you are asked what your next move will be, return it 
        iteration is the number of which battle we are on 0->N (you probably wont need it)
        
        You can look at:
            self.opponent_allocations
            which is a list of the opponents strategies so far. Careful not to change it!
            
            self.past_scores
            which is a list of the scores so far in this match.
            
            self.num_fields
            number of fields to place orders in, this should be the length of the list you return.
            Or just assume its 8 for the main contest
            
            self.my_argument
            You can use anything which you put in at setup (__init__ above).
            This might be used for more complicated strategies.
        """
        #Do some calucalations, use some random numbers, do whatever you want
        strategy = [50, 0, 50, 0, 0, 0, 0, 0]
        return strategy
