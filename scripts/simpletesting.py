__author__ = "Linus"

# For testing strategies written in the simple.py style.
# quickTest returns a tuple (totalscore, numberOfWins).
# These are the results under the two different scoring methods.
# totalscore is player1's final score minus player2's.
# numberOfWins is number of player1's wins minus number of player2's.

# If either program plays an invalid move (e.g. too many soldiers),
# then this helpfully crashes.


def quickTest(player1, player2, weights, soldiers, iterations):
    """Call like
    quickTest(Copier, CleverStrat, 10, 100, 9000)"""
    totalscore = 0 # positive for P1 winning, negative for P2 winning
    numberOfWins = 0 # +1 when P1 wins, -1 when P2 wins

    P1 = player1(weights, soldiers)
    P2 = player2(weights, soldiers)

    move1 = next(P1)
    move2 = next(P2)
    
    for i in range(iterations):
        result = score(move1, move2, weights, soldiers)
        totalscore += result
        if result > 0: numberOfWins += 1
        if result < 0: numberOfWins -= 1
        move1, move2 = P1.send(move2), P2.send(move1)

    return totalscore, numberOfWins


def score(move1, move2, weights, soldiers):
    check(move1, weights, soldiers)
    check(move2, weights, soldiers)

    total = 0
    for i in range(len(weights)):
        if move1[i] > move2[i]: total += weights[i]
        if move1[i] < move2[i]: total -= weights[i]

    return total


def check(move, weights, soldiers):
    assert(all(a == int(a) for a in move))
    assert(sum(move) == soldiers)
    assert(len(move) == len(weights))
    move[0]
