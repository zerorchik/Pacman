from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # # Useful information you can extract from a GameState (pacman.py)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # oldFood = currentGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #
    # return successorGameState.getScore()

    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()  # Pacman position after moving
    newFood = successorGameState.getFood()  # Remaining food
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_YOUR_CODE
    listFood = newFood.asList()  # All remaining food as list
    ghostPos = successorGameState.getGhostPositions()  # Get the ghost position
    # Initialize with list
    mFoodDist = []
    mGhostDist = []

    # Find the distance of all the foods to the pacman
    for food in listFood:
      mFoodDist.append(manhattanDistance(food, newPos))

    # Find the distance of all the ghost to the pacman
    for ghost in ghostPos:
      mGhostDist.append(manhattanDistance(ghost, newPos))

    if currentGameState.getPacmanPosition() == newPos:
      return (-(float("inf")))

    for ghostDistance in mGhostDist:
      if ghostDistance < 2:
        return (-(float("inf")))

    if len(mFoodDist) == 0:
      return float("inf")
    else:
      minFoodDist = min(mFoodDist)
      maxFoodDist = max(mFoodDist)

    return 1000 / sum(mFoodDist) + 10000 / len(mFoodDist)
    # END_YOUR_CODE


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    " Max Value for computing the best direction of the pacman "
    def max_value(gameState, depth):
      " Cases checking "
      actionList = gameState.getLegalActions(0)  # Get actions of pacman
      if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
        return (self.evaluationFunction(gameState), None)
      " Initializing the value of v and action to be returned "
      v = -(float("inf"))
      goAction = None
      for thisAction in actionList:
        successorValue = min_value(gameState.generateSuccessor(0, thisAction), 1, depth)[0]
        " Get value of v and action, max(v, successorValue) "
        if (successorValue > v):
          v, goAction = successorValue, thisAction

      # Для тестування значень
      # print(v)
      return (v, goAction)

    " Min Value for computing the worst case direction of the ghost "
    def min_value(gameState, agentID, depth):
      " Cases checking "
      actionList = gameState.getLegalActions(agentID)  # Get the actions of the ghost
      if len(actionList) == 0:
        return (self.evaluationFunction(gameState), None)
      " Initializing the value of v and action to be returned "
      v = float("inf")
      goAction = None
      for thisAction in actionList:
        if (agentID == gameState.getNumAgents() - 1):
          successorValue = max_value(gameState.generateSuccessor(agentID, thisAction), depth + 1)[0]
        else:
          successorValue = min_value(gameState.generateSuccessor(agentID, thisAction), agentID + 1, depth)[0]
        " Get value of v and action, min(v, successorValue) "
        if (successorValue < v):
          v, goAction = successorValue, thisAction
      return (v, goAction)

    return max_value(gameState, 0)[1]
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    " Max value "
    def max_value(gameState, depth, alpha, beta):
      " Cases checking "
      actionList = gameState.getLegalActions(0)  # Get actions of pacman
      if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
        return (self.evaluationFunction(gameState), None)
      " Initializing the value of v and action to be returned "
      v = -(float("inf"))
      goAction = None
      for thisAction in actionList:
        successorValue = min_value(gameState.generateSuccessor(0, thisAction), 1, depth, alpha, beta)[0]
        " v = max(v, successorValue) "
        if (v < successorValue):
          v, goAction = successorValue, thisAction
        if (v > beta):
          return (v, goAction)
        alpha = max(alpha, v)

      # Для тестування значень
      # print(v)
      return (v, goAction)

    " Min value "
    def min_value(gameState, agentID, depth, alpha, beta):
      " Cases checking "
      actionList = gameState.getLegalActions(agentID)  # Get the actions of the ghost
      if len(actionList) == 0:
        return (self.evaluationFunction(gameState), None)
      " Initializing the value of v and action to be returned "
      v = float("inf")
      goAction = None
      for thisAction in actionList:
        if (agentID == gameState.getNumAgents() - 1):
          successorValue = max_value(gameState.generateSuccessor(agentID, thisAction), depth + 1, alpha, beta)[0]
        else:
          successorValue = \
          min_value(gameState.generateSuccessor(agentID, thisAction), agentID + 1, depth, alpha, beta)[0]
        " v = min(v, successorValue) "
        if (successorValue < v):
          v, goAction = successorValue, thisAction
        if (v < alpha):
          return (v, goAction)
        beta = min(beta, v)
      return (v, goAction)

    alpha = -(float("inf"))
    beta = float("inf")
    return max_value(gameState, 0, alpha, beta)[1]
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """

  # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
  pacmanPos = currentGameState.getPacmanPosition()
  ghostList = currentGameState.getGhostStates()
  foods = currentGameState.getFood()
  capsules = currentGameState.getCapsules()
  # Return based on game state
  if currentGameState.isWin():
    return float("inf")
  if currentGameState.isLose():
    return float("-inf")
  # Populate foodDistList and find minFoodDist
  foodDistList = []
  for each in foods.asList():
    foodDistList = foodDistList + [util.manhattanDistance(each, pacmanPos)]
  minFoodDist = min(foodDistList)
  # Populate ghostDistList and scaredGhostDistList, find minGhostDist and minScaredGhostDist
  ghostDistList = []
  scaredGhostDistList = []
  for each in ghostList:
    if each.scaredTimer == 0:
      ghostDistList = ghostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
    elif each.scaredTimer > 0:
      scaredGhostDistList = scaredGhostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
  minGhostDist = -1
  if len(ghostDistList) > 0:
    minGhostDist = min(ghostDistList)
  minScaredGhostDist = -1
  if len(scaredGhostDistList) > 0:
    minScaredGhostDist = min(scaredGhostDistList)
  # Evaluate score
  # score = scoreEvaluationFunction(currentGameState)
  score = 0
  # Distance to closest food
  score = score + (-1.5 * minFoodDist)
  # Distance to closest ghost
  score = score + (-2 * (1.0 / minGhostDist))
  # Distance to closest scared ghost
  score = score + (-2 * minScaredGhostDist)
  # Number of capsules
  score = score + (-20 * len(capsules))
  # Number of food
  score = score + (-4 * len(foods.asList()))
  return score
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
