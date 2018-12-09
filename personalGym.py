import coin_picker, gridworld, game2048, othello, connect4
import experiments.virtual_machine as virtual_machine

def make(game_title, tweak_param=None):
    if game_title == "CoinPicker-v0":
        return coin_picker.env()
    elif game_title == "CoinPickerPixels-v0":
        return coin_picker.env(True)
    elif game_title == "GridWorld-v0":
        return gridworld.chooseLevel(tweak_param)
    elif game_title == "2048-v0":
        return game2048.env()
    elif game_title == "Othello-v0":
        return othello.Env(tweak_param)
    elif game_title == "Connect4-v0":
        return connect4.Env(tweak_param)
    elif game_title == "QProgramming-v0":
        return virtual_machine.VM(tweak_param)
    else:
        raise EnvironmentError("The environment with that name doesn't exist")
