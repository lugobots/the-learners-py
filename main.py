import signal
import threading

from concurrent.futures import ThreadPoolExecutor
import lugo4py
import lugo4py.mapper as mapper
import lugo4py.rl as rl
import my_bot

from training_function import my_training_function

grpc_address = "localhost:5000"
grpc_insecure = True

stop = threading.Event()

if __name__ == "__main__":
    team_side = lugo4py.TeamSide.HOME
    print('main: Training bot team side = ', team_side)
    # The map will help us see the field in quadrants (called regions) instead of working with coordinates
    # The Mapper will translate the coordinates based on the side the bot is playing on
    my_mapper = mapper.Mapper(20, 10, lugo4py.TeamSide.HOME)

    # Our bot strategy defines our bot initial position based on its number
    initial_region = my_mapper.get_region(5, 4)

    # Now we can create the bot. We will use a shortcut to create the client from the config, but we could use the
    # client constructor as well
    lugo_client = lugo4py.LugoClient(
        grpc_address,
        grpc_insecure,
        "",
        team_side,
        my_bot.TRAINING_PLAYER_NUMBER,
        initial_region.get_center()
    )
    # The RemoteControl is a gRPC client that will connect to the Game Server and change the element positions
    rc = rl.RemoteControl()
    rc.connect(grpc_address)  # Pass address here

    bot = my_bot.MyBotTrainer(rc)

    gym_executor = ThreadPoolExecutor()
    # Now we can create the Gym, which will control all async work and allow us to focus on the learning part
    gym = rl.Gym(gym_executor, rc, bot, my_training_function, {"debugging_log": False})

    players_executor = ThreadPoolExecutor(22)
    # here we are using zombie players, but you may also use another bot or other helping players.
    # read the main Readme file to find more ways to run other bots.
    gym.with_zombie_players(grpc_address).start(lugo_client, players_executor)

    def signal_handler(_, __):
        print("Stop requested\n")
        lugo_client.stop()
        gym.stop()
        players_executor.shutdown(wait=True)
        gym_executor.shutdown(wait=True)


    signal.signal(signal.SIGINT, signal_handler)

    stop.wait()
