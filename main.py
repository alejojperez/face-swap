import os
from faceswap import FaceSwap


if __name__ == '__main__':
    app = FaceSwap()

    invalid = True

    while invalid:
        option = input("Enter 'i' - for image swap, 'v' - for video swap: ")
        destinationFile = input("Enter absolute path for generated file: ")

        if "i" == option:
            invalid = False
            app.imageSwap(destinationFile)

        if "v" == option:
            invalid = False
            multiprocessing = input("Would you like to use multiprocessing? y/n: ") == "y"
            processes = None

            if multiprocessing:
                processes = int(input("How many processes? (default: all availables): ") or os.cpu_count())

            app.videoSwap(destinationFile, multiprocessing, processes)

        if invalid:
            print("Invalid option, please try again...")