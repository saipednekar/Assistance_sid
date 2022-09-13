#!/usr/bin/env python3
from pyfirmata import Arduino, util
from time import sleep

board = Arduino('/dev/ttyACM1') # Change to your port

l4= board.get_pin('d:4:o')
l5= board.get_pin('d:5:o')

# l2a= board.get_pin('d:6:o')
# l2b= board.get_pin('d:7:o')


l8= board.get_pin('d:8:o')
l9= board.get_pin('d:9:o')


l10= board.get_pin('d:10:o')
# l4b= board.get_pin('d:11:o')


# l5a= board.get_pin('d:12:o')
# l5b= board.get_pin('d:13:o')

print("Start blinking D13")



def led1():
    l4.write(1)
    l5.write(0)

    sleep(0.2)
    l8.write(0)
    l9.write(1)
    l10.write(1)


def led2():
    l4.write(1)
    l5.write(0)

    sleep(0.2)
    l8.write(1)
    l9.write(0)
    l10.write(1)


def led3():
    l4.write(1)
    l5.write(0)

    sleep(0.2)
    l8.write(1)
    l9.write(1)
    l10.write(0)


def led4():
    l5.write(1)
    l4.write(0)
    sleep(0.2)
    l8.write(0)
    l9.write(1)
    l10.write(1)


def led5():
    l5.write(1)
    l4.write(0)

    sleep(0.2)
    l8.write(1)
    l9.write(0)
    l10.write(1)


def led6():
    l5.write(1)
    l4.write(0)

    sleep(0.2)
    l8.write(1)
    l9.write(1)
    l10.write(0)

def tri():
    l4.write(1)
    l5.write(0)

    sleep(0.2)
    l8.write(0)
    l9.write(1)
    l10.write(1)


    l4.write(1)
    l5.write(0)

    sleep(0.2)
    l8.write(1)
    l9.write(0)
    l10.write(1)


    l4.write(1)
    l5.write(0)

    sleep(0.2)
    l8.write(1)
    l9.write(1)
    l10.write(0)


    # l5.write(1)
    # l4.write(0)
    # sleep(0.2)
    # l8.write(0)
    # l9.write(1)
    # l10.write(1)
    l10.write(1)


    l5.write(1)
    l4.write(0)

    sleep(0.2)
    l8.write(1)
    l9.write(0)
    l10.write(1)
    sleep(0.2)



    # l5.write(1)
    # l4.write(0)

    # sleep(0.2)
    # l8.write(1)
    # l9.write(1)
    # l10.write(0)

while True:
        tri()
