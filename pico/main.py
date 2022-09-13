from machine import Pin,UART,PWM
from time import sleep


uart = UART(0,9600)
pwm = PWM(Pin(2))

pwm.freq(50)




def degree(deg):
    mul_uart_degree=9600/270
    uart=int(deg)*round(mul_uart_degree)
    return uart
    
    

while True:
    
    if uart.any():
        val = uart.readline()
        d={
        b'a':30,
        b'b':40,
        b'c':50,
        b'd':60,
        b'e':70,
        b'f':80,
        b'g':90,
        b'h':100,
        b'i':110,
        b'j':120,
        b'k':130,
        b'l':140,
        b'm':150,
        b'n':160,
        b'o':170,
        b'p':180,
        b'q':190,
        b'r':200,
        b's':210,
        b't':220,
        b'u':230,
        b'v':240,
        }
        
        
        if val in d.keys():
            print(val)

            
            v=d[val]
            
    
    
            print("in",v)
        
            uart_value=degree(v)
            print(uart_value)

            pwm.duty_u16(uart_value)
            print("done")
            sleep(0.01)
    
            