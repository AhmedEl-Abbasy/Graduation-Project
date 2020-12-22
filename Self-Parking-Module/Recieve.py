##Recieve order from inside 
import numpy as np
#from first_module import order 
from Stop_mode import stop_order,send_SMS,alarm


def order_to_stop(input_1):
    if input_1 ==1:
       stop_order()
       send_SMS()
       alarm()

    



