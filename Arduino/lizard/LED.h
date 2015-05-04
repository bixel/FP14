#ifndef LED_h
#define LED_h

#include "Arduino.h"

class LED {
    public:
        LED(int pin);
        void on();
        void off();
        int getStatus();
    private:
        int _pin;
        int _status;
};

#endif

