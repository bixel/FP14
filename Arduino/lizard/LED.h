#ifndef LED_h
#define LED_h

#include "Arduino.h"

class LED {
    public:
        LED(int pin, int rank=0);
        void on();
        void off();
        int getStatus();
        int getRank();
    private:
        int _pin;
        int _rank;
        int _status;
};

#endif

