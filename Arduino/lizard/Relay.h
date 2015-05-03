#ifndef Relay_h
#define Relay_h

#include "Arduino.h"

class Relay {
    public:
        Relay(int pin);
        void switchStatus();
        int getStatus();
        const int RED;
        const int YELLOW;
    private:
        int _pin;
        int _status;
};

#endif
