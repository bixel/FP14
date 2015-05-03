#ifndef GamePhase_h
#define GamePhase_h

#include "Arduino.h"

class GamePhase {
    public:
        GamePhase();
        bool isRunning();
        void run();
    protected:
        int _timer;
        bool _running;
};

#endif
