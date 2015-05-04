#ifndef GamePhase_h
#define GamePhase_h

#include "Arduino.h"
#include "Relay.h"
#include "Player.h"
#include "LED.h"

class GamePhase {
    public:
        GamePhase(Relay& relay,
                  Player& player1,
                  Player& player2,
                  LED& l1,
                  LED& l2,
                  LED& l3);
        bool isRunning();
        void run();
    protected:
        int _timer;
        bool _running;
        Relay _relay;
        Player _p1;
        Player _p2;
        LED _l1;
        LED _l2;
        LED _l3;
        LED _currentLED;
};

#endif

